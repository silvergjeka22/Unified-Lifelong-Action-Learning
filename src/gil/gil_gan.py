"""
GIL GAN — Generative Iterative Learning components.

Paper: "Continual Learning Improves Zero-Shot Action Recognition"
       Gowda, Moltisanti & Sevilla-Lara — arXiv 2410.10497
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# Default dimensions (match paper + project setup)
FEAT_DIM   = 256   # ResNet50+LSTM hidden state output (hidden_size=256 checkpoint)
SEM_DIM    = 384   # sentence-transformers all-MiniLM-L6-v2
LATENT_DIM = 256   # CVAE latent space
NOISE_DIM  = 256   # Generator noise input
HIDDEN     = 4096  # GAN hidden layers (from paper)


# ── CVAE (E) ──────────────────────────────────────────────────────────────────
class CVAE(nn.Module):
    """
    Conditional VAE: semantic embedding a(384) → prototype (μ̂, σ̂)(512 each).

    Encoder: a → Linear(2048) → Linear(1024) → μ_z, log_σ_z(512)
    Decoder: z(512) → Linear(1024) → Linear(2048) → (μ̂||σ̂)(1024)
    """

    def __init__(self, sem_dim=SEM_DIM, feat_dim=FEAT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc_hidden = nn.Sequential(
            nn.Linear(sem_dim, 2048), nn.ReLU(),
            nn.Linear(2048, 1024),   nn.ReLU(),
        )
        self.enc_mu      = nn.Linear(1024, latent_dim)
        self.enc_log_var = nn.Linear(1024, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 2048),       nn.ReLU(),
            nn.Linear(2048, feat_dim * 2),
        )

    def encode(self, a):
        h = self.enc_hidden(a)
        return self.enc_mu(h), self.enc_log_var(h)

    def reparameterise(self, mu, log_var):
        return mu + torch.randn_like(mu) * (0.5 * log_var).exp()

    def forward(self, a):
        mu_z, log_var = self.encode(a)
        z = self.reparameterise(mu_z, log_var)
        out = self.dec(z)
        mu_hat, sigma_hat = out.chunk(2, dim=-1)
        return mu_hat, sigma_hat, mu_z, log_var

    @staticmethod
    def loss(mu_hat, sigma_hat, mu_real, sigma_real, mu_z, log_var, beta=1.0):
        recon = F.mse_loss(
            torch.cat([mu_hat, sigma_hat], dim=-1),
            torch.cat([mu_real, sigma_real], dim=-1),
        )
        kl = -0.5 * (1 + log_var - mu_z.pow(2) - log_var.exp()).mean()
        return recon + beta * kl


# ── Feature Generator (F) ─────────────────────────────────────────────────────
class FeatureGenerator(nn.Module):
    """
    Generator: concat(μ, σ, z) → synthesised feature x̂ ∈ R^feat_dim.

    Input  : μ(512) || σ(512) || z~N(0,I)(512)  → 1536
    Hidden : Linear(4096) → ReLU → Linear(4096) → ReLU
    Output : Linear(feat_dim=512)
    """

    def __init__(self, feat_dim=FEAT_DIM, noise_dim=NOISE_DIM, hidden=HIDDEN):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim + feat_dim + noise_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                          nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, mu, sigma, z=None):
        if z is None:
            z = torch.randn(mu.size(0), self.noise_dim, device=mu.device)
        return self.net(torch.cat([mu, sigma, z], dim=-1))


# ── Discriminator (G) ─────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    WGAN-GP discriminator: concat(x, a) → unbounded scalar.

    Input  : x(512) || a(384)  → 896
    Hidden : Linear(4096) → LeakyReLU → Linear(4096) → LeakyReLU
    Output : Linear(1)
    """

    def __init__(self, feat_dim=FEAT_DIM, sem_dim=SEM_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + sem_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),             nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))


# ── Projection Head (H) ───────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    Projects visual feature x back to semantic embedding space.
    Used in L_MI = -cosine_similarity(H(F(a,z)), a).
    """

    def __init__(self, feat_dim=FEAT_DIM, sem_dim=SEM_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, sem_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── GIL Classifier (2-layer MLP head) ────────────────────────────────────────
class GILClassifier(nn.Module):
    """
    Lightweight 2-layer MLP that sits on top of the frozen backbone.
    This is the "last 2 FC layers of M" that get fine-tuned during
    incremental learning and zero-shot adaptation.
    """

    def __init__(self, feat_dim=FEAT_DIM, num_classes=10, dropout_p=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def expand(self, new_num_classes):
        """Returns a new GILClassifier with an expanded head, copying shared weights."""
        old_out = self.net[-1].out_features
        if new_num_classes <= old_out:
            return self
        new_cls = GILClassifier(
            feat_dim=self.net[0].in_features,
            num_classes=new_num_classes,
            dropout_p=self.net[2].p,
        )
        # Copy existing weights
        with torch.no_grad():
            new_cls.net[0].weight.copy_(self.net[0].weight)
            new_cls.net[0].bias.copy_(self.net[0].bias)
            new_cls.net[-1].weight[:old_out].copy_(self.net[-1].weight)
            new_cls.net[-1].bias[:old_out].copy_(self.net[-1].bias)
        return new_cls


# ── GIL Replay Buffer ─────────────────────────────────────────────────────────
class GILReplayBuffer:
    """
    Stores per-class prototype statistics: {class_name → (μ, σ, sem_emb, label)}.

    Each call to generate_all() uses the frozen generator F to synthesise
    J feature vectors per buffered class.
    """

    def __init__(self):
        self._store     = {}   # class_name → {"mu", "sigma", "sem"}
        self._class_idx = {}   # class_name → int label

    def add(self, class_name: str, mu: torch.Tensor, sigma: torch.Tensor,
            sem_emb: torch.Tensor, label: int):
        self._store[class_name]     = {
            "mu":    mu.detach().cpu(),
            "sigma": sigma.detach().cpu(),
            "sem":   sem_emb.detach().cpu(),
        }
        self._class_idx[class_name] = label

    def generate_all(self, F: FeatureGenerator, J: int, device):
        """Generates J synthetic features per buffered class. Returns (feats, labels)."""
        F.eval()
        all_feats, all_labels = [], []
        with torch.no_grad():
            for name, entry in self._store.items():
                mu    = entry["mu"].unsqueeze(0).expand(J, -1).to(device)
                sigma = entry["sigma"].unsqueeze(0).expand(J, -1).to(device)
                x_hat = F(mu, sigma)
                all_feats.append(x_hat.cpu())
                all_labels.extend([self._class_idx[name]] * J)
        F.train()
        if not all_feats:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        return (
            torch.cat(all_feats, dim=0),
            torch.tensor(all_labels, dtype=torch.long),
        )

    def items(self):
        return self._store.items()

    @property
    def class_list(self):
        return list(self._store.keys())

    def label_of(self, class_name):
        return self._class_idx.get(class_name, -1)

    def __len__(self):
        return len(self._store)

    def __contains__(self, name):
        return name in self._store


# ── WGAN-GP Gradient Penalty ──────────────────────────────────────────────────
def gradient_penalty(D: Discriminator, real: torch.Tensor, fake: torch.Tensor,
                     a: torch.Tensor, device):
    """Computes WGAN-GP penalty on linear interpolations of real and fake samples."""
    B   = real.size(0)
    eps = torch.rand(B, 1, device=device)
    interp = (eps * real.detach() + (1 - eps) * fake.detach()).requires_grad_(True)
    d_int  = D(interp, a)
    grads  = torch.autograd.grad(
        d_int, interp,
        grad_outputs=torch.ones_like(d_int),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# ── GAN Training (Phase 1, Part 2) ───────────────────────────────────────────
def train_gan(gen: FeatureGenerator, G: Discriminator, H: ProjectionHead,
              buffer: GILReplayBuffer, class_to_label: dict, device,
              epochs: int = 30, lam1: float = 0.01, lam2: float = 0.1,
              alpha: float = 10.0, lr: float = 1e-4, batch_size: int = 64,
              n_critic: int = 5):
    """
    WGAN-GP training loop.

    L_D = E[G(fake,a)] - E[G(real,a)] + alpha * gradient_penalty  (minimise)
    L_F = -E[G(fake,a)] + lam1*L_CLS + lam2*L_MI                 (minimise)

    After training, gen is frozen permanently.
    Returns history dict with keys: d_loss, g_loss, cls_loss.
    """
    if len(buffer) == 0:
        raise ValueError("Buffer is empty — run feature extraction first.")

    # Build tensors from buffer prototypes
    mu_list, sig_list, sem_list, lab_list = [], [], [], []
    for name, entry in buffer.items():
        mu_list.append(entry["mu"])
        sig_list.append(entry["sigma"])
        sem_list.append(entry["sem"])
        lab_list.append(class_to_label[name])

    mu_t  = torch.stack(mu_list).to(device)
    sig_t = torch.stack(sig_list).to(device)
    sem_t = torch.stack(sem_list).to(device)
    lab_t = torch.tensor(lab_list, dtype=torch.long).to(device)

    num_classes = len(buffer)
    aux_cls = nn.Linear(FEAT_DIM, num_classes).to(device)

    opt_D = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_F = torch.optim.Adam(
        list(gen.parameters()) + list(H.parameters()) + list(aux_cls.parameters()),
        lr=lr, betas=(0.5, 0.9),
    )

    n      = len(mu_t)
    loader = DataLoader(
        TensorDataset(mu_t, sig_t, sem_t, lab_t),
        batch_size=min(batch_size, n),
        shuffle=True,
        drop_last=(n > batch_size),
    )

    history = {"d_loss": [], "g_loss": [], "cls_loss": []}

    for epoch in range(1, epochs + 1):
        gen.train(); G.train(); H.train()
        ed, eg, ec, nb = 0.0, 0.0, 0.0, 0

        for mu_b, sig_b, sem_b, lab_b in loader:
            B = mu_b.size(0)

            # ── Discriminator update ──────────────────────────────────────
            for _ in range(n_critic):
                z      = torch.randn(B, NOISE_DIM, device=device)
                x_hat  = gen(mu_b, sig_b, z).detach()
                gp     = gradient_penalty(G, mu_b, x_hat, sem_b, device)
                loss_D = G(x_hat, sem_b).mean() - G(mu_b, sem_b).mean() + alpha * gp
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # ── Generator / Head / Aux-classifier update ──────────────────
            z        = torch.randn(B, NOISE_DIM, device=device)
            x_hat    = gen(mu_b, sig_b, z)
            adv_loss = -G(x_hat, sem_b).mean()
            cls_loss = F.cross_entropy(aux_cls(x_hat), lab_b)
            mi_loss  = -F.cosine_similarity(H(x_hat), sem_b).mean()
            loss_F   = adv_loss + lam1 * cls_loss + lam2 * mi_loss
            opt_F.zero_grad()
            loss_F.backward()
            opt_F.step()

            ed += loss_D.item()
            eg += adv_loss.item()
            ec += cls_loss.item()
            nb += 1

        history["d_loss"].append(ed / nb)
        history["g_loss"].append(eg / nb)
        history["cls_loss"].append(ec / nb)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[GAN] Epoch {epoch:02d}/{epochs}  "
                  f"D: {history['d_loss'][-1]:+.4f}  "
                  f"G: {history['g_loss'][-1]:+.4f}  "
                  f"CLS: {history['cls_loss'][-1]:.4f}")

    # Freeze gen permanently after Phase 1
    for p in gen.parameters():
        p.requires_grad_(False)
    print("FeatureGenerator frozen permanently.")

    return history


# ── Fine-tune Last 2 FC Layers ────────────────────────────────────────────────
def fine_tune_last2(classifier: GILClassifier, mixed_loader,
                    epochs: int, lr: float, device):
    """
    Fine-tunes the GILClassifier (2-layer MLP head) on pre-extracted
    512-dim feature tensors. The backbone stays frozen.

    mixed_loader yields (h_last: Tensor[B, feat_dim], labels: Tensor[B]).
    """
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    classifier.to(device).train()

    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for feats, labels in mixed_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = classifier(feats)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [FT] Epoch {epoch}/{epochs}  "
                  f"Loss {total_loss / len(mixed_loader):.4f}  "
                  f"Acc {correct / max(total, 1):.2%}")

    classifier.eval()
    return classifier


# ── Phase 3: Update Buffer and Fine-tune CVAE ─────────────────────────────────
def update_buffer_and_cvae(model, E: CVAE, new_classes: list,
                            buffer: GILReplayBuffer, class_feats: dict,
                            sem_embs: dict, class_to_label: dict,
                            optimizer_E: torch.optim.Optimizer, device,
                            cvae_epochs: int = 5):
    """
    After each incremental step:
    1. Computes (μ, σ) for new classes and adds them to the buffer.
    2. Fine-tunes E on all buffered classes so it can map new semantics → prototypes.
    """
    model.eval()
    E.to(device).train()

    for cls_name in new_classes:
        feats = class_feats[cls_name].to(device)
        mu    = feats.mean(0).detach()
        sigma = feats.std(0).clamp_min(1e-6).detach()
        sem   = sem_embs[cls_name].to(device)
        buffer.add(cls_name, mu, sigma, sem, class_to_label[cls_name])

    for _ in range(cvae_epochs):
        for cls_name, entry in buffer.items():
            sem_b   = entry["sem"].unsqueeze(0).to(device)
            mu_b    = entry["mu"].unsqueeze(0).to(device)
            sigma_b = entry["sigma"].unsqueeze(0).to(device)
            mu_hat, sigma_hat, mu_z, log_var = E(sem_b)
            loss = CVAE.loss(mu_hat, sigma_hat, mu_b, sigma_b, mu_z, log_var)
            optimizer_E.zero_grad()
            loss.backward()
            optimizer_E.step()

    E.eval()


# ── Zero-Shot Testing ─────────────────────────────────────────────────────────
def zero_shot_test(model, gen: FeatureGenerator, E: CVAE,
                   unseen_classes: list, sem_embs: dict, class_to_label: dict,
                   test_loader, device, J: int = 50):
    """
    Zero-shot evaluation pipeline:
    1. For each unseen class: sem_emb → E → (μ̂, σ̂) → gen × J → synthetic feats.
    2. Fine-tune a fresh GILClassifier on these synthetic features.
    3. 1-NN against synthetic class centroids at test time.

    Returns (top1_acc, all_preds, all_true).
    """
    E.eval()
    gen.eval()

    num_unseen = len(unseen_classes)
    synth_feats, synth_labels, centroids = [], [], {}

    with torch.no_grad():
        for cls_name in unseen_classes:
            a = sem_embs[cls_name].unsqueeze(0).to(device)
            mu_hat, sigma_hat, _, _ = E(a)
            mu_hat    = mu_hat.expand(J, -1)
            sigma_hat = sigma_hat.expand(J, -1)
            x_hat = gen(mu_hat, sigma_hat)
            synth_feats.append(x_hat.cpu())
            lbl = class_to_label[cls_name]
            synth_labels.extend([lbl] * J)
            centroids[lbl] = x_hat.mean(0).cpu()

    synth_feats  = torch.cat(synth_feats, dim=0)
    synth_labels = torch.tensor(synth_labels, dtype=torch.long)

    # Remap labels to 0…num_unseen-1 for the ZSL classifier
    zsl_label_list = [class_to_label[c] for c in unseen_classes]
    global_to_local = {g: l for l, g in enumerate(zsl_label_list)}
    local_labels = torch.tensor(
        [global_to_local[lb.item()] for lb in synth_labels], dtype=torch.long
    )

    zsl_clf = GILClassifier(feat_dim=FEAT_DIM, num_classes=num_unseen).to(device)
    zsl_loader = DataLoader(
        TensorDataset(synth_feats, local_labels),
        batch_size=64, shuffle=True,
    )
    fine_tune_last2(zsl_clf, zsl_loader, epochs=5, lr=1e-4, device=device)

    # 1-NN using synthetic centroids
    centroid_mat = torch.stack([centroids[k] for k in zsl_label_list])

    model.eval()
    correct, total = 0, 0
    all_preds, all_true = [], []

    with torch.no_grad():
        for clips, labels in tqdm(test_loader, desc="ZSL 1-NN Eval"):
            _, h = model(clips.to(device))
            h    = h.cpu()
            dists = torch.cdist(h, centroid_mat)
            nn_idx = dists.argmin(dim=1)
            preds  = torch.tensor([zsl_label_list[i] for i in nn_idx.tolist()])
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.tolist())
            all_true.extend(labels.tolist())

    acc = correct / max(total, 1)
    print(f"ZSL Top-1: {acc:.2%}  ({correct}/{total})")
    return acc, all_preds, all_true


# ── Ablation Baseline ─────────────────────────────────────────────────────────
def run_incremental_baseline(
    strategy: str,
    seen_classes: list,
    base_class_feats: dict,
    all_seen_feats: dict,
    buffer_ref: GILReplayBuffer,
    F_gen: FeatureGenerator,
    known_label_local: dict,
    label_to_class: dict,
    device,
    ft_epochs: int = 5,
    ft_lr: float = 1e-4,
):
    """
    Runs the incremental loop with a given replay strategy for ablation.

    strategy: 'none' | 'random' | 'gil'
    Returns list of base-class accuracies per iteration.
    """
    import src.config.config as _cfg
    from src.utils.train import chunks as _chunks

    num_base    = len(_cfg.SELECTED_CLASSES)
    num_known   = num_base + len(seen_classes)
    clf         = GILClassifier(feat_dim=FEAT_DIM, num_classes=num_known).to(device)
    chunk_size  = max(1, len(seen_classes) // 10)
    accs        = []

    for class_chunk in _chunks(seen_classes, chunk_size):
        real_feats  = torch.cat([all_seen_feats[c] for c in class_chunk if c in all_seen_feats])
        real_labels = torch.cat([
            torch.full((all_seen_feats[c].shape[0],), known_label_local[c], dtype=torch.long)
            for c in class_chunk if c in all_seen_feats
        ])

        if strategy == "none":
            mixed_feats, mixed_labels = real_feats, real_labels

        elif strategy == "random":
            J_iter  = max(1, len(real_feats) // max(len(buffer_ref), 1))
            n_synth = J_iter * len(buffer_ref)
            synth_f = torch.randn(n_synth, FEAT_DIM)
            buf_labels = []
            for cls_name in buffer_ref.class_list:
                buf_labels.extend([known_label_local[cls_name]] * J_iter)
            synth_l = torch.tensor(buf_labels[:n_synth], dtype=torch.long)
            mixed_feats  = torch.cat([real_feats, synth_f], dim=0)
            mixed_labels = torch.cat([real_labels, synth_l], dim=0)

        else:  # 'gil'
            J_iter = max(1, len(real_feats) // max(len(buffer_ref), 1))
            synth_f, synth_g = buffer_ref.generate_all(F_gen, J=J_iter, device=device)
            if len(synth_f) > 0:
                synth_l = torch.tensor(
                    [known_label_local[label_to_class[lb.item()]] for lb in synth_g],
                    dtype=torch.long,
                )
                mixed_feats  = torch.cat([real_feats, synth_f], dim=0)
                mixed_labels = torch.cat([real_labels, synth_l], dim=0)
            else:
                mixed_feats, mixed_labels = real_feats, real_labels

        loader = DataLoader(TensorDataset(mixed_feats, mixed_labels), batch_size=64, shuffle=True)
        fine_tune_last2(clf, loader, epochs=ft_epochs, lr=ft_lr, device=device)

        clf.eval()
        val_feats  = torch.cat(list(base_class_feats.values()), dim=0)
        val_labels = torch.cat([
            torch.full((base_class_feats[c].shape[0],), known_label_local[c], dtype=torch.long)
            for c in _cfg.SELECTED_CLASSES if c in base_class_feats
        ])
        with torch.no_grad():
            acc = (clf(val_feats.to(device)).argmax(1).cpu() == val_labels).float().mean().item()
        accs.append(acc)

    return accs
