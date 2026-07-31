"""
GIL GAN — Generative Iterative Learning components.

Paper: "Continual Learning Improves Zero-Shot Action Recognition"
       Gowda, Moltisanti & Sevilla-Lara — arXiv 2410.10497
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Default dimensions (match paper + project setup)
FEAT_DIM   = 256   # ResNet50+LSTM hidden state output (hidden_size=256 checkpoint)
SEM_DIM    = 384   # sentence-transformers all-MiniLM-L6-v2
NOISE_DIM  = 256   # Generator noise input
HIDDEN     = 4096  # GAN hidden layers (from paper)


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


# ── GAN Training on REAL per-clip features (corrected) ───────────────────────
def train_gan_real(gen: FeatureGenerator, D: Discriminator, H: ProjectionHead,
                   feats: torch.Tensor, labels: torch.Tensor, class_stats: dict,
                   device, epochs: int = 30, lam1: float = 0.01, lam2: float = 0.1,
                   alpha: float = 10.0, lr: float = 1e-4, batch_size: int = 64,
                   n_critic: int = 5):
    """
    WGAN-GP on REAL per-clip features. Returns history {d_loss, g_loss, cls_loss}.

    The discriminator sees the real per-clip features (not the class mean), so the
    generator must match the real spread, not collapse to x_hat = mu.

    Args:
        feats       : (N, feat_dim) real per-clip features
        labels      : (N,) integer class labels
        class_stats : {label: {"mu": (D,), "sigma": (D,), "sem": (S,)}}
    """
    rows        = sorted(class_stats.keys())
    row_of      = {lab: i for i, lab in enumerate(rows)}
    num_classes = len(rows)
    feat_dim    = feats.shape[1]

    mu_all  = torch.stack([class_stats[int(l)]["mu"]    for l in labels]).to(device)
    sig_all = torch.stack([class_stats[int(l)]["sigma"] for l in labels]).to(device)
    sem_all = torch.stack([class_stats[int(l)]["sem"]   for l in labels]).to(device)
    x_all   = feats.to(device)
    lab_all = torch.tensor([row_of[int(l)] for l in labels], dtype=torch.long).to(device)

    aux_cls = nn.Linear(feat_dim, num_classes).to(device)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_F = torch.optim.Adam(
        list(gen.parameters()) + list(H.parameters()) + list(aux_cls.parameters()),
        lr=lr, betas=(0.5, 0.9),
    )

    loader = DataLoader(
        TensorDataset(x_all, mu_all, sig_all, sem_all, lab_all),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    history   = {"d_loss": [], "g_loss": [], "cls_loss": []}
    noise_dim = gen.noise_dim

    for epoch in range(1, epochs + 1):
        gen.train(); D.train(); H.train()
        ed = eg = ec = 0.0
        nb = 0
        for x_real, mu_b, sig_b, sem_b, lab_b in loader:
            B = x_real.size(0)

            for _ in range(n_critic):
                z      = torch.randn(B, noise_dim, device=device)
                x_hat  = gen(mu_b, sig_b, z).detach()
                gp     = gradient_penalty(D, x_real, x_hat, sem_b, device)
                loss_D = D(x_hat, sem_b).mean() - D(x_real, sem_b).mean() + alpha * gp
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            z        = torch.randn(B, noise_dim, device=device)
            x_hat    = gen(mu_b, sig_b, z)
            adv_loss = -D(x_hat, sem_b).mean()
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

    for p in gen.parameters():
        p.requires_grad_(False)
    print("FeatureGenerator frozen.")
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
