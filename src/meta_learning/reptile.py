import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.meta_learning.sampler import EmbeddingEpisodeSampler


def _snap(model: nn.Module) -> dict:
    """Snapshot current parameter values (detached clone)."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _reptile_update(model: nn.Module, W0: dict, epsilon: float) -> None:
    """
    Reptile outer update:  θ ← θ_0 + ε · (θ_inner − θ_0)
    Moves the meta-parameters a fraction ε towards the inner-loop solution.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in W0:
                p.copy_(W0[name] + epsilon * (p - W0[name]))


# ─────────────────────────────────────────────────────────
# VALIDATION HELPER
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def eval_head(
    head: nn.Module,
    s_embs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Compute classification accuracy of `head` over all (s_embs, labels) pairs.

    Args:
        head    : EmbeddingHead (or any module returning (logits, _) from s_emb)
        s_embs  : Student embedding matrix  [N, D]
        labels  : Ground-truth class indices [N]
        device  : Inference device

    Returns:
        Accuracy in [0, 1]
    """
    head.eval()
    loader = DataLoader(TensorDataset(s_embs, labels), batch_size=128, shuffle=False)
    correct = total = 0
    for s, y in loader:
        s, y = s.to(device), y.to(device)
        logits, _ = head(s, training=False)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────
# PHASE 1 — REPTILE META-TRAINING
# ─────────────────────────────────────────────────────────

def train_reptile(
    head: nn.Module,
    sampler: EmbeddingEpisodeSampler,
    val_s_embs: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    reptile_epochs: int,
    episodes_per_epoch: int,
    k_support: int,
    k_query: int,
    inner_lr: float,
    inner_steps: int,
    src_epsilon: float,
    ce_weight: float,
    mse_weight: float,
    reptile_patience: int,
) -> nn.Module:
    """
    Train `head` with the Reptile meta-learning algorithm.

    Each episode:
      1. Snapshot current weights W0.
      2. Run inner_steps of SGD on a random N-way K-shot support set
         using a combined CE + MSE (distillation) loss.
      3. Apply the Reptile outer update towards the inner-loop solution.

    Early-stopping is applied on validation accuracy with patience
    reptile_patience. The best checkpoint is restored before returning.

    Args:
        head                : EmbeddingHead to meta-train (modified in-place)
        sampler             : Pre-built EmbeddingEpisodeSampler
        val_s_embs          : Validation student embeddings
        val_labels          : Validation labels
        num_classes         : Total number of action classes (= n_way)
        device              : Training device
        reptile_epochs      : Number of outer meta-training epochs
        episodes_per_epoch  : Number of episodes sampled per epoch
        k_support           : Support set size per class
        k_query             : Query set size per class
        inner_lr            : Inner-loop SGD learning rate
        inner_steps         : Number of inner-loop gradient steps
        src_epsilon         : Reptile outer step size ε
        ce_weight           : Weight for cross-entropy loss term
        mse_weight          : Weight for MSE distillation loss term
        reptile_patience    : Early-stopping patience (epochs)

    Returns:
        head with best-validation-accuracy weights loaded
    """
    head.to(device)
    mse_fn = nn.MSELoss()
    n_way  = num_classes

    best_acc   = 0.0
    best_state = copy.deepcopy(head.state_dict())
    patience_counter = 0

    print("\n--- Running Phase 1: Reptile Meta-Learning Engine ---")

    for epoch in range(reptile_epochs):
        head.train()

        for _ in range(episodes_per_epoch):
            sup_s, sup_t, sup_y, _, _ = sampler.sample(n_way, k_support, k_query, device)

            W0 = _snap(head)

            inner_opt = optim.SGD(
                head.parameters(), lr=inner_lr, momentum=0.9, nesterov=True
            )
            for _ in range(inner_steps):
                inner_opt.zero_grad(set_to_none=True)
                logits, proj = head(sup_s, training=True)
                ce   = F.cross_entropy(logits, sup_y, label_smoothing=0.1)
                mse  = mse_fn(proj, sup_t)
                (ce_weight * ce + mse_weight * mse).backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                inner_opt.step()

            _reptile_update(head, W0, src_epsilon)

        val_acc = eval_head(head, val_s_embs, val_labels, device)

        if val_acc > best_acc:
            best_acc     = val_acc
            best_state   = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= reptile_patience:
                print(f"  [Meta-EarlyStop] Triggered at Epoch {epoch + 1}")
                break

    head.load_state_dict(best_state)
    return head