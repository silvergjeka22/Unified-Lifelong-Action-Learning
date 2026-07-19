import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SmartReplayBuffer:
    """
    Prototype-aware replay buffer for continual learning.

    Instead of random FIFO sampling, stores per-class prototypes (mean + std)
    and keeps the most informative exemplars using one of three strategies:

        'hard'    — samples farthest from the class prototype (hardest examples)
        'diverse' — greedy coreset: maximises coverage in embedding space
        'random'  — baseline random selection

    Supports cross-domain mixing: UCF101 exemplars + YouTube AL-selected clips
    can coexist in the same buffer slot for a class.

    Usage:
        buf = SmartReplayBuffer(max_per_class=10)
        buf.add_class(class_idx, s_embs, t_embs, strategy="hard")
        s, t, y = buf.get_all()
        s_b, t_b, y_b = buf.sample_batch(32)
    """

    def __init__(self, max_per_class: int = 10):
        self.max_per_class = max_per_class
        self.s_store  = {}   # class_idx -> (N, D_s) student embeddings
        self.t_store  = {}   # class_idx -> (N, D_t) teacher embeddings
        self.mu_store = {}   # class_idx -> (D_s,)   class prototype

    # ── Adding / updating a class ─────────────────────────────────────────────
    def add_class(self, class_idx: int, s_embs: torch.Tensor,
                  t_embs: torch.Tensor = None, strategy: str = "hard"):
        """t_embs may be None before the KD stage — no teacher target exists yet."""
        if t_embs is None:
            t_embs = torch.zeros(s_embs.shape[0], 0)
        # flatten(1) keeps this shape-agnostic: works for (N, D) latents and for
        # (N, T, D) frame-feature sequences alike.
        flat  = s_embs.flatten(1).float()
        mu    = s_embs.mean(0)
        dists = (s_embs - mu).flatten(1).float().norm(dim=1)

        if strategy == "hard":
            idx = dists.argsort(descending=True)[:self.max_per_class]

        elif strategy == "diverse":
            # Greedy coreset: iteratively pick sample farthest from selected set
            selected  = [dists.argmax().item()]
            remaining = list(range(len(s_embs)))
            remaining.remove(selected[0])
            while len(selected) < min(self.max_per_class, len(s_embs)):
                d    = torch.cdist(flat[remaining], flat[selected]).min(1).values
                best = remaining[d.argmax().item()]
                selected.append(best)
                remaining.remove(best)
            idx = torch.tensor(selected)

        else:  # random
            idx = torch.randperm(len(s_embs))[:self.max_per_class]

        self.s_store[class_idx]  = s_embs[idx]
        self.t_store[class_idx]  = t_embs[idx]
        self.mu_store[class_idx] = mu

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def get_all(self):
        if not self.s_store:
            return None, None, None
        s = torch.cat(list(self.s_store.values()))
        t = torch.cat(list(self.t_store.values()))
        y = torch.cat([
            torch.full((v.shape[0],), k, dtype=torch.long)
            for k, v in self.s_store.items()
        ])
        return s, t, y

    def sample_batch(self, n: int):
        s, t, y = self.get_all()
        if s is None:
            return None, None, None
        idx = torch.randperm(len(s))[:n]
        return s[idx], t[idx], y[idx]

    def summary(self, class_names: list = None):
        print(f"SmartReplayBuffer — {len(self.s_store)} classes, "
              f"max_per_class={self.max_per_class}")
        for k, v in sorted(self.s_store.items()):
            name  = class_names[k] if class_names else str(k)
            kb    = v.element_size() * v.nelement() / 1e3
            print(f"  [{k:2d}] {name:<22}: {v.shape[0]} samples  "
                  f"shape={tuple(v.shape[1:])}  {kb:.0f} KB")


# ── Standalone training loop with smart replay ────────────────────────────────
def train_with_smart_replay(
    head,
    buffer: SmartReplayBuffer,
    new_s: torch.Tensor,
    new_t: torch.Tensor,
    new_y: torch.Tensor,
    val_s: torch.Tensor,
    val_y: torch.Tensor,
    device,
    epochs: int = 10,
    lr: float = 5e-4,
    wd: float = 0.03,
    batch_size: int = 32,
    ce_weight: float = 1.0,
    mse_weight: float = 0.5,
    label_smoothing: float = 0.1,
):
    """
    Fine-tune an EmbeddingHead on new-task embeddings mixed with buffer replay.
    Returns the best head (by val accuracy).
    """
    import copy
    import torch.nn as nn
    import torch.optim as optim

    mse_fn    = nn.MSELoss()
    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)
    head.to(device)

    best_acc, best_state = 0.0, copy.deepcopy(head.state_dict())

    for epoch in range(1, epochs + 1):
        head.train()
        # Mix new samples with buffer samples
        buf_s, buf_t, buf_y = buffer.sample_batch(len(new_s))
        if buf_s is not None:
            s = torch.cat([new_s, buf_s])
            t = torch.cat([new_t, buf_t])
            y = torch.cat([new_y, buf_y])
        else:
            s, t, y = new_s, new_t, new_y

        loader = DataLoader(TensorDataset(s, t, y), batch_size=batch_size, shuffle=True)
        total_loss = 0.0

        for sb, tb, yb in loader:
            sb, tb, yb = sb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, proj = head(sb, training=True)
            loss = (
                ce_weight  * F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
                + mse_weight * mse_fn(proj, tb)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        head.eval()
        with torch.no_grad():
            loader_v = DataLoader(TensorDataset(val_s, val_y), batch_size=128)
            correct = total = 0
            for sv, yv in loader_v:
                logits, _ = head(sv.to(device), training=False)
                correct += (logits.argmax(1) == yv.to(device)).sum().item()
                total   += yv.size(0)
            val_acc = correct / max(total, 1)

        print(f"[SmartReplay] Epoch {epoch:02d}/{epochs} | "
              f"Loss {total_loss/len(loader):.4f} | Val {val_acc:.2%}")

        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    print(f"  -> Best val acc: {best_acc:.2%}")
    return head
