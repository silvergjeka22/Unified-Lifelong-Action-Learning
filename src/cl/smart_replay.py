import torch
import torch.nn.functional as F


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
