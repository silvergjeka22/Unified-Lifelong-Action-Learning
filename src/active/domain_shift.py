"""
Domain shift utilities for Active Domain Adaptation.

Measures the distributional gap between UCF101 embeddings and
out-of-domain (e.g. YouTube) embeddings, and runs the Reptile
adaptation loop on AL-selected samples.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src.kd.utils import eval_head


# ── Distribution shift measurement ───────────────────────────────────────────
def compute_class_stats(embs: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Compute per-class mean (mu) and std (sigma) of embeddings.

    Returns:
        {class_idx: {"mu": tensor, "sigma": tensor}}
    """
    stats = {}
    for idx in labels.unique().tolist():
        e = embs[labels == idx]
        stats[int(idx)] = {"mu": e.mean(0), "sigma": e.std(0)}
    return stats


def measure_shift(ucf_stats: dict, yt_stats: dict, class_names: list = None) -> dict:
    """
    Report L2 distance between UCF101 and YouTube class prototype means.

    Returns:
        {class_idx: shift_distance (float)}
    """
    shifts = {}
    header = f"  {'Class':<22} | {'UCF std':>9} | {'YT std':>9} | {'Mean shift':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for idx in sorted(ucf_stats.keys()):
        if idx not in yt_stats:
            continue
        shift = (ucf_stats[idx]["mu"] - yt_stats[idx]["mu"]).norm().item()
        u_std = ucf_stats[idx]["sigma"].mean().item()
        y_std = yt_stats[idx]["sigma"].mean().item()
        name  = class_names[idx] if class_names else str(idx)
        print(f"  {name:<22} | {u_std:>9.4f} | {y_std:>9.4f} | {shift:>11.4f}")
        shifts[idx] = shift

    return shifts


# ── Reptile adaptation on AL-selected samples ─────────────────────────────────
def reptile_adapt(
    head,
    al_s: torch.Tensor,
    al_t: torch.Tensor,
    al_y: torch.Tensor,
    buffer,                 # SmartReplayBuffer or None
    device,
    episodes: int = 30,
    inner_lr: float = 5e-4,
    inner_steps: int = 10,
    epsilon: float = 0.20,
    mse_weight: float = 0.5,
    label_smoothing: float = 0.1,
    eval_s: torch.Tensor = None,
    eval_y: torch.Tensor = None,
    log_every: int = 10,
):
    """
    Reptile meta-adaptation on AL-selected YouTube embeddings.

    Each episode:
      1. Mix AL clips with a buffer sample (replay = no UCF101 forgetting)
      2. Run `inner_steps` SGD steps → adapted weights θ'
      3. Reptile update: θ ← θ + ε * (θ' - θ)

    Args:
        head        : TemporalHead to adapt (modified in-place, also returned)
        al_s / al_t : AL-selected student/teacher embeddings
        al_y        : labels for AL samples
        buffer      : SmartReplayBuffer (can be None)
        eval_s/y    : optional val embeddings for logging accuracy
    """
    head.to(device)
    mse_fn = nn.MSELoss()

    def snap(m):
        return {n: p.detach().clone() for n, p in m.named_parameters()}

    def reptile_update(m, W0, eps):
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n in W0:
                    p.copy_(W0[n] + eps * (p - W0[n]))

    print(f"Reptile adaptation: {episodes} episodes | "
          f"inner_lr={inner_lr} inner_steps={inner_steps} ε={epsilon}\n")

    for ep in range(1, episodes + 1):
        # Mix AL + buffer
        if buffer is not None:
            buf_s, buf_t, buf_y = buffer.sample_batch(min(len(al_s), 32))
        else:
            buf_s = buf_t = buf_y = None

        if buf_s is not None:
            cs = torch.cat([al_s, buf_s])
            ct = torch.cat([al_t, buf_t])
            cy = torch.cat([al_y, buf_y])
        else:
            cs, ct, cy = al_s, al_t, al_y

        cs, ct, cy = cs.to(device), ct.to(device), cy.to(device)

        W0  = snap(head)
        opt = optim.SGD(head.parameters(), lr=inner_lr, momentum=0.9, nesterov=True)

        head.train()
        for _ in range(inner_steps):
            opt.zero_grad(set_to_none=True)
            logits, proj = head(cs, training=True)
            loss = (
                F.cross_entropy(logits, cy, label_smoothing=label_smoothing)
                + mse_weight * mse_fn(proj, ct)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

        reptile_update(head, W0, epsilon)

        if ep % log_every == 0:
            yt_acc = eval_head(head, al_s, al_y, device) if eval_s is None else \
                     eval_head(head, eval_s, eval_y, device)
            print(f"  Episode {ep:3d}/{episodes} | YT acc: {yt_acc:.2%} | "
                  f"loss: {loss.item():.4f}")

    print("\n  Adaptation complete.")
    return head
