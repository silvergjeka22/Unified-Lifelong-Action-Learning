"""
Synthetic domain shift for the domain-adaptation study. A domain shift is applied directly to a
preprocessed clip tensor (no external data): box-blur plus lower contrast plus a brightness offset
stand in for a different camera / lighting / codec. Training on shifted clips makes the model robust
to a real domain change (e.g. a YouTube video), which is the adaptation we test.
"""

import torch
import torch.nn.functional as F


def domain_shift(clip, severity=0.8, noise=0.0):
    """Return a domain-shifted copy of a normalized clip (T, C, H, W): box-blur + lower contrast +
    brightness offset. severity in 0..1 scales the effect; noise > 0 adds gaussian noise (train
    augmentation only)."""
    k          = 3 if severity < 0.5 else 5
    blurred    = F.avg_pool2d(clip, kernel_size=k, stride=1, padding=k // 2)
    out        = blurred * (1.0 - 0.4 * severity) + 0.5 * severity
    if noise > 0:
        out = out + torch.randn_like(out) * noise
    return out


def shift_train(clip):
    """Random-strength domain shift for TRAIN augmentation: a clip is shifted by a random severity
    plus a little noise, so the model learns to be robust to appearance changes."""
    severity = 0.3 + 0.6 * torch.rand(1).item()
    return domain_shift(clip, severity=severity, noise=0.15)
