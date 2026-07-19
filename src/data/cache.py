"""
Frame-feature cache — the interface between Section 2 and everything after it.

Section 2 runs the frozen ResNet once and writes (N, T, backbone_dim) fp16 tensors to
Drive, one per task/split. Every later section loads that cache instead of touching
video again:

    raw clip (fp32)       16 x 3 x 224 x 224 x 4  =  9.63 MB
    frame features (fp16) 16 x 2048 x 2           =  65.5 KB    147x smaller

Splits are kept SEPARATE — train fits, val selects, test reports. (The older
extract_class_features() swept train+val+test into one tensor, which is why the GAN
notebook's "seen accuracy" was really training accuracy.)
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader


# ── Label space ───────────────────────────────────────────────────────────────
def build_label_space(task_classes: list):
    """
    Flatten a list of per-task class lists into one global label space.

    A class keeps the same integer in every notebook, which is what makes the
    per-task accuracy matrix meaningful.

    Returns (global_classes, global_c2i, num_after) where num_after[t] is the head
    width required after task t.
    """
    global_classes = [c for group in task_classes for c in group]
    global_c2i     = {c: i for i, c in enumerate(global_classes)}
    num_after      = np.cumsum([len(g) for g in task_classes]).tolist()
    return global_classes, global_c2i, num_after


def describe_label_space(task_classes, task_names, num_after, task_3=None):
    for t, name in enumerate(task_names):
        lo = num_after[t] - len(task_classes[t])
        print(f"{name:<6}: labels {lo:2d}-{num_after[t]-1:2d}  ({len(task_classes[t])} classes)")
    if task_3:
        print(f"\nTASK_3 ({len(task_3)}) held out for zero-shot — never trained on.")
    if len(task_classes) > 1 and len(task_classes[1]) < 10:
        print(f"\n>>> TASK_1 has {len(task_classes[1])}/10 classes — 7 are commented out")
        print(">>> in src/config/config.py. Uncomment for the full 50-class design.")


# ── Clip loaders (Section 2 only — everything else uses the cache) ────────────
def build_clip_loader(root, classes, global_c2i, split, batch_size=8, num_workers=2,
                      pin_memory=False):
    """DataLoader over preprocessed .pt clips for one task/split."""
    from src.data.dataset import UCF101Clips

    c2i = {c: global_c2i[c] for c in classes}
    ds  = UCF101Clips(os.path.join(root, split), class_to_idx=c2i)
    ld  = DataLoader(ds, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=pin_memory)
    return ds, ld


# ── Cache I/O ─────────────────────────────────────────────────────────────────
def save_feature_cache(path, feats: dict, lstm_state: dict, backbone_dim: int,
                       clip_len: int, global_classes: list, task_names: list):
    """Write the cache plus the Task-0 LSTM weights the freeze study needs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "feats":          feats,
        "lstm_state":     lstm_state,
        "backbone_dim":   backbone_dim,
        "clip_len":       clip_len,
        "global_classes": global_classes,
        "task_names":     task_names,
    }, path)
    mb = sum(f.element_size() * f.nelement() for f, _ in feats.values()) / 1e6
    print(f"Saved {mb:.0f} MB -> {path}")


def load_feature_cache(path, to_float32: bool = True, verbose: bool = True):
    """
    Load the cache. Returns (CACHE, lstm_state, meta).

    Stored fp16 to halve the file; cast to fp32 in RAM because the LSTM needs fp32.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No feature cache at {path}\n"
            "Run 2_extract_features/2.0_cache_frame_features.ipynb first."
        )

    blob  = torch.load(path, map_location="cpu")
    feats = blob["feats"]
    cache = {k: (f.float(), y) for k, (f, y) in feats.items()} if to_float32 else feats
    meta  = {k: v for k, v in blob.items() if k not in ("feats", "lstm_state")}

    if verbose:
        mb = sum(f.element_size() * f.nelement() for f, _ in cache.values()) / 1e6
        any_f = next(iter(cache.values()))[0]
        print(f"Loaded {len(cache)} tensors, {mb:.0f} MB in RAM")
        print(f"  per clip: ({any_f.shape[1]}, {any_f.shape[2]})")

    return cache, blob["lstm_state"], meta


# ── Slicing ───────────────────────────────────────────────────────────────────
def cat_upto(cache: dict, t: int, split: str):
    """Concatenate tasks 0..t for one split — the 'all classes seen so far' view."""
    fs = [cache[f"t{i}_{split}"][0] for i in range(t + 1) if f"t{i}_{split}" in cache]
    ys = [cache[f"t{i}_{split}"][1] for i in range(t + 1) if f"t{i}_{split}" in cache]
    if not fs:
        raise KeyError(f"No cached tensors for tasks 0..{t}, split '{split}'")
    return torch.cat(fs), torch.cat(ys)


def task_split(cache: dict, t: int, split: str):
    return cache[f"t{t}_{split}"]


# ── Validation ────────────────────────────────────────────────────────────────
def verify_cache(cache: dict, task_classes: list, global_c2i: dict,
                 backbone_dim: int, clip_len: int = None):
    """Assert shapes are right and no task's labels leak into another's tensors."""
    for k, (f, _) in cache.items():
        assert f.ndim == 3, f"{k}: expected (N, T, D), got {tuple(f.shape)}"
        assert f.shape[2] == backbone_dim, f"{k}: dim {f.shape[2]} != {backbone_dim}"
        if clip_len:
            assert f.shape[1] == clip_len, f"{k}: T={f.shape[1]} != {clip_len}"

    for t in range(len(task_classes)):
        expected = {global_c2i[c] for c in task_classes[t]}
        for split in ("train", "val", "test"):
            k = f"t{t}_{split}"
            if k in cache:
                got = set(cache[k][1].tolist())
                assert got <= expected, f"{k}: labels outside its task: {got - expected}"

    n_clips  = sum(f.shape[0] for f, _ in cache.values())
    cache_mb = sum(f.element_size() * f.nelement() for f, _ in cache.values()) / 1e6
    raw_mb   = n_clips * (clip_len or 16) * 3 * 224 * 224 * 4 / 1e6
    print(f"OK — {n_clips} clips, tasks disjoint, train/val/test separate.")
    print(f"raw video would be {raw_mb:,.0f} MB   cache is {cache_mb:.0f} MB "
          f"({raw_mb / max(cache_mb, 1e-9):.0f}x smaller)")
    return n_clips
