"""
Evaluation helpers shared by every continual-learning notebook.

Two evaluation paths on the same trained head:

    embeddings   fast, uses the cached (T, 2048) features
    real images  the honest end-to-end check: frozen ResNet + trained LSTM + head
                 run on raw video clips

Both must agree. If they do not, the cache and the raw clips have drifted apart.
"""

import os

import numpy as np
import torch
import torch.nn as nn

from src.utils.probe import to_numpy
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset


# ── Label space ───────────────────────────────────────────────────────────────
def build_class_index(task_classes: list):
    """
    Flat class list and name -> global index, matching the notebook convention.

    Works for any number of tasks: [10] + [3, 3] today, [30] + [10, 10, 10] later.
    """
    all_classes  = [c for group in task_classes for c in group]
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    return all_classes, class_to_idx


def check_cache_labels(meta: dict, all_classes: list):
    """
    The cache stores the label space it was built with. If the notebook now uses a
    different one, every label silently points at the wrong class.
    """
    cached = meta.get("global_classes")
    if not cached:
        return True
    n = len(all_classes)
    if cached[:n] != all_classes:
        raise RuntimeError(
            "Label space does not match the feature cache.\n"
            f"  cache: {cached[:n]}\n"
            f"  now  : {all_classes}\n"
            "Re-run Section 2 after changing the class configuration."
        )
    return True


# ── Embedding loaders ─────────────────────────────────────────────────────────
def emb_dataset(cache: dict, t: int, split: str = "test"):
    f, y = cache[f"t{t}_{split}"]
    return TensorDataset(f, y)


def emb_loaders(cache: dict, task_ids: list, split: str = "test", batch_size: int = 64):
    """Per-task loaders plus one loader over all of them concatenated."""
    dsets   = [emb_dataset(cache, t, split) for t in task_ids if f"t{t}_{split}" in cache]
    per     = [DataLoader(d, batch_size=batch_size, shuffle=False) for d in dsets]
    combined = DataLoader(ConcatDataset(dsets), batch_size=batch_size, shuffle=False)
    return per, combined


# ── Full model: raw clips -> logits ───────────────────────────────────────────
class FullModel(nn.Module):
    """
    Frozen CNN backbone + the trained TemporalHead, as one model over raw clips.

    The head was trained on cached (T, backbone_dim) features. This puts the backbone
    back in front of it so the same weights can be evaluated on real video, which is
    what the deployed model would actually see.

    Input : (B, T, C, H, W)
    Output: logits (B, num_classes)
    """

    def __init__(self, backbone, head, backbone_dim: int = 2048):
        super().__init__()
        self.backbone     = backbone
        self.head         = head
        self.backbone_dim = backbone_dim
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        B, T, C, H, W = x.shape
        with torch.no_grad():
            f = self.backbone(x.view(B * T, C, H, W)).view(B, T, self.backbone_dim)
        logits, _ = self.head(f, training=False)
        return logits


def load_frozen_backbone(device, teacher_cls, ckpt_path, hidden_size, num_classes,
                         dropout_p, remap_fn):
    """Load the Task-0 teacher and return just its frozen backbone."""
    teacher = teacher_cls(hidden_size=hidden_size, num_classes=num_classes,
                          dropout_p=dropout_p).to(device)
    mapped = remap_fn(torch.load(ckpt_path, map_location=device))
    missing, _ = teacher.load_state_dict(mapped, strict=False)
    real_missing = [k for k in missing if "num_batches_tracked" not in k]
    if real_missing:
        raise RuntimeError(f"Checkpoint missing keys: {real_missing[:5]}")
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher.backbone


# ── Prediction sweeps ─────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, loader, device):
    """Returns (preds, labels) as numpy arrays. Accepts logits or (logits, emb)."""
    model.eval()
    P, Y = [], []
    for x, y in loader:
        out = model(x.to(device))
        logits = out[0] if isinstance(out, (tuple, list)) else out
        P.append(logits.argmax(1).cpu())
        Y.append(y)
    return to_numpy(torch.cat(P)).astype(int), to_numpy(torch.cat(Y)).astype(int)


@torch.no_grad()
def predict_head(head, cache: dict, task_ids: list, device, split: str = "test"):
    """Same, but straight from the cached embeddings."""
    P, Y = [], []
    head.eval()
    for t in task_ids:
        k = f"t{t}_{split}"
        if k not in cache:
            continue
        f, y = cache[k]
        logits, _ = head(f.to(device), training=False)
        P.append(logits.argmax(1).cpu())
        Y.append(y)
    return to_numpy(torch.cat(P)).astype(int), to_numpy(torch.cat(Y)).astype(int)


# ── Reporting ─────────────────────────────────────────────────────────────────
def report_accuracy(preds, labels, task_classes, task_names, class_to_idx,
                    title="Accuracy", seen_tasks=None):
    """
    Per-task accuracy plus the combined number, printed loudly.

    Combined is computed over all clips at once, so it is weighted by class support —
    it is not the mean of the per-task numbers.
    """
    seen_tasks = seen_tasks if seen_tasks is not None else range(len(task_classes))
    preds, labels = np.asarray(preds), np.asarray(labels)

    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)

    per_task = {}
    for t in seen_tasks:
        idx  = [class_to_idx[c] for c in task_classes[t] if c in class_to_idx]
        mask = np.isin(labels, idx)
        if mask.sum() == 0:
            continue
        acc = float((preds[mask] == labels[mask]).mean())
        per_task[t] = acc
        print(f"  {task_names[t]:<10} {acc:>8.2%}   ({int(mask.sum())} clips, "
              f"{len(idx)} classes)")

    combined = float((preds == labels).mean())
    mean_task = float(np.mean(list(per_task.values()))) if per_task else 0.0
    print("  " + "-" * 58)
    print(f"  {'COMBINED':<10} {combined:>8.2%}   ({len(labels)} clips, all classes)")
    print(f"  {'mean/task':<10} {mean_task:>8.2%}")
    print("=" * 62)

    return {"per_task": per_task, "combined": combined, "mean_task": mean_task}


def compare_embeddings_vs_real(emb_res: dict, real_res: dict, task_names: list):
    """Side-by-side check that the two evaluation paths agree."""
    print()
    print("=" * 62)
    print("  EMBEDDINGS vs REAL IMAGES")
    print("=" * 62)
    print(f"  {'Split':<12}{'embeddings':>13}{'real images':>14}{'diff':>9}")
    print("  " + "-" * 58)
    for t, a in emb_res["per_task"].items():
        b = real_res["per_task"].get(t)
        if b is None:
            continue
        print(f"  {task_names[t]:<12}{a:>13.2%}{b:>14.2%}{b - a:>+9.2%}")
    print("  " + "-" * 58)
    print(f"  {'COMBINED':<12}{emb_res['combined']:>13.2%}"
          f"{real_res['combined']:>14.2%}{real_res['combined'] - emb_res['combined']:>+9.2%}")
    print("=" * 62)
    d = abs(real_res["combined"] - emb_res["combined"])
    print("  Paths agree." if d < 0.02 else
          f"  WARNING: {d:.2%} gap. The cache and the raw clips may be out of sync.")


# ── Latent space of a trained head ────────────────────────────────────────────
@torch.no_grad()
def head_embeddings(head, cache: dict, task_ids: list, device, split: str = "test"):
    """
    The head's own 256-d representation (post-LSTM), not the raw cached features.

    This is what the classifier actually sees, so it shows how continual training
    reshaped the space.
    """
    head.eval()
    E, Y = [], []
    for t in task_ids:
        k = f"t{t}_{split}"
        if k not in cache:
            continue
        f, y = cache[k]
        E.append(head.embed(f.to(device)).cpu())
        Y.append(y)
    return torch.cat(E), torch.cat(Y)


@torch.no_grad()
def head_logits(head, cache: dict, task_ids: list, device, split: str = "test", batch_size: int = 256):
    """The head's raw class logits over the given tasks. Returns (logits, labels)."""
    head.eval()
    L, Y = [], []
    for t in task_ids:
        k = f"t{t}_{split}"
        if k not in cache:
            continue
        f, y = cache[k]
        for i in range(0, len(f), batch_size):
            logits, _ = head(f[i:i + batch_size].to(device), training=False)
            L.append(logits.cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


# ── Raw clip loaders (real-image evaluation) ──────────────────────────────────
def clips_available(task_roots: list, split: str = "test"):
    """Colab wipes /content between sessions, so the raw clips may be gone."""
    return all(os.path.isdir(os.path.join(r, split)) and os.listdir(os.path.join(r, split))
               for r in task_roots)


def clip_loaders(task_roots, task_classes, class_to_idx, split="test",
                 batch_size=8, num_workers=2, pin_memory=False):
    """Per-task and combined loaders over the preprocessed .pt clips."""
    from src.data.dataset import UCF101Clips

    dsets = []
    for root, classes in zip(task_roots, task_classes):
        c2i = {c: class_to_idx[c] for c in classes if c in class_to_idx}
        dsets.append(UCF101Clips(os.path.join(root, split), class_to_idx=c2i))

    per = [DataLoader(d, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory) for d in dsets]
    combined = DataLoader(ConcatDataset(dsets), batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
    return per, combined


def project_head_space(head, cache: dict, task_ids: list, device, seed: int = 42,
                       split: str = "test"):
    """
    2-D t-SNE of the head's own embedding space. Returns (xy, labels).

    PCA to 50-d first so t-SNE on 256-d features stays fast.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    emb, lab = head_embeddings(head, cache, task_ids, device, split)
    x = to_numpy(emb)
    if x.shape[1] > 50:
        x = PCA(n_components=50, random_state=seed).fit_transform(x)
    xy = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto",
              perplexity=min(30, max(5, len(x) // 4 - 1))).fit_transform(x)
    return xy, to_numpy(lab).astype(int)
