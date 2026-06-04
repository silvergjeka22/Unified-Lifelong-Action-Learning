"""
Continual Learning evaluation metrics.

Standard metrics used to compare CL methods:
    average_accuracy     — mean accuracy across all tasks seen so far
    forgetting_measure   — drop in accuracy on old tasks after learning new ones
    forward_transfer     — how past learning helps future tasks
    backward_transfer    — how new tasks affect old-task performance (negative = forgetting)
    build_accuracy_matrix — full T×T accuracy table
    cl_report            — print a full CL results report
"""

import numpy as np
import torch


def forgetting(accs: list) -> float:
    """
    Average forgetting across tasks.
    accs[i] = accuracy on task i after final training step.
    Computes max(accs) - accs[-1] averaged over all tasks.
    """
    if len(accs) < 2:
        return 0.0
    return float(np.mean([max(a, accs[-1]) - accs[-1] for a in accs[:-1]]))


def avg_intra_dist(emb: torch.Tensor, labels: torch.Tensor) -> float:
    """Average intra-class pairwise L2 distance across all classes."""
    dists = []
    for cls in labels.unique():
        mask = labels == cls
        feats = emb[mask].float()
        if feats.shape[0] < 2:
            continue
        diff = feats.unsqueeze(0) - feats.unsqueeze(1)          # (N,N,D)
        d    = diff.norm(dim=-1)                                  # (N,N)
        idx  = torch.triu_indices(d.shape[0], d.shape[1], offset=1)
        dists.append(d[idx[0], idx[1]].mean().item())
    return float(np.mean(dists)) if dists else 0.0


def build_accuracy_matrix(results_per_task: list) -> np.ndarray:
    """
    Build the T×T accuracy matrix R where R[i][j] = accuracy on task j
    evaluated after training on task i.

    Args:
        results_per_task : list of dicts, one per task.
                           Each dict maps task_idx -> accuracy (0–1).
                           Example: [{0: 0.82}, {0: 0.71, 1: 0.88}, ...]

    Returns:
        R : (T, T) numpy array, nan for not-yet-evaluated entries
    """
    T = len(results_per_task)
    R = np.full((T, T), np.nan)
    for i, task_results in enumerate(results_per_task):
        for j, acc in task_results.items():
            R[i][j] = acc
    return R


def average_accuracy(R: np.ndarray) -> float:
    """
    AA = mean of last-row accuracies (accuracy on all tasks after full training).
    """
    T   = R.shape[0]
    acc = [R[T - 1][j] for j in range(T) if not np.isnan(R[T - 1][j])]
    return float(np.mean(acc)) if acc else 0.0


def forgetting_measure(R: np.ndarray) -> float:
    """
    Average forgetting:
        F = (1/T-1) * Σ_{j=0}^{T-2} [ max_{i≤T-1} R[i][j] - R[T-1][j] ]

    Positive forgetting = catastrophic forgetting.
    """
    T  = R.shape[0]
    fg = []
    for j in range(T - 1):
        col     = R[:T, j]
        valid   = col[~np.isnan(col)]
        if len(valid) >= 2:
            fg.append(float(np.max(valid) - valid[-1]))
    return float(np.mean(fg)) if fg else 0.0


def backward_transfer(R: np.ndarray) -> float:
    """
    BWT = (1/T-1) * Σ_{j=0}^{T-2} [ R[T-1][j] - R[j][j] ]

    Negative BWT = forgetting; positive BWT = backward knowledge transfer.
    """
    T   = R.shape[0]
    bwt = []
    for j in range(T - 1):
        if not (np.isnan(R[T - 1][j]) or np.isnan(R[j][j])):
            bwt.append(R[T - 1][j] - R[j][j])
    return float(np.mean(bwt)) if bwt else 0.0


def forward_transfer(R: np.ndarray, random_baseline: float = 0.1) -> float:
    """
    FWT = (1/T-1) * Σ_{j=1}^{T-1} [ R[j-1][j] - b_j ]

    where b_j is the random-chance baseline for task j.
    Positive FWT = past learning helps new tasks.
    """
    T   = R.shape[0]
    fwt = []
    for j in range(1, T):
        if not np.isnan(R[j - 1][j]):
            fwt.append(R[j - 1][j] - random_baseline)
    return float(np.mean(fwt)) if fwt else 0.0


def cl_report(R: np.ndarray, method_name: str = "", class_names_per_task: list = None):
    """
    Print a full continual learning report for a given accuracy matrix.

    Args:
        R                    : (T, T) accuracy matrix from build_accuracy_matrix()
        method_name          : label for the report header
        class_names_per_task : optional list of task labels for columns
    """
    T    = R.shape[0]
    aa   = average_accuracy(R)
    fg   = forgetting_measure(R)
    bwt  = backward_transfer(R)
    fwt  = forward_transfer(R)

    task_labels = class_names_per_task or [f"T{j}" for j in range(T)]

    W = 70
    print("\n" + "=" * W)
    print(f"  CL REPORT — {method_name}")
    print("=" * W)

    # Accuracy matrix
    header = "  After \\ On  | " + " | ".join(f"{l:>6}" for l in task_labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(T):
        row = f"  Task {i:<6}  | "
        row += " | ".join(
            f"{R[i][j]:>6.2%}" if not np.isnan(R[i][j]) else f"{'—':>6}"
            for j in range(T)
        )
        print(row)

    print("=" * W)
    print(f"  Average Accuracy (AA)  : {aa:.2%}")
    print(f"  Forgetting Measure (F) : {fg:.2%}")
    print(f"  Backward Transfer (BWT): {bwt:+.2%}")
    print(f"  Forward Transfer  (FWT): {fwt:+.2%}")
    print("=" * W)

    return {"AA": aa, "F": fg, "BWT": bwt, "FWT": fwt}
