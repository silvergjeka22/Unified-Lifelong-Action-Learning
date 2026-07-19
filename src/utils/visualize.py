import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.config.config import SELECTED_CLASSES


# GLOBAL STYLE
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         18,
})

TRAIN_COLOR = "#2563EB"   # blue
VAL_COLOR   = "#DC2626"   # red


# TRAINING CURVES
def plot_training_results(results, task_name="", save_path=None):
    """
    Plot train/val Loss and Accuracy from fine_tune_model() output.

    Args:
        results   : dict returned by fine_tune_model()
                    keys: train_losses, val_losses, train_accs, val_accs
        task_name : label shown in the figure title, e.g. "Base", "Task 1"
        save_path : optional file path to save the figure (PNG / PDF)

    Usage:
        plot_training_curves(results, task_name="Base")
        plot_training_curves(results, task_name="Task 1", save_path="curves_t1.png")
    """
    epochs = range(1, len(results["train_losses"]) + 1)
    title  = f"Training Curves — {task_name}" if task_name else "Training Curves"

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    # Loss subplot
    ax_l.plot(epochs, results["train_losses"], label="Train", color=TRAIN_COLOR,
              linewidth=2, marker="o", markersize=4)
    ax_l.plot(epochs, results["val_losses"],   label="Val",   color=VAL_COLOR,
              linewidth=2, marker="o", markersize=4)
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Loss")
    ax_l.set_title("Loss")
    ax_l.legend()

    # Accuracy subplot
    ax_a.plot(epochs, results["train_accs"], label="Train", color=TRAIN_COLOR,
              linewidth=2, marker="o", markersize=4)
    ax_a.plot(epochs, results["val_accs"],   label="Val",   color=VAL_COLOR,
              linewidth=2, marker="o", markersize=4)
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Accuracy")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_title("Accuracy")
    ax_a.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# Backward-compatible alias (accepts flat lists instead of the history dict)

# CONFUSION MATRIX
def plot_confusion_matrix(all_labels, all_preds,
                          num_classes=None, classes_list=None,
                          name="", save_path=None):
    """
    Plot raw-count AND normalised confusion matrices (two figures).
    Font sizes and figure dimensions scale automatically with num_classes
    so numbers inside cells are always large and readable.

    Args:
        all_labels   : list / array of ground-truth labels
        all_preds    : list / array of predicted labels
        num_classes  : int  (default: len(SELECTED_CLASSES))
        classes_list : list of class name strings (default: SELECTED_CLASSES)
        name         : short string appended to figure titles
        save_path    : path prefix — saves  <save_path>_counts.png
                                     and    <save_path>_norm.png

    Usage:
        plot_confusion_matrix(all_labels, all_preds,
                              num_classes=10, classes_list=SELECTED_CLASSES,
                              name="Base", save_path="results/cm_base")
    """
    if classes_list is None:
        classes_list = SELECTED_CLASSES
    if num_classes is None:
        num_classes = len(classes_list)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    cm      = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    cm_norm = np.nan_to_num(cm.astype("float") / cm.sum(axis=1, keepdims=True))

    tick_labels = list(classes_list[:num_classes])
    tag = f" ({name})" if name else f" ({num_classes} classes)"

    _draw_cm(cm,      "d",    "Blues",  f"Confusion Matrix — Counts{tag}",     tick_labels, save_path, "_counts")
    _draw_cm(cm_norm, ".2f", "YlOrRd", f"Confusion Matrix — Normalised{tag}", tick_labels, save_path, "_norm")


def _draw_cm(data, fmt, cmap, title, tick_labels, save_path=None, suffix=""):
    """
    Internal: draw a single heatmap.
    Font sizes and cell dimensions are computed from n = num_classes.
    """
    n = len(tick_labels)

    # Numbers inside cells — hard floor at 14, goes up to 32 for small matrices
    fs_annot = max(14, min(32, int(320 / n)))
    fs_tick  = max(8,  min(15, int(200 / n)))
    fs_label = max(12, min(20, int(240 / n)))

    # Cell size — very generous so large numbers fit without clipping
    cell  = max(1.4, min(3.0, 52 / n))
    fig_w = n * cell + 6
    fig_h = n * cell + 5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        linecolor="#e5e7eb",
        annot_kws={
            "size":             fs_annot,
            "weight":           "bold",
            "color":            "black",
            "va":               "center",
            "ha":               "center",
            "fontfamily":       "monospace",
        },
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={"shrink": 0.55, "pad": 0.02},
        ax=ax,
    )

    ax.set_title(title,        fontsize=fs_label, fontweight="bold", pad=18)
    ax.set_xlabel("Predicted", fontsize=fs_label, fontweight="bold", labelpad=14)
    ax.set_ylabel("True",      fontsize=fs_label, fontweight="bold", labelpad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=fs_tick)
    ax.tick_params(axis="y", rotation=0,  labelsize=fs_tick)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}{suffix}.png", bbox_inches="tight", dpi=150)
    plt.show()


# Convenience alias — infers num_classes from data if classes not provided

# COLLECT PREDICTIONS

# CONTINUAL LEARNING — FORGETTING PLOT
def plot_cl_forgetting(snapshots, save_path=None):
    """
    Plot per-task accuracy AND loss across all CL training stages.
    Solid lines = individual tasks, dashed lines = combined sets.

    Args:
        snapshots : list of dicts, one entry per CL stage.
                    Each dict is the output of evaluate_all_tasks(), e.g.:
                      [
                        {"base": {"accuracy": 0.82, "loss": 0.5}},
                        {"base": {...}, "task1_only": {...}},
                        {"base": {...}, "task1_only": {...}, "task2_only": {...}},
                        ...
                      ]
        save_path : optional file path to save the figure

    Usage:
        snapshots = []
        snapshots.append(evaluate_all_tasks(model, device, base_loader))
        # ... train task 1 ...
        snapshots.append(evaluate_all_tasks(model, device, base_loader,
                         task_loaders=[t1_loader]))
        plot_cl_forgetting(snapshots, save_path="results/forgetting.png")
    """
    if not snapshots:
        print("[plot_cl_forgetting] No snapshots provided.")
        return

    # Preserve insertion order of split keys
    all_splits = []
    for snap in snapshots:
        for k in snap:
            if k not in all_splits:
                all_splits.append(k)

    x_labels = [f"Stage {i}" for i in range(len(snapshots))]
    markers   = ["o", "s", "^", "D", "v", "P", "*", "X"]

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Catastrophic Forgetting — Accuracy & Loss per Task",
                 fontsize=14, fontweight="bold", y=1.01)

    for idx, split in enumerate(all_splits):
        acc_vals  = [snap[split]["accuracy"] if split in snap else None for snap in snapshots]
        loss_vals = [snap[split]["loss"]     if split in snap else None for snap in snapshots]

        xs_a = [x_labels[i] for i, v in enumerate(acc_vals)  if v is not None]
        xs_l = [x_labels[i] for i, v in enumerate(loss_vals) if v is not None]
        ys_a = [v for v in acc_vals  if v is not None]
        ys_l = [v for v in loss_vals if v is not None]

        ls = "--" if "combined" in split else "-"
        mk = markers[idx % len(markers)]

        ax_acc.plot(xs_a, ys_a,  label=split, linestyle=ls, marker=mk, linewidth=2)
        ax_loss.plot(xs_l, ys_l, label=split, linestyle=ls, marker=mk, linewidth=2)

    ax_acc.set_title("Accuracy per Task")
    ax_acc.set_xlabel("CL Stage")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=9, loc="lower left")

    ax_loss.set_title("Loss per Task")
    ax_loss.set_xlabel("CL Stage")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Embedding-space diagnostics (Section 2) — see src/utils/probe.py
# ══════════════════════════════════════════════════════════════════════════════

def plot_probe_bars(report: dict, ax=None, save=None):
    """
    Probe accuracy per task group vs chance.

    The gap between the bar and its chance line is the real signal: a 3-way task at
    90% is far less impressive than a 16-way joint probe at 70%.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8, 4))

    names  = list(report)
    linear = [report[n]["linear"] for n in names]
    knn    = [report[n]["knn"] for n in names]
    chance = [report[n]["chance"] for n in names]

    x, w = np.arange(len(names)), 0.36
    ax.bar(x - w / 2, linear, w, label="linear probe", color="#2563EB")
    ax.bar(x + w / 2, knn,    w, label="1-NN",         color="#93C5FD")
    for i, c in enumerate(chance):
        ax.plot([i - 0.45, i + 0.45], [c, c], color="#DC2626", lw=2,
                label="chance" if i == 0 else None)
    for i, v in enumerate(linear):
        ax.text(i - w / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Accuracy on held-out test split")
    ax.set_title("Can the FROZEN features separate each group?", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    if own:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches="tight")
        plt.show()
    return ax


def plot_latent_space(xy, labels, class_names, title="Latent space", ax=None,
                      save=None, task_of=None, legend=True):
    """
    2-D scatter of the embedding space, one colour per class.

    task_of: optional {label: task_index}. When given, classes are coloured by TASK
    rather than individually — which is how you see whether the tasks occupy separate
    regions or sit on top of each other.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 6))

    if task_of is not None:
        tcol = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]
        for t in sorted(set(task_of.values())):
            mask = np.array([task_of.get(int(l), -1) == t for l in labels])
            if mask.sum():
                ax.scatter(xy[mask, 0], xy[mask, 1], s=14, alpha=0.6,
                           color=tcol[t % len(tcol)], label=f"Task {t}")
    else:
        uniq = sorted(set(int(l) for l in labels))
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 2)))
        for i, c in enumerate(uniq):
            mask = labels == c
            ax.scatter(xy[mask, 0], xy[mask, 1], s=14, alpha=0.7, color=cmap[i],
                       label=class_names[c] if c < len(class_names) else str(c))

    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(fontsize=6, loc="best", markerscale=1.6, ncol=2)

    if own:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches="tight")
        plt.show()
    return ax


def plot_centroid_heatmap(labels, dist, class_names, ax=None, save=None):
    """
    Pairwise distance between class centroids.

    Dark off-diagonal cells = classes that live in the same region. Those are the pairs
    that will fight during continual learning — this plot predicts your confusions
    before you train anything.
    """
    import matplotlib.pyplot as plt

    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(9, 7.5))

    names = [class_names[l] if l < len(class_names) else str(l) for l in labels]
    im = ax.imshow(dist, cmap="viridis")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Distance between class centroids\n(dark = similar = will compete)",
                 fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, label="L2 distance")

    if own:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches="tight")
        plt.show()
    return ax
