# Run from notebook: %run /content/src/fine_tune/visualizer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config.config import SELECTED_CLASSES


# GLOBAL STYLE
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         16,
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
def plot_train_val_results(train_acc, val_acc, train_loss, val_loss,
                          task_index=1, save_path=None):
    plot_training_results(
        {"train_accs": train_acc, "val_accs": val_acc,
         "train_losses": train_loss, "val_losses": val_loss},
        task_name=f"Task {task_index}",
        save_path=save_path,
    )


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

    # Font sizes — large for small matrices, degrade gracefully for big ones
    fs_annot = max(9,  min(24, int(230 / n)))   # numbers inside cells
    fs_tick  = max(7,  min(14, int(200 / n)))   # axis tick labels
    fs_label = max(11, min(20, int(240 / n)))   # axis + title labels

    # Figure size — generous cell size so numbers have room to breathe
    cell  = max(1.0, min(2.4, 42 / n))
    fig_w = n * cell + 5
    fig_h = n * cell + 4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        linewidths=0.4,
        linecolor="#e5e7eb",
        annot_kws={
            "size":   fs_annot,
            "weight": "bold",
            "color":  "black",
            "va":     "center",
            "ha":     "center",
        },
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={"shrink": 0.6, "pad": 0.02},
        ax=ax,
    )

    ax.set_title(title,        fontsize=fs_label, fontweight="bold", pad=16)
    ax.set_xlabel("Predicted", fontsize=fs_label, fontweight="bold", labelpad=12)
    ax.set_ylabel("True",      fontsize=fs_label, fontweight="bold", labelpad=12)
    ax.tick_params(axis="x", rotation=45, labelsize=fs_tick)
    ax.tick_params(axis="y", rotation=0,  labelsize=fs_tick)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}{suffix}.png", bbox_inches="tight", dpi=150)
    plt.show()


# Convenience alias — infers num_classes from data if classes not provided
def plot_confusion_matrix_from_preds(all_labels, all_preds,
                                     name="task", classes=None,
                                     save_path=None):
    """
    Same as plot_confusion_matrix() but infers class names automatically
    when classes is None. Useful for quick inspection in notebooks.

    Usage:
        plot_confusion_matrix_from_preds(labels, preds, name="Task 1")
    """
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    if classes is None:
        n = int(max(all_labels.max(), all_preds.max()) + 1)
        classes = [str(i) for i in range(n)]

    plot_confusion_matrix(
        all_labels, all_preds,
        num_classes=len(classes),
        classes_list=classes,
        name=name,
        save_path=save_path,
    )


# COLLECT PREDICTIONS
def collect_predictions(loader, model, device, task_offset=0):
    """
    Run inference on a DataLoader and return labels + predictions as numpy
    arrays. Global labels are produced by applying task_offset.

    Args:
        loader       : DataLoader
        model        : PyTorch model
        device       : "cuda" or "cpu"
        task_offset  : global label offset for CL (default 0 = no shift)

    Returns:
        (all_labels, all_preds) — both np.ndarray

    Usage:
        labels, preds = collect_predictions(test_loader, model, device)
        labels, preds = collect_predictions(t1_loader, model, device, task_offset=10)
    """
    model.to(device)
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for x, y in loader:
            x      = x.to(device)
            y      = (y + task_offset).to(device).long()
            preds  = torch.argmax(model(x), dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


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
