# Run from notebook: %run /content/src/fine_tune/visualizer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config.config import SELECTED_CLASSES

# STYLE DEFAULTS
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         12,
})

_COLORS = {
    "train": "#2563EB",   # blue
    "val":   "#DC2626",   # red
    "test":  "#16A34A",   # green
}


# TRAINING CURVES
def plot_training_results(results, task_index=None, save_path=None):
    """
    Plot train/val loss AND accuracy from fine_tune_model() output.
    A single figure with two subplots side by side.

    Args:
        results     : dict returned by fine_tune_model()
        task_index  : optional int — appended to the figure title, e.g. 1 -> "Task 1"
        save_path   : optional path to save the figure (PNG/PDF)
    """
    tag    = f" — Task {task_index}" if task_index is not None else ""
    epochs = range(1, len(results['train_losses']) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training Curves{tag}", fontsize=15, fontweight='bold', y=1.01)

    # Loss
    ax_loss.plot(epochs, results['train_losses'], label="Train",
                 color=_COLORS["train"], linewidth=2, marker='o', markersize=4)
    ax_loss.plot(epochs, results['val_losses'],   label="Val",
                 color=_COLORS["val"],   linewidth=2, marker='o', markersize=4)
    ax_loss.set_xlabel("Epoch");  ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss");    ax_loss.legend()

    # Accuracy
    ax_acc.plot(epochs, results['train_accs'], label="Train",
                color=_COLORS["train"], linewidth=2, marker='o', markersize=4)
    ax_acc.plot(epochs, results['val_accs'],   label="Val",
                color=_COLORS["val"],   linewidth=2, marker='o', markersize=4)
    ax_acc.set_xlabel("Epoch");  ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05);   ax_acc.set_title("Accuracy")
    ax_acc.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_train_val_curves(train_acc, val_acc, train_loss, val_loss,
                          task_index=1, save_path=None):
    """
    Identical to plot_training_results() but accepts flat lists directly
    instead of the history dict — kept for backward compatibility.

    Args:
        train_acc, val_acc, train_loss, val_loss : lists of per-epoch values
        task_index  : int for figure title
        save_path   : optional save path
    """
    results = {
        'train_accs':   train_acc,
        'val_accs':     val_acc,
        'train_losses': train_loss,
        'val_losses':   val_loss,
    }
    plot_training_results(results, task_index=task_index, save_path=save_path)


# CONFUSION MATRICES
def _draw_heatmap(data, fmt, cmap, title, tick_labels, ax=None):
    """Internal helper — draw one heatmap on ax (or current axes)."""
    n = len(tick_labels)
    fs_annot = max(5,  min(10, int(120 / n)))
    fs_tick  = max(6,  min(12, int(180 / n)))
    fs_label = max(10, min(18, int(220 / n)))

    if ax is None:
        cell = max(0.6, min(1.5, 30 / n))
        fig_w = n * cell + 4
        fig_h = n * cell + 3
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        linewidths=0.3,
        linecolor='#e5e7eb',
        annot_kws={'size': fs_annot, 'weight': 'bold'},
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={'shrink': 0.6, 'pad': 0.01},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=fs_label, fontweight='bold', labelpad=10)
    ax.set_ylabel("True",      fontsize=fs_label, fontweight='bold', labelpad=10)
    ax.set_title(title,        fontsize=fs_label, fontweight='bold', pad=14)
    ax.tick_params(axis='x', rotation=45,  labelsize=fs_tick)
    ax.tick_params(axis='y', rotation=0,   labelsize=fs_tick)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')


def plot_confusion_matrix(all_labels, all_preds,
                          num_classes=None, classes_list=None,
                          name="", save_path=None):
    """
    Show raw-count AND normalised confusion matrices (two separate figures).
    Figure size scales automatically with num_classes.

    Args:
        all_labels   : list/array of ground-truth labels
        all_preds    : list/array of predicted labels
        num_classes  : int (default: len(SELECTED_CLASSES))
        classes_list : list of class name strings (default: SELECTED_CLASSES)
        name         : string appended to figure titles
        save_path    : if set, saves both figures (<save_path>_counts.png
                       and <save_path>_norm.png)
    """
    if classes_list is None:
        classes_list = SELECTED_CLASSES
    if num_classes is None:
        num_classes = len(classes_list)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    cm      = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    tick_labels = list(classes_list[:num_classes])
    tag = f" ({name})" if name else f" ({num_classes} classes)"

    _draw_heatmap(cm,      'd',   'Blues',  f"Confusion Matrix — Counts{tag}",     tick_labels)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_counts.png", bbox_inches='tight', dpi=150)
    plt.show()

    _draw_heatmap(cm_norm, '.2f', 'YlOrRd', f"Confusion Matrix — Normalised{tag}", tick_labels)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_norm.png", bbox_inches='tight', dpi=150)
    plt.show()


def plot_confusion_matrix_from_preds(all_labels, all_preds,
                                     name="task", classes=None,
                                     save_path=None):
    """
    Identical behaviour to plot_confusion_matrix() — accepts raw arrays and
    an optional class list; infers num_classes from data if classes is None.
    Kept as a convenience alias for notebooks that call this signature.

    Args:
        all_labels : list/array of ground-truth labels
        all_preds  : list/array of predicted labels
        name       : string tag for figure titles
        classes    : list of class name strings; inferred if None
        save_path  : optional save path prefix
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

# COLLECT PREDICTIONS  (helper — mirrors collect_predictions from old file)
def collect_predictions(loader, model, device, task_offset=0):
    """
    Run inference on a DataLoader and return (all_labels, all_preds) as
    numpy arrays with global labels (offset applied).

    Args:
        loader       : DataLoader
        model        : PyTorch model
        device       : 'cuda' or 'cpu'
        task_offset  : global label offset for CL (default 0)

    Returns:
        (all_labels, all_preds) — both np.ndarray
    """
    model.to(device)
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for x, y in loader:
            x      = x.to(device)
            y      = (y + task_offset).to(device).long()
            logits = model(x)
            preds  = torch.argmax(logits, dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# CL FORGETTING PLOT
def plot_cl_forgetting(snapshots, save_path=None):
    """
    Plot per-task accuracy across CL stages to visualise catastrophic
    forgetting.

    Args:
        snapshots : list of dicts, one per CL stage, each dict is the output
                    of evaluate_all_tasks():
                        [
                          {"base": {"accuracy": 0.82, ...}},            # after base
                          {"base": {...}, "task1_only": {...}},          # after task1
                          {"base": {...}, "task1_only": {...}, ...},     # after task2
                          ...
                        ]
        save_path : optional path to save the figure

    Example:
        snapshots = []
        snapshots.append(evaluate_all_tasks(model, device, base_loader))
        # ... train task 1 ...
        snapshots.append(evaluate_all_tasks(model, device, base_loader,
                                            task_loaders=[t1_loader]))
        plot_cl_forgetting(snapshots)
    """
    if not snapshots:
        print("[plot_cl_forgetting] No snapshots provided.")
        return

    # Collect all split keys across all snapshots
    all_splits = []
    for snap in snapshots:
        for k in snap:
            if k not in all_splits:
                all_splits.append(k)

    n_stages = len(snapshots)
    x_labels = [f"Stage {i}" for i in range(n_stages)]

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CL Task Accuracy & Loss Across Stages",
                 fontsize=14, fontweight='bold', y=1.01)

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    for idx, split in enumerate(all_splits):
        ys_acc  = [snap[split]["accuracy"] if split in snap else None for snap in snapshots]
        ys_loss = [snap[split]["loss"]     if split in snap else None for snap in snapshots]
        x_valid_a = [x_labels[i] for i, v in enumerate(ys_acc)  if v is not None]
        x_valid_l = [x_labels[i] for i, v in enumerate(ys_loss) if v is not None]
        y_valid_a = [v for v in ys_acc  if v is not None]
        y_valid_l = [v for v in ys_loss if v is not None]

        style = '--' if 'combined' in split else '-'
        mk    = markers[idx % len(markers)]

        ax_acc.plot(x_valid_a,  y_valid_a,  label=split, linestyle=style,
                    marker=mk, linewidth=2)
        ax_loss.plot(x_valid_l, y_valid_l,  label=split, linestyle=style,
                     marker=mk, linewidth=2)

    ax_acc.set_ylabel("Accuracy");  ax_acc.set_ylim(0, 1.05)
    ax_acc.set_xlabel("CL Stage");  ax_acc.set_title("Accuracy per Task")
    ax_acc.legend(fontsize=9, loc='lower left')

    ax_loss.set_ylabel("Loss");     ax_loss.set_xlabel("CL Stage")
    ax_loss.set_title("Loss per Task")
    ax_loss.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
