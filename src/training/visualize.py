import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.config.config import SELECTED_CLASSES

TRAIN_COLOR = "#2563EB"
VAL_COLOR   = "#DC2626"


def plot_training_results(history, task_name=""):
    """Plot train/val loss and accuracy from a train_model() history dict."""
    epochs = range(1, len(history["train_losses"]) + 1)
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(13, 5))
    title = f"Training - {task_name}" if task_name else "Training"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax_l.plot(epochs, history["train_losses"], label="train", color=TRAIN_COLOR, marker="o")
    ax_l.plot(epochs, history["val_losses"],   label="val",   color=VAL_COLOR,   marker="o")
    ax_l.set_xlabel("epoch"); ax_l.set_ylabel("loss"); ax_l.set_title("Loss"); ax_l.legend()

    ax_a.plot(epochs, history["train_accs"], label="train", color=TRAIN_COLOR, marker="o")
    ax_a.plot(epochs, history["val_accs"],   label="val",   color=VAL_COLOR,   marker="o")
    ax_a.set_xlabel("epoch"); ax_a.set_ylabel("accuracy"); ax_a.set_ylim(0, 1.05)
    ax_a.set_title("Accuracy"); ax_a.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, num_classes=None, classes_list=None, name=""):
    """Plot the normalised confusion matrix."""
    if classes_list is None:
        classes_list = SELECTED_CLASSES
    if num_classes is None:
        num_classes = len(classes_list)

    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    cm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))
    ticks = list(classes_list[:num_classes])

    plt.figure(figsize=(max(6, num_classes * 0.7), max(5, num_classes * 0.6)))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlOrRd", square=True,
                xticklabels=ticks, yticklabels=ticks, cbar_kws={"shrink": 0.6})
    plt.title(f"Confusion Matrix {name}", fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_cl_forgetting(snapshots):
    """
    Grouped bar chart of per-task accuracy at each CL stage. snapshots is a list of
    evaluate_all_tasks() outputs, one per stage. Within a task, bars shrinking from
    stage to stage is forgetting.
    """
    def nice(k):
        if k == "base":
            return "Base"
        return k.replace("_only", "").replace("task", "Task ")

    def stage_name(s):
        return "base" if s == 0 else f"after T{s}"

    tasks = []
    for snap in snapshots:
        for k in snap:
            if not k.startswith("combined") and k not in tasks:
                tasks.append(k)

    n_stage = len(snapshots)
    x = np.arange(len(tasks))
    width = 0.8 / max(n_stage, 1)

    plt.figure(figsize=(1.8 * len(tasks) + 3, 5))
    for s, snap in enumerate(snapshots):
        vals = [snap[k]["accuracy"] if k in snap else np.nan for k in tasks]
        pos  = x + (s - (n_stage - 1) / 2) * width
        plt.bar(pos, vals, width, label=stage_name(s))
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                plt.text(p, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, [nice(k) for k in tasks])
    plt.ylim(0, 1.15)
    plt.ylabel("accuracy")
    plt.title("Per-task accuracy across CL stages (shrinking bars = forgetting)", fontweight="bold")
    plt.legend(title="measured")
    plt.tight_layout()
    plt.show()


def plot_forgetting_history(history, epochs_per_task=5):
    """
    Line timeline of each task's test accuracy over the whole CL stream. `history` is a
    list of {task_name: accuracy} dicts, one per epoch (index 0 = before any CL task),
    as filled by the trainers' track_history. A line rising then falling is a task being
    learned then forgotten; a line staying high is a task being retained.
    """
    tasks = []
    for entry in history:
        for k in entry:
            if k not in tasks:
                tasks.append(k)

    x = np.arange(len(history))
    n_task = max((len(history) - 1) // max(epochs_per_task, 1), 0)

    plt.figure(figsize=(11, 5.5))
    for name in tasks:
        y = [entry.get(name, np.nan) for entry in history]
        plt.plot(x, y, marker="o", linewidth=2, label=name)

    for k in range(2, n_task + 1):                       # dividers between task phases
        plt.axvline(0.5 + (k - 1) * epochs_per_task, color="gray", linestyle="--", linewidth=0.8)
    for k in range(1, n_task + 1):                       # phase labels
        center = 1 + (k - 1) * epochs_per_task + (epochs_per_task - 1) / 2
        plt.text(center, 1.08, f"training Task {k}", ha="center", fontsize=9, color="#5F5E5A")

    plt.xticks(x)
    plt.ylim(0, 1.15)
    plt.xlabel("training step  (0 = before any continual-learning task)")
    plt.ylabel("test accuracy")
    plt.title("Forgetting history: per-task accuracy over the CL stream", fontweight="bold")
    plt.legend(loc="center left")
    plt.tight_layout()
    plt.show()
