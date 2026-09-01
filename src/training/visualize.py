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
    Plot per-split accuracy across CL stages. snapshots is a list of
    evaluate_all_tasks() outputs, one per stage.
    """
    splits = []
    for snap in snapshots:
        for k in snap:
            if k not in splits:
                splits.append(k)
    stages = [f"stage {i}" for i in range(len(snapshots))]

    plt.figure(figsize=(9, 5))
    for split in splits:
        xs = [stages[i] for i, snap in enumerate(snapshots) if split in snap]
        ys = [snap[split]["accuracy"] for snap in snapshots if split in snap]
        style = "--" if "combined" in split else "-"
        plt.plot(xs, ys, marker="o", linestyle=style, label=split)

    plt.title("Accuracy per split across CL stages", fontweight="bold")
    plt.xlabel("CL stage"); plt.ylabel("accuracy"); plt.ylim(0, 1.05)
    plt.legend(); plt.tight_layout(); plt.show()
