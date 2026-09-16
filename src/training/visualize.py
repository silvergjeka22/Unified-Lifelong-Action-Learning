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


def plot_old_new_confusion(all_labels, all_preds, n_base, name=""):
    """Row-normalised 2x2 old-vs-new confusion: how often a true OLD (label < n_base) or NEW clip is
    predicted as old vs new. A readable, realistic stand-in for a big per-class matrix when there are
    many base classes - it shows the drift toward the new classes at a glance."""
    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    true_new = (labels >= n_base).astype(int)
    pred_new = (preds  >= n_base).astype(int)
    cm = confusion_matrix(true_new, pred_new, labels=[0, 1]).astype(float)
    cm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlOrRd", square=True,
                xticklabels=["old", "new"], yticklabels=["old", "new"], cbar=False)
    plt.title(f"Old vs new  {name}", fontweight="bold")
    plt.xlabel("predicted"); plt.ylabel("true")
    plt.tight_layout()
    plt.show()


def plot_old_vs_new(class_names, base_old, base_new, maml_old, maml_new, k_shot=5):
    """
    Compare no-meta vs MAML on OLD (support-video) and NEW (held-out-video) clips. Each *_old /
    *_new is a per-class list of accuracies. Left panel: overall seen-vs-unseen. Right panel:
    per-class accuracy on the NEW videos - the real generalisation test where MAML should win.
    """
    n = len(class_names)
    x2, xc, w = np.arange(2), np.arange(n), 0.38
    fig, (ax_o, ax_c) = plt.subplots(1, 2, figsize=(13, 5))

    ax_o.bar(x2 - w / 2, [np.nanmean(base_old), np.nanmean(base_new)], w, label="no meta")
    ax_o.bar(x2 + w / 2, [np.nanmean(maml_old), np.nanmean(maml_new)], w, label="MAML")
    ax_o.set_xticks(x2); ax_o.set_xticklabels(["old\n(support video)", "new\n(held-out videos)"])
    ax_o.set_ylim(0, 1.05); ax_o.set_ylabel("accuracy")
    ax_o.set_title("Overall: seen vs unseen videos"); ax_o.legend()

    ax_c.bar(xc - w / 2, base_new, w, label="no meta")
    ax_c.bar(xc + w / 2, maml_new, w, label="MAML")
    ax_c.set_xticks(xc); ax_c.set_xticklabels(class_names, rotation=20, ha="right")
    ax_c.set_ylim(0, 1.05); ax_c.set_ylabel("accuracy")
    ax_c.set_title(f"Per-class on NEW videos ({k_shot}-shot)"); ax_c.legend()

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
