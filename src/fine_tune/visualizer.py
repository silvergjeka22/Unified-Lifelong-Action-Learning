# Run from notebook: %run /content/src/fine_tune/visualizer.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config.config import SELECTED_CLASSES


def plot_training_results(results):
    """
    Plot train/val loss and accuracy curves.

    Args:
        results : dict returned by fine_tune_model()
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'], label="Train Loss")
    plt.plot(results['val_losses'],   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(results['train_accs'], label="Train Acc")
    plt.plot(results['val_accs'],   label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(all_labels, all_preds,
                          num_classes=None, classes_list=None):
    """
    Plot raw count and normalized confusion matrices.

    Args:
        all_labels   : list of ground-truth labels
        all_preds    : list of predicted labels
        num_classes  : number of classes (default: len(SELECTED_CLASSES))
        classes_list : list of class names (default: SELECTED_CLASSES)
    """
    if classes_list is None:
        classes_list = SELECTED_CLASSES
    if num_classes is None:
        num_classes = len(classes_list)

    cm      = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    tick_labels = classes_list[:num_classes]

    # Counts
    plt.figure(figsize=(20, 18))
    ax1 = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                      annot_kws={'size': 6},
                      xticklabels=tick_labels,
                      yticklabels=tick_labels)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title(f"Confusion Matrix — Counts ({num_classes} Classes)")
    ax1.tick_params(axis='x', rotation=90, labelsize=9)
    ax1.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    plt.show()

    # Normalized
    plt.figure(figsize=(20, 18))
    ax2 = sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', square=True,
                      annot_kws={'size': 6},
                      xticklabels=tick_labels,
                      yticklabels=tick_labels)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title(f"Confusion Matrix — Normalized ({num_classes} Classes)")
    ax2.tick_params(axis='x', rotation=90, labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    plt.show()
