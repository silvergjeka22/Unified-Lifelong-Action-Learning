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
    Figure size scales automatically with num_classes (1.5 inch per cell).

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

    # figsize scales with num_classes so cells are always big enough
    cell_inch = 1.5
    fig_w = num_classes * cell_inch + 8   # +8 for y-labels + colorbar
    fig_h = num_classes * cell_inch + 6   # +6 for x-labels + title

    def _draw(data, fmt, cmap, title):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            square=True,
            linewidths=0.5,
            linecolor='#dddddd',
            annot_kws={'size': 25, 'weight': 'bold', 'color': 'black'},
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            cbar_kws={'shrink': 0.5, 'pad': 0.01},
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=30, fontweight='bold', labelpad=16)
        ax.set_ylabel("True",      fontsize=30, fontweight='bold', labelpad=16)
        ax.set_title(title,        fontsize=30, fontweight='bold', pad=20)
        ax.tick_params(axis='x', rotation=45, labelsize=25)
        ax.tick_params(axis='y', rotation=0,  labelsize=25)
        plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
        plt.tight_layout()
        plt.show()

    _draw(cm,      'd',   'Blues',  f"Confusion Matrix — Counts ({num_classes} Classes)")
    _draw(cm_norm, '.2f', 'YlOrRd', f"Confusion Matrix — Normalized ({num_classes} Classes)")
