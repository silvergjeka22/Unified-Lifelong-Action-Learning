import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report,
)

from src.config.config import SELECTED_CLASSES


def _group_metrics(labels, preds, classes_list, span, name):
    """Print weighted metrics for one class range [span[0], span[1])."""
    mask = (labels >= span[0]) & (labels < span[1])
    if mask.sum() == 0:
        print(f"  (no samples for {name})")
        return
    labs = np.arange(span[0], span[1])
    print(f"\n{name} [{span[0]}-{span[1]-1}]  ({span[1]-span[0]} classes)")
    print(f"  Accuracy : {accuracy_score(labels[mask], preds[mask]):.4f}")
    print(f"  F1-Score : {f1_score(labels[mask], preds[mask], average='weighted', labels=labs, zero_division=0):.4f}")


def print_detailed_metrics(all_labels, all_preds, num_classes=None, classes_list=None,
                           split_old=None, split_new=None):
    """
    Print overall accuracy / F1, the per-class report, and (for CL) the old vs new
    class split. Reporting is this function's job, so it prints.
    """
    if classes_list is None:
        classes_list = SELECTED_CLASSES
    if num_classes is None:
        num_classes = len(classes_list)

    labels = np.array(all_labels)
    preds  = np.array(all_preds)

    print(f"\nMETRICS FOR {num_classes} CLASSES")
    print(f"  Accuracy : {accuracy_score(labels, preds):.4f}")
    print(f"  F1-Score : {f1_score(labels, preds, average='weighted', zero_division=0):.4f}")

    print("\nPER-CLASS REPORT")
    print(classification_report(labels, preds, labels=np.arange(num_classes),
                                target_names=classes_list[:num_classes], zero_division=0))

    if split_old is not None:
        _group_metrics(labels, preds, classes_list, split_old, "OLD CLASSES")
    if split_new is not None:
        _group_metrics(labels, preds, classes_list, split_new, "NEW CLASSES")
