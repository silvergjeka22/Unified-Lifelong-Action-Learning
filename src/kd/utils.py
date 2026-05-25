import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run `model` over every batch in `loader` and collect the LSTM hidden states
    (second return value of each model's forward pass).

    Args:
        model   : Teacher or Student (must return (logits, hidden))
        loader  : DataLoader yielding (clip_tensor, label) batches
        device  : Inference device

    Returns:
        embeddings  [N, hidden_size]  — concatenated hidden states
        labels      [N]               — corresponding class indices
    """
    model.eval()
    model.to(device)
    embs, labs = [], []

    for x, y in loader:
        _, h = model(x.to(device))
        embs.append(h.cpu())
        labs.append(y.cpu())

    return torch.cat(embs, 0), torch.cat(labs, 0)


def extract_all(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> tuple[dict, dict]:
    """
    Extract embeddings from both teacher and student for every data split.

    Args:
        teacher : Teacher model
        student : Student model
        loaders : {"train": ..., "val": ..., "test": ...}
        device  : Inference device

    Returns:
        t_raw   : {"train": (embs, labels), "val": ..., "test": ...}  — teacher
        s_raw   : {"train": (embs, labels), "val": ..., "test": ...}  — student
    """
    t_raw: dict = {}
    s_raw: dict = {}

    for split, loader in loaders.items():
        te, tl = extract_embeddings(teacher, loader, device)
        se, sl = extract_embeddings(student, loader, device)

        t_raw[split] = (te, tl)
        s_raw[split] = (se, sl)

    return t_raw, s_raw


def restore_head_to_student(
    student: nn.Module,
    head: nn.Module,
) -> nn.Module:
    """
    Copy the trained EmbeddingHead classifier weights into `student.fc`
    so the student can be evaluated as a standalone model.

    Args:
        student : MobileNetV3SmallLSTMStudent (or any model with a `.fc` Linear)
        head    : Trained EmbeddingHead whose `.classifier` matches student.fc

    Returns:
        student with updated fc weights (same object, modified in-place)
    """
    with torch.no_grad():
        student.fc.weight.copy_(head.classifier.weight)
        student.fc.bias.copy_(head.classifier.bias)
    print("  Restored Head Linear into student.fc safely ✓")
    return student


@torch.no_grad()
def calculate_accuracies(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> tuple[dict[int, float], float]:
    """
    Compute overall and per-class frame-level classification accuracy.

    Args:
        model       : Student model (returns (logits, _) for each batch)
        loader      : Test DataLoader
        num_classes : Total number of action classes
        device      : Inference device

    Returns:
        class_accs  : {class_idx: accuracy_percent, ...}
        overall_acc : float — overall accuracy in percent
    """
    model.eval()
    model.to(device)

    class_correct = np.zeros(num_classes)
    class_total   = np.zeros(num_classes)
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y  = x.to(device), y.to(device)
        logits, _ = model(x)
        preds = logits.argmax(dim=1)

        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

        for i in range(num_classes):
            mask = (y == i)
            class_correct[i] += (preds[mask] == i).sum().item()
            class_total[i]   += mask.sum().item()

    overall_acc = (total_correct / max(total_samples, 1)) * 100
    class_accs  = {
        i: (class_correct[i] / max(class_total[i], 1)) * 100
        for i in range(num_classes)
    }

    return class_accs, overall_acc