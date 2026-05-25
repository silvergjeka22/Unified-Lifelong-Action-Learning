import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    model.to(device)

    embeddings = []
    labels = []

    for x, y in loader:
        x = x.to(device)
        _, emb = model(x)
        embeddings.append(emb.cpu())
        labels.append(y.cpu())

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def extract_all_embeddings(teacher, student, loaders, device):
    teacher_data = {}
    student_data = {}

    for split_name, loader in loaders.items():
        t_emb, t_lab = extract_embeddings(teacher, loader, device)
        s_emb, s_lab = extract_embeddings(student, loader, device)

        teacher_data[split_name] = (t_emb, t_lab)
        student_data[split_name] = (s_emb, s_lab)

    return teacher_data, student_data


@torch.no_grad()
def evaluate_head(head, student_embeddings, labels, device, batch_size=128):
    head.eval()

    loader = DataLoader(
        TensorDataset(student_embeddings, labels),
        batch_size=batch_size,
        shuffle=False,
    )

    correct = 0
    total = 0

    for s, y in loader:
        s = s.to(device)
        y = y.to(device)

        logits, _ = head(s, training=False)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / max(total, 1)


@torch.no_grad()
def calculate_accuracies(model, loader, num_classes, device):
    model.eval()
    model.to(device)

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        preds = logits.argmax(dim=1)

        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

        for class_idx in range(num_classes):
            mask = (y == class_idx)
            class_correct[class_idx] += (preds[mask] == class_idx).sum().item()
            class_total[class_idx] += mask.sum().item()

    class_acc = {
        i: (class_correct[i] / max(class_total[i], 1)) * 100
        for i in range(num_classes)
    }
    overall_acc = (total_correct / max(total_samples, 1)) * 100

    return class_acc, overall_acc