import torch
import numpy as np
import torch.nn as nn


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    model.to(device)
    embs, labs = [], []

    for x, y in loader:
        _, h = model(x.to(device))
        embs.append(h.cpu())
        labs.append(y.cpu())

    return torch.cat(embs, 0), torch.cat(labs, 0)


def extract_all(teacher, student, loaders, device):
    t_raw: dict = {}
    s_raw: dict = {}

    for split, loader in loaders.items():
        te, tl = extract_embeddings(teacher, loader, device)
        se, sl = extract_embeddings(student, loader, device)
        t_raw[split] = (te, tl)
        s_raw[split] = (se, sl)

    return t_raw, s_raw


def restore_head_to_student(student, head):
    with torch.no_grad():
        student.fc.weight.copy_(head.classifier.weight)
        student.fc.bias.copy_(head.classifier.bias)
    print(" Restored Head Linear into student.fc")
    return student


@torch.no_grad()
def calculate_accuracies(model, loader, num_classes, device):
    model.eval()
    model.to(device)

    class_correct = np.zeros(num_classes)
    class_total   = np.zeros(num_classes)
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y      = x.to(device), y.to(device)
        logits, _ = model(x)
        preds     = logits.argmax(dim=1)

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