import torch
import random
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    model.to(device)
    embs, labs = [], []
    for x, y in loader:
        _, h = model(x.to(device))
        embs.append(h.cpu())
        labs.append(y.cpu())
    return torch.cat(embs), torch.cat(labs)

@torch.no_grad()
def verify_embeddings(model, loader, cached_embs, device, tol=1e-3):
    model.eval()
    model.to(device)
    x, _ = next(iter(loader))
    _, h = model(x.to(device))
    h = h.cpu()
    diff = (h - cached_embs[:h.shape[0]]).abs().max().item()
    ok = diff < tol
    print(f"  Embedding check [{'OK' if ok else 'MISMATCH'}] max|diff|={diff:.2e} (tol={tol})")
    return ok

def restore_head_to_student(student, head):
    with torch.no_grad():
        student.fc.weight.copy_(head.classifier.weight)
        student.fc.bias.copy_(head.classifier.bias)
        student.norm.weight.copy_(head.norm.weight)
        student.norm.bias.copy_(head.norm.bias)
    print("  Restored: head.norm + head.classifier -> student.norm + student.fc")
    return student

@torch.no_grad()
def eval_head(head, s_embs, labels, device, batch_size=128):
    head.eval()
    loader = DataLoader(TensorDataset(s_embs, labels), batch_size=batch_size, shuffle=False)
    correct = total = 0
    for s, y in loader:
        s, y = s.to(device), y.to(device)
        logits, _ = head(s, training=False)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)

@torch.no_grad()
def calculate_accuracies(model, loader, num_classes, device):
    model.eval()
    model.to(device)
    class_correct = np.zeros(num_classes)
    class_total   = np.zeros(num_classes)
    total_correct = total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        preds = logits.argmax(1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

        for c in range(num_classes):
            mask = (y == c)
            class_correct[c] += (preds[mask] == c).sum().item()
            class_total[c]   += mask.sum().item()

    class_acc   = {i: class_correct[i] / max(class_total[i], 1) * 100 for i in range(num_classes)}
    overall_acc = total_correct / max(total_samples, 1) * 100
    return class_acc, overall_acc

@torch.no_grad()
def extract_features(model, clip_tensor, device=None):
    """
    Extract LSTM hidden states from a ResNet+LSTM model (ResNet50LSTM style).
    model must have .resnet and .lstm attributes.
    clip_tensor: (B, T, C, H, W)
    """
    if device is not None:
        clip_tensor = clip_tensor.to(device)
    B, T, C, H, W = clip_tensor.shape
    features  = model.resnet(clip_tensor.view(B * T, C, H, W))
    features  = features.view(B, T, -1)
    lstm_out, _ = model.lstm(features)
    return lstm_out[:, -1, :].cpu()


def process_dataloader(dataloader, model, features_file, labels_file, device=None):
    """
    Run extract_features over a full DataLoader and save to .pt files.
    Returns (all_features, all_labels).
    """
    if device is None:
        device = next(model.parameters()).device
    all_features, all_labels = [], []
    for clips, labels in dataloader:
        feats = extract_features(model, clips, device=device)
        all_features.append(feats)
        all_labels.append(labels.cpu())
    all_features = torch.cat(all_features, dim=0)
    all_labels   = torch.cat(all_labels, dim=0)
    torch.save(all_features, features_file)
    torch.save(all_labels, labels_file)
    print(f"Saved features to {features_file}")
    print(f"Saved labels to {labels_file}")
    return all_features, all_labels
