import torch
from torch.utils.data import DataLoader, TensorDataset


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
def compare_head_vs_student_on_same_batch(student, head, loader, device, max_batches=1):
    student.eval()
    head.eval()
    checked = 0
    for x, _ in loader:
        x = x.to(device)
        slogits, semb = student(x)
        hlogits, _ = head(semb, training=False)
        diff = (slogits - hlogits).abs().max().item()
        print(f"  max|student_logits - head_logits| = {diff:.6e}")
        checked += 1
        if checked >= max_batches:
            break