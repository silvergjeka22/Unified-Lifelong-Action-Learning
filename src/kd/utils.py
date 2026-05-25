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
def eval_head(head, s_embs, labels, device, batch_size=128):
    head.eval()
    loader = DataLoader(TensorDataset(s_embs, labels), batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    for s, y in loader:
        s = s.to(device)
        y = y.to(device)
        logits, _ = head(s, training=False)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / max(total, 1)


@torch.no_grad()
def calculate_accuracies(model, loader, num_classes, device):
    model.eval()
    model.to(device)

    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        preds = logits.argmax(1)

        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

        for c in range(num_classes):
            mask = (y == c)
            class_correct[c] += (preds[mask] == c).sum().item()
            class_total[c] += mask.sum().item()

    class_acc = {
        i: class_correct[i].item() / max(class_total[i].item(), 1) * 100
        for i in range(num_classes)
    }
    overall_acc = total_correct / max(total_samples, 1) * 100
    return class_acc, overall_acc


def restore_head_to_student(student, head):
    with torch.no_grad():
        student.fc.weight.copy_(head.classifier.weight)
        student.fc.bias.copy_(head.classifier.bias)
        student.norm.weight.copy_(head.norm.weight)
        student.norm.bias.copy_(head.norm.bias)

    return student


def remap_teacher_checkpoint(raw_ckpt: dict) -> dict:
    remapped = {}

    for k, v in raw_ckpt.items():
        if k.endswith("num_batches_tracked"):
            continue

        if k.startswith("resnet."):
            remapped["backbone." + k[len("resnet."):]] = v
        else:
            remapped[k] = v

    return remapped