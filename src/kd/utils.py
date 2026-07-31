import torch
from torch.utils.data import DataLoader, TensorDataset

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
