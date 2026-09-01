import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.train import evaluate_model


def distillation_loss(student_logits, teacher_logits, T=5.0):
    """KL divergence on temperature-softened logits, scaled by T^2 (Hinton et al.)."""
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


class ReplayBuffer:
    """A small store of past clips, kept to a fixed size, sampled to mix into new-task batches."""

    def __init__(self, max_size=300):
        self.max_size = max_size
        self.data = []

    def add_batch(self, x, y):
        for i in range(len(x)):
            self.data.append((x[i].cpu(), y[i].cpu()))
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]

    def sample(self, n):
        if not self.data:
            return None, None
        picks = random.sample(self.data, min(n, len(self.data)))
        x, y = zip(*picks)
        return torch.stack(x), torch.tensor(y)


def fill_buffer(buffer, loader, per_class=None):
    """Add clips from a loader into the buffer (optionally capping samples per class)."""
    seen = {}
    for clips, y in loader:
        if per_class is None:
            buffer.add_batch(clips, y)
            continue
        keep_x, keep_y = [], []
        for i in range(len(y)):
            c = int(y[i])
            if seen.get(c, 0) < per_class:
                keep_x.append(clips[i]); keep_y.append(y[i]); seen[c] = seen.get(c, 0) + 1
        if keep_x:
            buffer.add_batch(torch.stack(keep_x), torch.tensor(keep_y))
    return buffer


def train_continual(model, train_loader, val_loader, device, buffer=None, teacher=None,
                    num_old_classes=10, lambda_distill=1.0, T=5.0, epochs=5, lr=1e-4):
    """
    Train on a new task with optional replay and optional LwF distillation.

        buffer only          -> replay
        teacher only         -> LwF (soft targets on old-class logits)
        buffer + teacher     -> replay + LwF
        neither              -> naive (use train_model instead)

    Returns a history dict.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    if teacher is not None:
        teacher.to(device).eval()
    history = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": []}

    for epoch in range(epochs):
        model.train()
        correct, total, run_loss = 0, 0, 0.0
        for x_new, y_new in train_loader:
            x, y = x_new.to(device), y_new.to(device)
            if buffer is not None:
                x_old, y_old = buffer.sample(len(x_new))
                if x_old is not None:
                    x = torch.cat([x, x_old.to(device)])
                    y = torch.cat([y, y_old.to(device)])

            logits = model(x)
            loss = ce(logits, y)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x)
                loss = loss + lambda_distill * distillation_loss(
                    logits[:, :num_old_classes], t_logits[:, :num_old_classes], T)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)

        train_acc, train_loss = correct / total, run_loss / total
        val_acc, val_loss     = evaluate_model(model, val_loader, device)
        history["train_accs"].append(train_acc)
        history["train_losses"].append(train_loss)
        history["val_accs"].append(val_acc)
        history["val_losses"].append(val_loss)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} Loss: {val_loss:.4f}")
    return history
