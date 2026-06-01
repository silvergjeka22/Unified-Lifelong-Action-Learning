import torch
import torch.nn as nn
import torch.nn.functional as F


def train_naive(model, train_loader, val_loader, optimizer, device, epochs=10, task_label="Task"):
    """
    Naive sequential fine-tuning — no forgetting mitigation.
    Used as the lower-bound continual learning baseline.

    Args:
        model        : PyTorch model with expandable head
        train_loader : DataLoader for current task
        val_loader   : DataLoader for current task validation
        optimizer    : pre-built optimizer
        device       : torch.device
        epochs       : number of training epochs
        task_label   : string label for logging

    Returns:
        history dict with train_accs, val_accs, train_losses, val_losses
    """
    model.to(device)
    history = {"train_accs": [], "val_accs": [], "train_losses": [], "val_losses": []}

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = total_correct = total_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss    += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += y.size(0)

        train_acc  = total_correct / total_samples
        train_loss = total_loss    / total_samples

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = val_correct = val_samples = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                val_loss    += F.cross_entropy(logits, y).item() * y.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_samples += y.size(0)

        val_acc  = val_correct / val_samples
        val_loss = val_loss    / val_samples

        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)

        print(f"[{task_label}] Epoch {epoch:02d}/{epochs} | "
              f"Train {train_acc:.4f} ({train_loss:.4f}) | "
              f"Val {val_acc:.4f} ({val_loss:.4f})")

    return history
