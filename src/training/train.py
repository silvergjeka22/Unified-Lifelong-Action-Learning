import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def evaluate_model(model, loader, device):
    """(accuracy, average loss) of a model on a loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    preds, labels, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for clips, y in loader:
            clips, y = clips.to(device), y.to(device)
            out  = model(clips)
            total_loss += criterion(out, y).item() * y.size(0)
            n += y.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return accuracy_score(labels, preds), total_loss / max(n, 1)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One training epoch. Returns (accuracy, average loss)."""
    model.train()
    correct, total, total_loss = 0, 0, 0.0
    for clips, y in tqdm(loader, desc="  train", leave=False):
        clips, y = clips.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(clips)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return correct / total, total_loss / total


def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4, device="cuda"):
    """Train a model, printing train/val accuracy each epoch. Returns a history dict."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": []}

    for epoch in range(num_epochs):
        train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_loss     = evaluate_model(model, val_loader, device)
        history["train_accs"].append(train_acc)
        history["train_losses"].append(train_loss)
        history["val_accs"].append(val_acc)
        history["val_losses"].append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} Loss: {val_loss:.4f}")
    return history


def test_model(model, test_loader, device="cuda"):
    """Run inference on the test set. Returns (accuracy, preds, labels)."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for clips, y in tqdm(test_loader, desc="  test", leave=False):
            clips = clips.to(device)
            preds.extend(model(clips).argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    acc = accuracy_score(labels, preds)
    print(f"Test Accuracy: {acc:.4f}")
    return acc, preds, labels


def evaluate_all_tasks(model, device, base_loader, task_loaders=None, combined_loaders=None):
    """Accuracy on the base split, each task split, and each combined split. Returns a dict."""
    task_loaders     = task_loaders or []
    combined_loaders = combined_loaders or []
    results = {}

    acc, loss = evaluate_model(model, base_loader, device)
    results["base"] = {"accuracy": acc, "loss": loss}
    print(f"  base             -> Acc={acc:.4f}  Loss={loss:.4f}")

    for i, loader in enumerate(task_loaders):
        acc, loss = evaluate_model(model, loader, device)
        results[f"task{i+1}_only"] = {"accuracy": acc, "loss": loss}
        print(f"  task{i+1}_only        -> Acc={acc:.4f}  Loss={loss:.4f}")

    for i, loader in enumerate(combined_loaders):
        acc, loss = evaluate_model(model, loader, device)
        results[f"combined_t{i+1}"] = {"accuracy": acc, "loss": loss}
        print(f"  combined_t{i+1}       -> Acc={acc:.4f}  Loss={loss:.4f}")

    return results


def measure_latency(model, loader, device, warmup=2, runs=10):
    """Average milliseconds per clip for a single forward pass. Used by the bake-off."""
    model.eval().to(device)
    clips, _ = next(iter(loader))
    clips = clips.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(clips)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            model(clips)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - start) / (runs * clips.size(0)) * 1000
