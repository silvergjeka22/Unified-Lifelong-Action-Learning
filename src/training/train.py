import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def free_gpu(device=None):
    """Release cached CUDA blocks so the next task starts with room (called between tasks)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def task_accs(model, loaders, device):
    """Accuracy of the model on each named loader, as {name: acc}. Builds a forgetting timeline."""
    return {name: evaluate_model(model, ldr, device)[0] for name, ldr in loaders.items()}


def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4, device="cuda",
                track_loaders=None, track_history=None):
    """
    Train a model, printing train/val accuracy each epoch. Returns a history dict.

    If track_loaders ({task_name: test_loader}) and track_history (a list) are given, one
    {task_name: accuracy} snapshot is appended per epoch (plus a baseline before the first
    epoch when the list is empty) so plot_forgetting_history can chart the whole stream.
    """
    model.to(device)
    free_gpu(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": []}

    if track_loaders and track_history is not None and len(track_history) == 0:
        track_history.append(task_accs(model, track_loaders, device))

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
        if track_loaders and track_history is not None:
            track_history.append(task_accs(model, track_loaders, device))
    return history


def test_model(model, test_loader, device="cuda", verbose=True):
    """Run inference on the test set. Returns (accuracy, preds, labels). verbose=False mutes the
    Test Accuracy print (used when scoring many models in a loop for a single table)."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for clips, y in tqdm(test_loader, desc="  test", leave=False):
            clips = clips.to(device)
            preds.extend(model(clips).argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    acc = accuracy_score(labels, preds)
    if verbose:
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
