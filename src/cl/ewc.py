import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.training.train import evaluate_model


def theta_star(model):
    """Snapshot the trainable weights after a task - the anchor EWC pulls toward."""
    return {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}


def get_fisher(train_loader, model, device):
    """
    Diagonal Fisher information: the average squared gradient per weight over the
    task's data. Large values mark the weights that matter for the old task.
    """
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.train().to(device)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()

    for clips, y in tqdm(train_loader, desc="  fisher", leave=False):
        clips, y = clips.to(device), y.to(device)
        model.zero_grad()
        F.cross_entropy(model(clips), y).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2)

    for n in fisher:
        fisher[n] = (fisher[n] / len(train_loader.dataset)).clamp(0.0, 1.0)
    return fisher


def ewc_loss(model, star, fisher, ewc_lambda, device):
    """EWC penalty: lambda * sum F * (theta - theta_star)^2, over the old weights."""
    loss = torch.zeros((), device=device)
    for n, p in model.named_parameters():
        if n in star and n in fisher:
            old = star[n].to(device)
            f   = fisher[n].to(device)
            c   = old.shape[0]                 # the classifier row count may have grown
            loss = loss + (f[:c] * (p[:c] - old).pow(2)).sum()
    return ewc_lambda * loss


def train_ewc(model, train_loader, val_loader, old_val_loader, star, fisher,
              device, num_epochs=10, lr=1e-5, ewc_lambda=5000.0):
    """
    Train on a new task with the EWC penalty. Prints new-task and old-task validation
    accuracy each epoch so the protection is visible. Returns a history dict.
    """
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": []}

    for epoch in range(num_epochs):
        model.train()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
        correct, total, run_loss = 0, 0, 0.0
        for clips, y in tqdm(train_loader, desc="  train", leave=False):
            clips, y = clips.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(clips)
            loss = criterion(out, y) + ewc_loss(model, star, fisher, ewc_lambda, device)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            correct  += (out.argmax(1) == y).sum().item()
            total    += y.size(0)

        train_acc, train_loss = correct / total, run_loss / total
        val_acc, val_loss     = evaluate_model(model, val_loader, device)
        old_acc, _            = evaluate_model(model, old_val_loader, device)
        history["train_accs"].append(train_acc)
        history["train_losses"].append(train_loss)
        history["val_accs"].append(val_acc)
        history["val_losses"].append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Acc: {train_acc:.4f} | "
              f"New Val: {val_acc:.4f} | Old Val: {old_acc:.4f}")
    return history
