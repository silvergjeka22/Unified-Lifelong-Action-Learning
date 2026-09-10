import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.training.train import evaluate_model, task_accs, free_gpu


def theta_star(model):
    """Snapshot the trainable weights after a task - the anchor EWC pulls toward."""
    return {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}


def get_fisher(train_loader, model, device):
    """
    Diagonal Fisher information: the mean squared gradient per weight over the task's
    data, then normalised per layer to [0, 1] so the most important weights sit near 1.
    Without that normalisation the raw values are tiny and the EWC penalty is inert.
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
        fisher[n] /= max(len(train_loader.dataset), 1)
        peak = fisher[n].max()
        if peak > 0 and not torch.isnan(peak):
            fisher[n] /= peak                       # per-layer -> [0, 1]
        if torch.isnan(fisher[n]).any():
            fisher[n].zero_()
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
              device, num_epochs=10, lr=1e-5, ewc_lambda=5000.0,
              track_loaders=None, track_history=None):
    """
    Train on a new task with the EWC penalty. Prints new-task and old-task validation
    accuracy each epoch so the protection is visible. Returns a history dict.

    track_loaders / track_history: see train_model - fills a per-epoch forgetting timeline.
    """
    model.to(device)
    free_gpu(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": []}

    if track_loaders and track_history is not None and len(track_history) == 0:
        track_history.append(task_accs(model, track_loaders, device))

    for epoch in range(num_epochs):
        model.train()   # dropout ON during training, same as every other arm
        correct, total = 0, 0
        ce_sum, pen_sum, nb = 0.0, 0.0, 0
        for clips, y in tqdm(train_loader, desc="  train", leave=False):
            clips, y = clips.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(clips)
            ce   = criterion(out, y)
            pen  = ewc_loss(model, star, fisher, ewc_lambda, device)   # already scaled by lambda
            (ce + pen).backward()
            optimizer.step()
            ce_sum  += ce.item()
            pen_sum += pen.item()
            nb      += 1
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)

        train_acc = correct / total
        ce_mean, pen_mean = ce_sum / max(nb, 1), pen_sum / max(nb, 1)
        val_acc, val_loss = evaluate_model(model, val_loader, device)
        old_acc, _        = evaluate_model(model, old_val_loader, device)
        history["train_accs"].append(train_acc)
        history["train_losses"].append(ce_mean)
        history["val_accs"].append(val_acc)
        history["val_losses"].append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train {train_acc:.4f} | "
              f"CE {ce_mean:.3f} | EWC {pen_mean:.3f} ({100*pen_mean/max(ce_mean,1e-9):.0f}% of CE) | "
              f"New Val {val_acc:.4f} | Old Val {old_acc:.4f}")
        if track_loaders and track_history is not None:
            track_history.append(task_accs(model, track_loaders, device))
    return history
