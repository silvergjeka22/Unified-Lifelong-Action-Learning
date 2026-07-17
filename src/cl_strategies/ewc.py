import torch
import torch.nn as nn
import torch.nn.functional as F
from fine_tune.trainer import evaluate_model
from tqdm import tqdm

# global EWC
EWC_LAMBDA  = 5000.0
fisher_dict = {}
optpar_dict = {}


# fisher computation 
def compute_fisher(model, dataloader, device, task_id, fisher_dict, optpar_dict):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
            m.eval()

    fisher_dict[task_id] = {
        n: torch.zeros_like(p)
        for n, p in model.named_parameters() if p.requires_grad
    }
    optpar_dict[task_id] = {
        n: p.data.clone()
        for n, p in model.named_parameters() if p.requires_grad
    }

    seen = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss = F.cross_entropy(model(x), y)     # real labels
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher_dict[task_id][n] += p.grad.data.pow(2) * x.size(0)
        seen += x.size(0)

    # Normalize per-layer to [0, 1] with full NaN safety
    for n in fisher_dict[task_id]:
        fisher_dict[task_id][n] /= max(seen, 1)
        f_max = fisher_dict[task_id][n].max()
        if f_max > 0 and not torch.isnan(f_max):
            fisher_dict[task_id][n] /= f_max
        if torch.isnan(fisher_dict[task_id][n]).any():
            fisher_dict[task_id][n].zero_()

    model.eval()
    mean_f = sum(
        v.abs().mean().item() for v in fisher_dict[task_id].values()
    ) / len(fisher_dict[task_id])
    print(f"[Fisher] task_id={task_id} | samples={seen} | mean={mean_f:.6f}")
    return fisher_dict, optpar_dict


# EWC lambda sanity check (call before any training loop)
def check_ewc_penalty(model, fisher_dict, optpar_dict, ewc_lambda, device):
    penalty = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for tid in fisher_dict:
            for n, p in model.named_parameters():
                if n in fisher_dict[tid]:
                    f       = fisher_dict[tid][n].to(device)
                    op      = optpar_dict[tid][n].to(device)
                    contrib = (f * (p - op).pow(2)).sum()
                    if not torch.isnan(contrib):
                        penalty = penalty + contrib
    raw    = penalty.item()
    scaled = ewc_lambda * raw
    status = " may cause NaN -> reduce λ or grad_clip" if scaled > 1e4 else "OK"
    print(
        f"\n[EWC pre-check] raw={raw:.6f} | "
        f"scaled (λ={ewc_lambda})={scaled:.4f} | {status}\n"
    )
    return raw, scaled


# Pad Fisher,optpar after FC expansion
def pad_fisher_after_expand(fisher_dict, optpar_dict, model):
    """
    After expand_classifier(), pad stored Fisher/optpar tensors to match
    new parameter shapes. Handles both 1-D (bias) and 2-D (weight) params.
    """
    cur = {n: p.data for n, p in model.named_parameters() if p.requires_grad}
    for tid in fisher_dict:
        for n in list(fisher_dict[tid].keys()):
            stored_shape  = fisher_dict[tid][n].shape
            current_shape = cur[n].shape
            if stored_shape == current_shape:
                continue

            new_fisher = torch.zeros(current_shape, device=fisher_dict[tid][n].device)
            new_optpar = torch.zeros(current_shape, device=optpar_dict[tid][n].device)

            # multi-dimensional parms
            slices = tuple(slice(0, s) for s in stored_shape)
            new_fisher[slices] = fisher_dict[tid][n]
            new_optpar[slices] = optpar_dict[tid][n]

            fisher_dict[tid][n] = new_fisher
            optpar_dict[tid][n] = new_optpar
            print(f"  [pad] {n}: {stored_shape} -> {current_shape}")
    return fisher_dict, optpar_dict


# EWC training loop
def train_ewc(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    fisher_dict,
    optpar_dict,
    ewc_lambda=EWC_LAMBDA,
    epochs=10,
    task_label="Task",
):
    for epoch in range(1, epochs + 1):
        model.train()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
                m.eval()

        total_loss = total_ce = total_ewc = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            ce      = F.cross_entropy(model(x), y)
            ewc_pen = torch.tensor(0.0, device=device)
            for tid in fisher_dict:
                for n, p in model.named_parameters():
                    if n in fisher_dict[tid]:
                        f       = fisher_dict[tid][n].to(device)
                        op      = optpar_dict[tid][n].to(device)
                        contrib = (f * (p - op).pow(2)).sum()
                        if not torch.isnan(contrib):
                            ewc_pen = ewc_pen + contrib

            loss = ce + ewc_lambda * ewc_pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce   += ce.item()
            total_ewc  += ewc_pen.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                correct += (model(x_v).argmax(1) == y_v).sum().item()
                total   += y_v.size(0)

        nb = len(train_loader)
        print(
            f"[{task_label}] Epoch {epoch:02d}/{epochs} | "
            f"Loss {total_loss/nb:.4f} "
            f"(CE {total_ce/nb:.4f} + EWC {ewc_lambda * total_ewc/nb:.4f}) | "
            f"Val Acc {correct/total:.4f}"
        )


# fisher diagnostic report
def ewc_report(model, fisher_dict, optpar_dict):
    """Per-task Fisher magnitude, weight drift and penalty breakdown."""
    cur = {n: p.data.cpu().float() for n, p in model.named_parameters() if p.requires_grad}
    print("\n" + "=" * 60)
    for tid in fisher_dict:
        names   = list(fisher_dict[tid].keys())
        f_mag   = [fisher_dict[tid][n].cpu().float().abs().mean().item() for n in names]
        drift   = [
            (cur[n] - optpar_dict[tid][n].cpu().float()).pow(2).mean().item()
            for n in names
        ]
        penalty = [
            (
                fisher_dict[tid][n].cpu().float()
                * (cur[n] - optpar_dict[tid][n].cpu().float()).pow(2)
            ).sum().item()
            for n in names
        ]
        total_p = sum(penalty)
        print(f"  Task {tid} | Mean Fisher: {sum(f_mag)/len(f_mag):.4f} | "
              f"Mean Drift: {sum(drift)/len(drift):.6f} | Total Penalty: {total_p:.4f}")
        for rank, i in enumerate(sorted(range(len(penalty)), key=lambda i: penalty[i], reverse=True)[:5], 1):
            print(f"    {rank}. {names[i][-45:]:<45}  {100*penalty[i]/(total_p+1e-12):5.1f}%")
    print("=" * 60)



def theta_star(model):
    "Where si the model."

    theta_star = {
        n: p.clone().detach()
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    return theta_star


def getFisherDiagonal(train_loader, model, fishermax=1e6, device='cuda'):

    fisher = {
        n: torch.zeros_like(p, device=device)
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    model.train()
    model.to(device)


    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


    for inputs, targets in tqdm(train_loader, desc="Computing Fisher", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)

        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2)

    # average over batches (standard)
    for n in fisher:
        fisher[n] /= len(train_loader.dataset)
        fisher[n] = torch.clamp(fisher[n], min=0.0, max=1.0)

    print(f"Fisher diagonal computed ")

    return fisher



def compute_ewc_loss(model, theta_star, fisher, lambda_ewc=5000, device='cuda'):
    ewc_loss = 0.0

    for n, p in model.named_parameters():
        if p.requires_grad and n in theta_star and n in fisher:
            old = theta_star[n].to(device)
            f = fisher[n].to(device)

            # SAFE handling
            if p.shape == old.shape:
                ewc_loss += (f * (p - old).pow(2)).sum()
            else:
                # handle classifier expansion
                if len(p.shape) == 2:  # weight [C, D]
                    c_old = old.shape[0]
                    ewc_loss += (f[:c_old] *
                                 (p[:c_old] - old).pow(2)).sum()

                elif len(p.shape) == 1:  # bias [C]
                    c_old = old.shape[0]
                    ewc_loss += (f[:c_old] *
                                 (p[:c_old] - old).pow(2)).sum()

    return lambda_ewc * ewc_loss



def train_one_epoch_EWC(model, train_loader, criterion, optimizer, device, theta_star, fisher):
    """
    Handles the training loop for a single epoch.
    """
    model.train()

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    running_loss, correct, total = 0.0, 0, 0

    for clips, labels in tqdm(train_loader, desc="  Training", leave=False):

        clips, labels = clips.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels) + compute_ewc_loss(model, theta_star, fisher)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total, running_loss / total



def train_cl_EWC(model, new_task_train_loader, new_task_val_loader, old_task_val_loader, device='cuda', num_epochs=10, lr=1e-5, theta_star=None, fisher=None):
    """
    
        new_task_train_loader: DataLoader for the new task training set
        new_task_val_loader: DataLoader for the new task validation set
        old_task_val_loader: DataLoader for the old task validation set
        device: Device to run the training on
        num_epochs: Number of epochs to train for
        lr: Learning rate
        theta_star: Dictionary of optimal parameters
        fisher: Dictionary of Fisher information matrices

    """

    model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()


    history = {
        'train_losses': [], 'val_losses': [],
        'train_accs': [], 'val_accs': [],
        'best_val_acc': 0.0
    }


    for epoch in range(num_epochs):
        
        train_acc, train_loss = train_one_epoch_EWC(model, new_task_train_loader, criterion, optimizer, device, theta_star,fisher)
        # Validation on new task
        
        val_acc, val_loss = evaluate_model(model, new_task_val_loader, device)

        old_task_acc, old_task_loss = evaluate_model(model, old_task_val_loader, device)

        

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} Loss: {val_loss:.4f} | "
              f"Old task Acc: {old_task_acc:.4f} Oldtask Loss {old_task_loss:.4f} ")

        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)

    return history
