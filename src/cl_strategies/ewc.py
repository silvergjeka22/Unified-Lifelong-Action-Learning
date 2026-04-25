import torch
import torch.nn as nn
import torch.nn.functional as F


# global EWC
EWC_LAMBDA   = 5000.0
fisher_dict  = {}
optpar_dict  = {}


# Fisher computation (call BEFORE expanding FC)
def compute_fisher(model, dataloader, device, task_id, fisher_dict, optpar_dict):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
            m.eval()

    fisher_dict[task_id] = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    optpar_dict[task_id] = {n: p.data.clone()       for n, p in model.named_parameters() if p.requires_grad}

    seen = 0
    for x, y in dataloader:
        x = x.to(device)
        model.zero_grad()
        log_p = F.log_softmax(model(x), dim=1)
        F.nll_loss(log_p, log_p.argmax(1).detach()).backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher_dict[task_id][n] += p.grad.data.pow(2) * x.size(0)
        seen += x.size(0)

    # Normalize each layer to [0, 1] so all layers are equally protected
    for n in fisher_dict[task_id]:
        fisher_dict[task_id][n] /= seen
        f_max = fisher_dict[task_id][n].max()
        if f_max > 0:
            fisher_dict[task_id][n] /= f_max

    model.eval()
    mean_f = sum(v.abs().mean().item() for v in fisher_dict[task_id].values()) / len(fisher_dict[task_id])
    print(f"[Fisher] task_id={task_id} | samples={seen} | mean={mean_f:.4f}")
    return fisher_dict, optpar_dict


# Pad Fisher/optpar after FC expansion 
def pad_fisher_after_expand(fisher_dict, optpar_dict, model):
    cur = {n: p.data for n, p in model.named_parameters() if p.requires_grad}
    for tid in fisher_dict:
        for n in list(fisher_dict[tid].keys()):
            stored_shape  = fisher_dict[tid][n].shape
            current_shape = cur[n].shape
            if stored_shape == current_shape:
                continue
            new_fisher = torch.zeros(current_shape, device=fisher_dict[tid][n].device)
            new_optpar = torch.zeros(current_shape, device=optpar_dict[tid][n].device)
            new_fisher[:stored_shape[0]] = fisher_dict[tid][n]
            new_optpar[:stored_shape[0]] = optpar_dict[tid][n]
            fisher_dict[tid][n] = new_fisher
            optpar_dict[tid][n] = new_optpar
            print(f"  [pad] {n}: {stored_shape} -> {current_shape}")
    return fisher_dict, optpar_dict


# EWC training loop
def train_ewc(model, train_loader, val_loader,
                     optimizer, device, fisher_dict, optpar_dict,
                     ewc_lambda=EWC_LAMBDA, epochs=10, task_label="Task"):
    for epoch in range(1, epochs + 1):
        model.train()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
                m.eval()

        total_loss = total_ce = total_ewc = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            ce = F.cross_entropy(model(x), y)
            ewc_pen = torch.tensor(0.0, device=device)
            for tid in fisher_dict:
                for n, p in model.named_parameters():
                    if n in fisher_dict[tid]:
                        ewc_pen += (fisher_dict[tid][n] * (p - optpar_dict[tid][n]).pow(2)).sum()
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
        print(f"[{task_label}] Epoch {epoch:02d}/{epochs} | "
              f"Loss {total_loss/nb:.4f} "
              f"(CE {total_ce/nb:.4f} + EWC {ewc_lambda * total_ewc/nb:.4f}) | "
              f"Val Acc {correct/total:.4f}")


# Fisher diagnostic report
def ewc_report(model, fisher_dict, optpar_dict):
    cur = {n: p.data.cpu().float() for n, p in model.named_parameters() if p.requires_grad}
    print("\n" + "=" * 60)
    for tid in fisher_dict:
        names   = list(fisher_dict[tid].keys())
        f_mag   = [fisher_dict[tid][n].cpu().float().abs().mean().item() for n in names]
        drift   = [(cur[n] - optpar_dict[tid][n].cpu().float()).pow(2).mean().item() for n in names]
        penalty = [(fisher_dict[tid][n].cpu().float() * (cur[n] - optpar_dict[tid][n].cpu().float()).pow(2)).sum().item() for n in names]
        total_p = sum(penalty)
        print(f"  Task {tid} | Mean Fisher: {sum(f_mag)/len(f_mag):.4f} | "
              f"Mean Drift: {sum(drift)/len(drift):.6f} | Total Penalty: {total_p:.4f}")
        for rank, i in enumerate(sorted(range(len(penalty)), key=lambda i: penalty[i], reverse=True)[:5], 1):
            print(f"    {rank}. {names[i][-45:]:<45}  {100*penalty[i]/(total_p+1e-12):5.1f}%")
    print("=" * 60)