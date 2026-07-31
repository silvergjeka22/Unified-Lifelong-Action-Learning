import torch
import torch.nn as nn
import torch.nn.functional as F


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
