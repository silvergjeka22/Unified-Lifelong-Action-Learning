import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.meta_learning.sampler import sample


def _snap(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _reptile_update(model, W0, epsilon):
    # Reptile outer update:  θ <- θ_0 + ε · (θ_inner − θ_0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in W0:
                p.copy_(W0[name] + epsilon * (p - W0[name]))


@torch.no_grad()
def eval_head(head, s_embs, labels, device):
    head.eval()
    loader = DataLoader(TensorDataset(s_embs, labels), batch_size=128, shuffle=False)
    correct = total = 0
    for s, y in loader:
        s, y = s.to(device), y.to(device)
        logits, _ = head(s, training=False)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return correct / max(total, 1)


def train_reptile(
    head,
    s_embs,
    t_embs,
    labels,
    val_s_embs,
    val_labels,
    num_classes,
    device,
    reptile_epochs,
    episodes_per_epoch,
    k_support,
    k_query,
    inner_lr,
    inner_steps,
    src_epsilon,
    ce_weight,
    mse_weight,
    reptile_patience,
):
    head.to(device)
    mse_fn = nn.MSELoss()
    n_way  = num_classes

    best_acc         = 0.0
    best_state       = copy.deepcopy(head.state_dict())
    patience_counter = 0

    for epoch in range(reptile_epochs):
        head.train()

        for _ in range(episodes_per_epoch):
            sup_s, sup_t, sup_y, _, _ = sample(
                s_embs, t_embs, labels, n_way, k_support, k_query, device
            )

            W0 = _snap(head)

            inner_opt = optim.SGD(
                head.parameters(), lr=inner_lr, momentum=0.9, nesterov=True
            )
            for _ in range(inner_steps):
                inner_opt.zero_grad(set_to_none=True)
                logits, proj = head(sup_s, training=True)
                ce  = F.cross_entropy(logits, sup_y, label_smoothing=0.1)
                mse = mse_fn(proj, sup_t)
                (ce_weight * ce + mse_weight * mse).backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                inner_opt.step()

            _reptile_update(head, W0, src_epsilon)

        val_acc   = eval_head(head, val_s_embs, val_labels, device)
        train_acc = eval_head(head, sup_s, sup_y, device)
        print(f"Val_Acc: {val_acc:.4f} | Train_Acc: {train_acc:.4f} | Loss: {ce.item():.4f}")

        if val_acc > best_acc:
            best_acc         = val_acc
            best_state       = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= reptile_patience:
                print(f"EarlyStop at epoch {epoch + 1}")
                break

    head.load_state_dict(best_state)
    return head