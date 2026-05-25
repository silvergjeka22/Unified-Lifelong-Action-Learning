import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from src.meta_learning.sampler import sample
from src.kd.utils import eval_head

def _snap(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}

def _reptile_update(model, W0, epsilon):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in W0:
                p.copy_(W0[n] + epsilon * (p - W0[n]))

def train_reptile(head, s_embs, t_embs, labels, val_s_embs, val_labels, num_classes, device,
                  reptile_epochs, episodes_per_epoch, k_support, k_query, inner_lr, inner_steps,
                  src_epsilon, ce_weight, mse_weight, reptile_patience):
    head.to(device)
    mse_fn = nn.MSELoss()
    n_way = num_classes
    best_acc, best_state, pat = 0.0, copy.deepcopy(head.state_dict()), 0

    for epoch in range(reptile_epochs):
        head.train()
        last_loss = 0.0

        for _ in range(episodes_per_epoch):
            sup_s, sup_t, sup_y, _, _ = sample(
                s_embs, t_embs, labels, n_way, k_support, k_query, device
            )

            W0 = _snap(head)
            inner_opt = optim.SGD(head.parameters(), lr=inner_lr, momentum=0.9, nesterov=True)

            for _ in range(inner_steps):
                inner_opt.zero_grad(set_to_none=True)
                logits, proj = head(sup_s, training=True)
                loss = (
                    ce_weight * F.cross_entropy(logits, sup_y, label_smoothing=0.1)
                    + mse_weight * mse_fn(proj, sup_t)
                )
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                inner_opt.step()
                last_loss = loss.item()

            _reptile_update(head, W0, src_epsilon)

        val_acc   = eval_head(head, val_s_embs, val_labels, device)
        train_acc = eval_head(head, s_embs, labels, device)
        print(f"[Reptile] Epoch {epoch+1:02d}/{reptile_epochs} | Train: {train_acc:.2%} | Val: {val_acc:.2%} | Loss: {last_loss:.4f}")

        if val_acc > best_acc:
            best_acc, best_state, pat = val_acc, copy.deepcopy(head.state_dict()), 0
        else:
            pat += 1
            if pat >= reptile_patience:
                print(f"  -> Early stop at epoch {epoch + 1}")
                break

    head.load_state_dict(best_state)
    print(f"  -> Best Reptile val acc: {best_acc:.2%}")
    return head