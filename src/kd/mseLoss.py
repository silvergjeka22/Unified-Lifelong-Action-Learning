import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.meta_learning.reptile import eval_head


def finetune_head(
    head,
    train_s,
    train_t,
    train_y,
    val_s,
    val_y,
    device,
    finetune_epochs,
    finetune_lr,
    finetune_wd,
    finetune_batch,
    finetune_patience,
    ce_weight,
    mse_weight,
):
    head.to(device)
    mse_fn    = nn.MSELoss()
    optimizer = optim.AdamW(head.parameters(), lr=finetune_lr, weight_decay=finetune_wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs, eta_min=finetune_lr / 20
    )

    loader = DataLoader(
        TensorDataset(train_s, train_t, train_y),
        batch_size=finetune_batch,
        shuffle=True,
    )

    best_acc         = 0.0
    best_state       = copy.deepcopy(head.state_dict())
    patience_counter = 0

    for epoch in range(finetune_epochs):
        head.train()

        for s, t, y in loader:
            s, t, y = s.to(device), t.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, proj = head(s, training=True)
            ce   = F.cross_entropy(logits, y, label_smoothing=0.15)
            mse  = mse_fn(proj, t)
            loss = ce_weight * ce + mse_weight * mse
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        val_acc   = eval_head(head, val_s, val_y, device)
        train_acc = eval_head(head, train_s, train_y, device)
        print(f"Train_Acc: {train_acc:.2%} | Val_Acc: {val_acc:.2%} | Loss: {loss.item():.4f}")

        if val_acc > best_acc:
            best_acc         = val_acc
            best_state       = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= finetune_patience:
                print(f"EarlyStop at Epoch {epoch + 1}")
                break

    head.load_state_dict(best_state)
    return head