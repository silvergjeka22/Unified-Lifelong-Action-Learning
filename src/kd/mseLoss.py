import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import eval_head


def finetune_head(
    head,
    train_s,
    train_t,
    train_y,
    val_s,
    val_y,
    device,
    epochs,
    lr,
    wd,
    batch_size,
    patience,
    ce_weight,
    mse_weight,
    label_smoothing=0.15,
):
    head.to(device)
    mse_fn = nn.MSELoss()
    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)

    loader = DataLoader(
        TensorDataset(train_s, train_t, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    best_acc = 0.0
    best_state = copy.deepcopy(head.state_dict())
    patience_count = 0

    for epoch in range(epochs):
        head.train()
        last_loss = 0.0

        for s, t, y in loader:
            s = s.to(device)
            t = t.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, proj = head(s, training=True)

            loss = (
                ce_weight * F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                + mse_weight * mse_fn(proj, t)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            last_loss = loss.item()

        scheduler.step()

        val_acc = eval_head(head, val_s, val_y, device)
        train_acc = eval_head(head, train_s, train_y, device)

        print(
            f"[FineTune] Epoch {epoch+1:02d}/{epochs} | "
            f"Train: {train_acc:.2%} | Val: {val_acc:.2%} | Loss: {last_loss:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stop at epoch {epoch + 1}")
                break

    head.load_state_dict(best_state)
    return head