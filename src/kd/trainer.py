import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.kd.utils import eval_head

def finetune_head(head, train_s, train_t, train_y, val_s, val_y, device,
                  epochs, lr, wd, batch_size, patience, ce_weight, mse_weight=0.0,
                  label_smoothing=0.15):
    """
    Fine-tune a head on cached embeddings.

    train_t may be None when mse_weight == 0 (no teacher target exists before the KD
    stage). Previously the MSE term was always computed and only then multiplied by
    zero, which forced callers to fabricate a shape-compatible dummy tensor — and
    crashed outright once `train_s` became a (N, T, D) sequence while `proj` stayed
    (N, teacher_hidden). Now the term is skipped entirely.
    """
    head.to(device)
    use_mse = mse_weight > 0 and train_t is not None
    if mse_weight > 0 and train_t is None:
        raise ValueError("mse_weight > 0 requires train_t (teacher embeddings).")

    mse_fn    = nn.MSELoss()
    optimizer = optim.AdamW([p for p in head.parameters() if p.requires_grad],
                            lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)

    ds     = TensorDataset(train_s, train_t, train_y) if use_mse else TensorDataset(train_s, train_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_acc, best_state, pat = 0.0, copy.deepcopy(head.state_dict()), 0

    for epoch in range(epochs):
        head.train()
        last_loss = 0.0

        for batch in loader:
            if use_mse:
                s, t, y = (b.to(device) for b in batch)
            else:
                (s, y), t = (b.to(device) for b in batch), None
            optimizer.zero_grad(set_to_none=True)
            logits, proj = head(s, training=True)
            loss = ce_weight * F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            if use_mse:
                loss = loss + mse_weight * mse_fn(proj, t)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            last_loss = loss.item()

        scheduler.step()
        val_acc   = eval_head(head, val_s, val_y, device)
        train_acc = eval_head(head, train_s, train_y, device)
        print(f"[FineTune] Epoch {epoch+1:02d}/{epochs} | Train: {train_acc:.2%} | Val: {val_acc:.2%} | Loss: {last_loss:.4f}")

        if val_acc > best_acc:
            best_acc, best_state, pat = val_acc, copy.deepcopy(head.state_dict()), 0
        else:
            pat += 1
            if pat >= patience:
                print(f"  -> Early stop at epoch {epoch + 1}")
                break

    head.load_state_dict(best_state)
    print(f"  -> Best fine-tune val acc: {best_acc:.2%}")
    return head