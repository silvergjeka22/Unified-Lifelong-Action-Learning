import copy
import torch
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


def train_student(head, train_s, train_y, val_s, val_y, device, mode="none",
                  teacher_logits=None, teacher_emb=None,
                  epochs=10, lr=1e-3, batch_size=64, T=2.0,
                  ce_weight=0.75, distill_weight=0.25, label_smoothing=0.0):
    """
    Train a student head under one distillation objective. Returns the best-val head.

    Ports the three methods from the PyTorch knowledge-distillation tutorial to the
    cached-feature setting:

        "none"   CE only, no teacher
        "kd"     CE + KL on softened logits * T^2   (needs teacher_logits)
        "cosine" CE + CosineEmbeddingLoss(proj, teacher_emb)   (needs teacher_emb)
        "mse"    CE + MSE(proj, teacher_emb); the projector is the regressor (needs teacher_emb)

    Weights follow the tutorial: ce_weight=0.75, distill_weight=0.25, T=2.
    """
    import copy
    from src.cl.rehearsal import distillation_loss

    head.to(device)
    optimizer = optim.Adam([p for p in head.parameters() if p.requires_grad], lr=lr)

    if mode == "kd":
        extra = teacher_logits
    elif mode in ("cosine", "mse"):
        extra = teacher_emb
    else:
        extra = torch.zeros(len(train_y), 1)

    loader = DataLoader(TensorDataset(train_s, train_y, extra), batch_size=batch_size, shuffle=True)
    best_acc, best_state = 0.0, copy.deepcopy(head.state_dict())

    for epoch in range(epochs):
        head.train()
        for sb, yb, eb in loader:
            sb, yb, eb = sb.to(device), yb.to(device), eb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, proj = head(sb, training=True)
            ce = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)

            if mode == "kd":
                distill = distillation_loss(logits, eb, T=T)
                loss = ce_weight * ce + distill_weight * distill
            elif mode == "cosine":
                target = torch.ones(sb.size(0), device=device)
                distill = F.cosine_embedding_loss(proj, eb, target)
                loss = ce_weight * ce + distill_weight * distill
            elif mode == "mse":
                distill = F.mse_loss(proj, eb)
                loss = ce_weight * ce + distill_weight * distill
            else:
                loss = ce

            loss.backward()
            nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad], 1.0)
            optimizer.step()

        val_acc = eval_head(head, val_s, val_y, device)
        print(f"[{mode:>6}] epoch {epoch + 1:02d}/{epochs} | val {val_acc:.2%}")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    head.eval()
    print(f"  -> [{mode}] best val {best_acc:.2%}")
    return head