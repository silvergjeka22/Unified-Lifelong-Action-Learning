import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.meta_learning.reptile import eval_head


def finetune_head(
    head: nn.Module,
    train_s: torch.Tensor,
    train_t: torch.Tensor,
    train_y: torch.Tensor,
    val_s: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    finetune_epochs: int,
    finetune_lr: float,
    finetune_wd: float,
    finetune_batch: int,
    finetune_patience: int,
    ce_weight: float,
    mse_weight: float,
    desc: str = "",
) -> nn.Module:
    """
    Phase 2 — Global fine-tuning of the EmbeddingHead with a combined
    cross-entropy + MSE knowledge-distillation loss.

    The optimizer is AdamW with cosine-annealing LR decay.
    Early-stopping is applied on validation accuracy.

    Args:
        head              : EmbeddingHead (output of train_reptile, or freshly initialised)
        train_s           : Training student embeddings  [N, D_s]
        train_t           : Training teacher embeddings  [N, D_t]  — KD targets
        train_y           : Training labels              [N]
        val_s             : Validation student embeddings [M, D_s]
        val_y             : Validation labels             [M]
        device            : Training device
        finetune_epochs   : Number of fine-tuning epochs
        finetune_lr       : Learning rate for AdamW
        finetune_wd       : Weight decay for AdamW
        finetune_batch    : Batch size for the training DataLoader
        finetune_patience : Early-stopping patience (epochs)
        ce_weight         : Weight for cross-entropy loss term
        mse_weight        : Weight for MSE distillation loss term
        desc              : Short string printed in progress messages

    Returns:
        head with best-validation-accuracy weights loaded
    """
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

    best_acc   = 0.0
    best_state = copy.deepcopy(head.state_dict())
    patience_counter = 0

    print(f"\n--- Running Phase 2: Global Fine-Tuning Head ({desc}) ---")

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

        val_acc = eval_head(head, val_s, val_y, device)

        if val_acc > best_acc:
            best_acc     = val_acc
            best_state   = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= finetune_patience:
                print(f"  [FineTune-EarlyStop] Triggered at Epoch {epoch + 1}")
                break

    head.load_state_dict(best_state)
    return head