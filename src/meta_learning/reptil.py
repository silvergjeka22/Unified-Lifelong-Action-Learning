import os
import copy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from src.fine_tune.trainer import evaluate_model

def compute_inner_loss(
    strategy,
    output,
    target,
    adapted_model,
    # EWC args
    fisher_dict=None,
    optpar_dict=None,
    ewc_lambda=5000.0,
    # Rehearsal args
    replay_buffer=None,
    teacher=None,
    num_old_classes=None,
    lambda_distill=1.0,
    T=5.0,
    kd=True,
):
    """
    Single loss function for all continual-learning strategies.

    strategy="naive"     -> plain cross-entropy
    strategy="ewc"       -> CE + Fisher-weighted penalty
    strategy="rehearsal" -> CE + replay CE + optional knowledge distillation
    """

    if strategy == "naive":
        return F.cross_entropy(output, target)

    if strategy == "ewc":
        ce      = F.cross_entropy(output, target)
        penalty = torch.tensor(0.0, device=output.device)
        for tid in fisher_dict:
            for n, p in adapted_model.named_parameters():
                if n in fisher_dict[tid]:
                    f  = fisher_dict[tid][n].to(p.device)
                    op = optpar_dict[tid][n].to(p.device)
                    penalty += (f * (p - op).pow(2)).sum()
        return ce + ewc_lambda * penalty

    if strategy == "rehearsal":
        ce           = F.cross_entropy(output, target)
        x_old, y_old = replay_buffer.sample(len(target))

        if x_old is None:
            return ce

        x_old      = x_old.to(output.device)
        y_old      = y_old.to(output.device)
        old_output = adapted_model(x_old)
        loss       = ce + F.cross_entropy(old_output, y_old)

        if kd and teacher is not None and num_old_classes is not None:
            with torch.no_grad():
                t_logits = teacher(x_old)
            s_old   = F.log_softmax(old_output[:, :num_old_classes] / T, dim=1)
            t_old   = F.softmax(t_logits[:, :num_old_classes]       / T, dim=1)
            loss_kd = F.kl_div(s_old, t_old, reduction='batchmean') * (T * T)
            loss    = loss + lambda_distill * loss_kd

        return loss

    raise ValueError(f"Unknown strategy '{strategy}'. Choose: naive | ewc | rehearsal")


def maml_fine_tune_task_real(
    model,
    meta_optimizer,
    support_loader,
    query_loader,
    inner_steps=3,
    inner_lr=1e-3,
    meta_epochs=5,
    task_name="MAML_Task",
    save_dir=None,
    strategy="naive",
    fisher_dict=None,
    optpar_dict=None,
    ewc_lambda=5000.0,
    replay_buffer=None,
    teacher=None,
    num_old_classes=None,
    lambda_distill=1.0,
    T=5.0,
    kd=True,
):
    """
    Reptile-style first-order meta-update.
    Reference: https://broutonlab.com/blog/intuitive-explanation-of-meta-learning/

    Strategy selection:
        strategy="naive"     -> plain cross-entropy inner loop
        strategy="ewc"       -> pass fisher_dict, optpar_dict, ewc_lambda
        strategy="rehearsal" -> pass replay_buffer; optionally teacher, num_old_classes
    """
    train_accs,   val_accs   = [], []
    train_losses, val_losses = [], []
    best_val_acc    = 0.0
    best_model_path = None

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, f"best_{task_name}.pt")

    print(f"MAML fine-tuning '{task_name}' [strategy: {strategy}]")

    for epoch in range(meta_epochs):
        model.train()

        # inner loop on a deep copy
        adapted_model   = copy.deepcopy(model).to(device)
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)
        avg_inner_loss  = 0.0

        for _ in range(inner_steps):
            running_loss = 0.0
            n_samples    = 0

            for data, target in tqdm(support_loader,
                                     desc=f"{task_name} Inner E{epoch + 1}",
                                     leave=False):
                data, target = data.to(device), target.to(device)

                inner_optimizer.zero_grad()
                output = adapted_model(data)

                loss = compute_inner_loss(
                    strategy        = strategy,
                    output          = output,
                    target          = target,
                    adapted_model   = adapted_model,
                    fisher_dict     = fisher_dict,
                    optpar_dict     = optpar_dict,
                    ewc_lambda      = ewc_lambda,
                    replay_buffer   = replay_buffer,
                    teacher         = teacher,
                    num_old_classes = num_old_classes,
                    lambda_distill  = lambda_distill,
                    T               = T,
                    kd              = kd,
                )

                loss.backward()
                inner_optimizer.step()

                running_loss += loss.item() * data.size(0)
                n_samples    += data.size(0)

            avg_inner_loss = running_loss / max(n_samples, 1)

        # Reptile meta-update: θ ← θ + (θ' − θ)
        with torch.no_grad():
            for p_meta, p_adapt in zip(model.parameters(),
                                       adapted_model.parameters()):
                p_meta.data += p_adapt.data - p_meta.data

        # evaluation
        train_acc, train_loss = evaluate_model(model, support_loader, device)
        val_acc,   val_loss   = evaluate_model(model, query_loader,   device)

        train_accs.append(train_acc);    val_accs.append(val_acc)
        train_losses.append(train_loss); val_losses.append(val_loss)

        print(
            f"[MAML] Epoch {epoch + 1}/{meta_epochs}: "
            f"Train Acc={train_acc:.4f}  Val Acc={val_acc:.4f}  "
            f"Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}  "
            f"Inner Loss={avg_inner_loss:.4f}"
        )

        if save_dir is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model to {best_model_path} "
                  f"(Val={best_val_acc:.4f})")

    return train_accs, val_accs, train_losses, val_losses