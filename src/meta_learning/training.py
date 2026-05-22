import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
from src.meta_learning.models import evaluate, distillation_loss, make_reptile_outer_optimizer

def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def _reptile_update(model, W_start, epsilon):
    sd = model.state_dict()
    with torch.no_grad():
        for k in W_start:
            sd[k] = W_start[k] + epsilon * (sd[k] - W_start[k])
    model.load_state_dict(sd)


def _safe_ce(logits, targets, num_classes, label_smoothing=0.0):
    targets = targets.clamp(0, num_classes - 1)
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


def _clip_step(optimizer, model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()


def _sample_replay(replay_buffer, batch_size, ratio, device):
    n_old = int(batch_size * ratio)
    if n_old <= 0:
        return None, None
    x_old, y_old = replay_buffer.sample(n_old)
    if x_old is None:
        return None, None
    return x_old.to(device), y_old.to(device)


def train_rehearsal(
    model,
    teacher,
    train_loader,
    val_loader,
    replay_buffer,
    optimizer,
    device,
    num_old_classes,
    num_classes=3,
    epochs=5,
    new_repeat=3,
    replay_ratio=0.5,
    lambda_distill=0.5,
    label_smoothing=0.1,
    kd=True,
    patience=3,
):
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val, patience_count, best_state = 0.0, 0, None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for _ in range(new_repeat):
            for x_new, y_new in train_loader:
                x_new, y_new = x_new.to(device), y_new.to(device)
                x_old, y_old = _sample_replay(replay_buffer, len(x_new), replay_ratio, device)

                if x_old is not None:
                    x = torch.cat([x_new, x_old])
                    y = torch.cat([y_new, y_old])
                else:
                    x, y = x_new, y_new

                logits = model(x)
                loss = _safe_ce(logits, y, num_classes, label_smoothing)

                if kd and teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss += lambda_distill * distillation_loss(
                        logits[:, :num_old_classes],
                        t_logits[:, :num_old_classes]
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _clip_step(optimizer, model)
                total_loss += loss.item()

        scheduler.step()
        _, val_acc = evaluate(model, val_loader, device)
        print(f"[Rehearsal] Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | ValAcc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[Rehearsal] Early stop at epoch {epoch+1}, best ValAcc: {best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_reptile(
    model,
    episode_buffer,
    train_loader,
    val_loader,
    device,
    teacher=None,
    num_old_classes=None,
    num_classes=3,
    n_way=10,
    k_support=4,
    k_query=16,
    inner_lr=0.01,
    inner_steps=5,
    epsilon=0.1,
    new_repeat=4,
    epochs=5,
    lambda_kd=0.5,
    lambda_qry=0.5,
    lambda_new=0.5,
    label_smoothing=0.1,
    kd=False,
    patience=3,
):
    outer_opt = make_reptile_outer_optimizer(model)
    scheduler = CosineAnnealingLR(outer_opt, T_max=epochs)
    base_outer_lr = outer_opt.param_groups[0]['lr']
    best_val, patience_count, best_state = 0.0, 0, None

    for epoch in range(epochs):
        model.train()

        for _ in range(new_repeat):
            for x_new, y_new in train_loader:
                x_new, y_new = x_new.to(device), y_new.to(device)
                W_start = _snapshot(model)

                sup_x, sup_y, qry_x, qry_y = episode_buffer.sample_episode(
                    n_way=n_way, k_support=k_support, k_query=k_query, device=device
                )

                current_outer_lr = outer_opt.param_groups[0]['lr']
                adapted_inner_lr = inner_lr * (current_outer_lr / base_outer_lr)
                adapted_epsilon  = epsilon  * (current_outer_lr / base_outer_lr)

                inner_opt = torch.optim.SGD(model.parameters(), lr=adapted_inner_lr)

                for _ in range(inner_steps):
                    logits = model(sup_x)
                    loss = _safe_ce(logits, sup_y, num_classes, label_smoothing)

                    if kd and teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(sup_x)
                        loss += lambda_kd * distillation_loss(
                            logits[:, :num_old_classes],
                            t_logits[:, :num_old_classes]
                        )

                    inner_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    inner_opt.step()

                _reptile_update(model, W_start, adapted_epsilon)

                outer_opt.zero_grad(set_to_none=True)
                loss_out = (
                    lambda_qry * _safe_ce(model(qry_x), qry_y, num_classes, label_smoothing)
                    + lambda_new * _safe_ce(model(x_new), y_new, num_classes, label_smoothing)
                )
                loss_out.backward()
                _clip_step(outer_opt, model)

        scheduler.step()
        _, val_acc = evaluate(model, val_loader, device)
        print(f"[Reptile] Epoch {epoch+1}/{epochs} | ValAcc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[Reptile] Early stop at epoch {epoch+1}, best ValAcc: {best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def train_hybrid(
    model,
    episode_buffer,
    replay_buffer,
    train_loader,
    val_loader,
    device,
    teacher=None,
    num_old_classes=None,
    num_classes=3,
    n_way=10,
    k_support=4,
    k_query=16,
    inner_lr=0.01,
    inner_steps=5,
    epsilon=0.1,
    new_repeat=4, 
    epochs=5,
    replay_ratio=0.5,
    lambda_qry=0.5,
    lambda_new=0.5,
    lambda_replay=0.3,
    lambda_kd=0.5,
    label_smoothing=0.1,
    kd=True,
    patience=5,      
):
    outer_opt = make_reptile_outer_optimizer(model)
    scheduler = CosineAnnealingLR(outer_opt, T_max=epochs)
    base_outer_lr = outer_opt.param_groups[0]['lr']
    best_val, patience_count, best_state = 0.0, 0, None

    for epoch in range(epochs):
        model.train()

        for _ in range(new_repeat):
            for x_new, y_new in train_loader:
                x_new, y_new = x_new.to(device), y_new.to(device)
                W_start = _snapshot(model)

                sup_x, sup_y, qry_x, qry_y = episode_buffer.sample_episode(
                    n_way=n_way, k_support=k_support, k_query=k_query, device=device
                )

                x_old, y_old = _sample_replay(replay_buffer, len(x_new), replay_ratio, device)

                current_outer_lr = outer_opt.param_groups[0]['lr']
                adapted_inner_lr = inner_lr * (current_outer_lr / base_outer_lr)
                adapted_epsilon  = epsilon  * (current_outer_lr / base_outer_lr)

                # Inner loop: support set ONLY (clean Reptile signal)
                inner_opt = torch.optim.SGD(model.parameters(), lr=adapted_inner_lr)

                for _ in range(inner_steps):
                    logits = model(sup_x)
                    loss = _safe_ce(logits, sup_y, num_classes, label_smoothing)

                    if kd and teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(sup_x)
                        loss += lambda_kd * distillation_loss(
                            logits[:, :num_old_classes],
                            t_logits[:, :num_old_classes]
                        )

                    inner_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    inner_opt.step()

                _reptile_update(model, W_start, adapted_epsilon)

                outer_opt.zero_grad(set_to_none=True)
                loss_meta_out = (
                    lambda_qry * _safe_ce(model(qry_x), qry_y, num_classes, label_smoothing)
                    + lambda_new * _safe_ce(model(x_new), y_new, num_classes, label_smoothing)
                )
                loss_meta_out.backward()
                _clip_step(outer_opt, model)

                if x_old is not None:
                    outer_opt.zero_grad(set_to_none=True)
                    loss_replay_out = lambda_replay * _safe_ce(
                        model(x_old), y_old, num_classes, label_smoothing
                    )
                    loss_replay_out.backward()
                    _clip_step(outer_opt, model)

        scheduler.step()
        _, val_acc = evaluate(model, val_loader, device)
        print(f"[Hybrid] Epoch {epoch+1}/{epochs} | ValAcc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[Hybrid] Early stop at epoch {epoch+1}, best ValAcc: {best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model