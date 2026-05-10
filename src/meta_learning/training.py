import torch
import torch.nn as nn
import torch.nn.functional as F

from src.meta_learning.models import evaluate, distillation_loss, weighted_ce

# Rehearsal
def train_rehearsal(model, teacher, train_loader, val_loader,
                    replay_buffer, optimizer, device,
                    num_old_classes, lambda_distill=0.3,
                    epochs=5, new_repeat=10, kd=True):
    ce = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss, buffer_seeded = 0.0, False
        for _ in range(new_repeat):
            for x_new, y_new in train_loader:
                x_new, y_new = x_new.to(device), y_new.to(device)
                x_old, y_old = replay_buffer.sample(len(x_new))
                if x_old is not None:
                    x = torch.cat([x_new, x_old.to(device)])
                    y = torch.cat([y_new, y_old.to(device)])
                else:
                    x, y = x_new, y_new
                logits = model(x)
                loss   = ce(logits, y)
                if kd:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss = loss + lambda_distill * distillation_loss(
                        logits[:, :num_old_classes],
                        t_logits[:, :num_old_classes])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
                if not buffer_seeded:
                    replay_buffer.add_batch(x_new.detach().cpu(), y_new.detach().cpu())
            buffer_seeded = True
        _, val_acc = evaluate(model, val_loader, device)
        print(f"[Rehearsal {'KD' if kd else 'no-KD'}] "
              f"Epoch {epoch+1}/{epochs} | Loss:{total_loss:.4f} | ValAcc:{val_acc:.4f}")


# Reptile
def _snapshot(model):
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def _reptile_update(model, W_start, epsilon):
    sd = model.state_dict()
    with torch.no_grad():
        for key in W_start:
            sd[key] = W_start[key] + epsilon * (sd[key] - W_start[key])
    model.load_state_dict(sd)


def train_reptile_full(model, episode_buffer, val_loader, device,
                       teacher=None, num_old_classes=None,
                       n_way=5, k_support=3, k_query=3,
                       inner_lr=0.01, inner_steps=5, epsilon=0.1,
                       episodes=200, epochs=5,
                       lambda_kd=0.3, new_class_bias=3,
                       new_class_ids=None, new_weight=1.0,
                       kd=False, tag="Reptile", trial=None):
    best_val     = 0.0
    use_weighted = (new_class_ids is not None and new_weight != 1.0)

    for epoch in range(epochs):
        model.train()
        for _ in range(episodes):
            W_start = _snapshot(model)
            sup_x, sup_y, qry_x, qry_y = episode_buffer.sample_episode(
                n_way, k_support, k_query, device, new_class_bias=new_class_bias)
            inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
            for _ in range(inner_steps):
                logits = model(sup_x)
                loss   = (weighted_ce(logits, sup_y, new_class_ids, new_weight, device)
                          if use_weighted else F.cross_entropy(logits, sup_y))
                if kd and teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(sup_x)
                    loss += lambda_kd * distillation_loss(
                        logits[:, :num_old_classes],
                        t_logits[:, :num_old_classes])
                inner_opt.zero_grad(); loss.backward(); inner_opt.step()
            _reptile_update(model, W_start, epsilon)

        _, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        print(f"[{tag}] Epoch {epoch+1}/{epochs} | ValAcc:{val_acc:.4f} | Best:{best_val:.4f}")

    return model, best_val