import torch
import torch.nn as nn
import torch.nn.functional as F

from src.meta_learning.models import evaluate, distillation_loss, make_reptile_outer_optimizer


def train_rehearsal(
    model,
    teacher,
    train_loader,
    val_loader,
    replay_buffer,
    optimizer,
    device,
    num_old_classes,
    lambda_distill=0.3,
    epochs=5,
    new_repeat=2,
    kd=True,
):
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for _ in range(new_repeat):
            for x_new, y_new in train_loader:
                x_new, y_new = x_new.to(device), y_new.to(device)

                x_old, y_old = replay_buffer.sample(len(x_new))
                if x_old is not None:
                    x = torch.cat([x_new, x_old.to(device)], dim=0)
                    y = torch.cat([y_new, y_old.to(device)], dim=0)
                else:
                    x, y = x_new, y_new

                logits = model(x)
                loss = ce(logits, y)

                if kd and teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss = loss + lambda_distill * distillation_loss(
                        logits[:, :num_old_classes],
                        t_logits[:, :num_old_classes]
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        _, val_acc = evaluate(model, val_loader, device)
        print(f"[Rehearsal {'KD' if kd else 'no-KD'}] Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss:.4f} | ValAcc: {val_acc:.4f}")

    return model


def _snapshot(model):
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def _reptile_update(model, W_start, epsilon):
    sd = model.state_dict()
    with torch.no_grad():
        for key in W_start:
            sd[key] = W_start[key] + epsilon * (sd[key] - W_start[key])
    model.load_state_dict(sd)


def train_reptile(
    model,
    episode_buffer,
    train_loader,
    val_loader,
    device,
    teacher=None,
    num_old_classes=None,
    n_way=3,
    k_support=2,
    k_query=1,
    inner_lr=0.01,
    lstm_lr=1e-4,
    fc_lr=1e-3,
    inner_steps=3,
    epsilon=0.05,
    new_repeat=2,
    epochs=5,
    lambda_kd=0.1,
    lambda_qry=0.5,
    kd=False,
    tag="Reptile"
):
    outer_opt = make_reptile_outer_optimizer(model, lstm_lr=lstm_lr, fc_lr=fc_lr)

    for epoch in range(epochs):
        model.train()

        for _ in range(new_repeat):
            for x_new, y_new in train_loader:
                x_new, y_new = x_new.to(device), y_new.to(device)

                W_start = _snapshot(model)

                sup_x, sup_y, qry_x, qry_y = episode_buffer.sample_episode(
                    n_way=n_way,
                    k_support=k_support,
                    k_query=k_query,
                    device=device
                )

                inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)

                for _ in range(inner_steps):
                    logits_sup = model(sup_x)
                    l_in = F.cross_entropy(logits_sup, sup_y)

                    if kd and teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(sup_x)
                        l_in = l_in + lambda_kd * distillation_loss(
                            logits_sup[:, :num_old_classes],
                            t_logits[:, :num_old_classes]
                        )

                    inner_opt.zero_grad()
                    l_in.backward()
                    inner_opt.step()

                _reptile_update(model, W_start, epsilon)

                q_loss = F.cross_entropy(model(qry_x), qry_y)
                outer_opt.zero_grad()
                (lambda_qry * q_loss).backward()
                outer_opt.step()

                n_loss = F.cross_entropy(model(x_new), y_new)
                outer_opt.zero_grad()
                n_loss.backward()
                outer_opt.step()

        _, val_acc = evaluate(model, val_loader, device)
        print(f"[{tag}] Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")

    return model