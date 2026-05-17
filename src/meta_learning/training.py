import torch
import torch.nn as nn
import torch.nn.functional as F

from src.meta_learning.models import evaluate, distillation_loss, make_reptile_outer_optimizer


def _snapshot(model):
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def _reptile_update(model, W_start, epsilon):
    #new_w = W_start + epsilon * (W_adapted - W_start)
    sd = model.state_dict()
    with torch.no_grad():
        for key in W_start:
            sd[key] = W_start[key] + epsilon * (sd[key] - W_start[key])
    model.load_state_dict(sd)



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
    new_repeat=3,
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
        print(
            f"[Rehearsal {'KD' if kd else 'no-KD'}] "
            f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | ValAcc: {val_acc:.4f}"
        )

    return model


def train_reptile(
    model,
    episode_buffer,
    train_loader,
    val_loader,
    device,
    teacher=None,
    num_old_classes=None,
    n_way=3,
    k_support=5,
    k_query=15,
    inner_lr=0.01,
    lstm_lr=0.001,
    fc_lr=0.01,
    inner_steps=5,
    epsilon=0.1,
    new_repeat=3,
    epochs=5,
    lambda_kd=0.3,
    lambda_qry=0.7,
    kd=False,
    tag="Reptile",
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
                    l_in = F.cross_entropy(model(sup_x), sup_y)

                    if kd and teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(sup_x)
                        l_in = l_in + lambda_kd * distillation_loss(
                            model(sup_x)[:, :num_old_classes],
                            t_logits[:, :num_old_classes]
                        )

                    inner_opt.zero_grad()
                    l_in.backward()
                    inner_opt.step()

                _reptile_update(model, W_start, epsilon)

                q_loss = lambda_qry * F.cross_entropy(model(qry_x), qry_y)
                outer_opt.zero_grad()
                q_loss.backward()
                outer_opt.step()

                n_loss = F.cross_entropy(model(x_new), y_new)
                outer_opt.zero_grad()
                n_loss.backward()
                outer_opt.step()

        _, val_acc = evaluate(model, val_loader, device)
        print(f"[{tag}] Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")

    return model



def train_rehearsal_reptile(
    model,
    episode_buffer,
    replay_buffer,
    train_loader,
    val_loader,
    device,
    teacher=None,
    num_old_classes=None,
    n_way=3,
    k_support=5,
    k_query=15,
    inner_lr=0.01,
    lstm_lr=0.001,
    fc_lr=0.01,
    inner_steps=5,
    epsilon=0.1,
    new_repeat=3,
    epochs=5,
    lambda_kd=0.3,
    lambda_qry=0.7,
    kd=True,
    tag="Rehearsal+Reptile",
):
    outer_opt = make_reptile_outer_optimizer(model, lstm_lr=lstm_lr, fc_lr=fc_lr)
    ce = nn.CrossEntropyLoss()

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
                    # No KD in inner loop — let it adapt freely to new classes
                    l_in = F.cross_entropy(model(sup_x), sup_y)
                    inner_opt.zero_grad()
                    l_in.backward()
                    inner_opt.step()

                _reptile_update(model, W_start, epsilon)

                q_loss = lambda_qry * F.cross_entropy(model(qry_x), qry_y)
                outer_opt.zero_grad()
                q_loss.backward()
                outer_opt.step()

                x_old, y_old = replay_buffer.sample(len(x_new))
                if x_old is not None:
                    x_all = torch.cat([x_new, x_old.to(device)], dim=0)
                    y_all = torch.cat([y_new, y_old.to(device)], dim=0)
                else:
                    x_all, y_all = x_new, y_new

                logits_all = model(x_all)
                loss_stream = ce(logits_all, y_all)

                if kd and teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x_all)
                    loss_stream = loss_stream + lambda_kd * distillation_loss(
                        logits_all[:, :num_old_classes],
                        t_logits[:, :num_old_classes]
                    )

                outer_opt.zero_grad()
                loss_stream.backward()
                outer_opt.step()

        _, val_acc = evaluate(model, val_loader, device)
        print(f"[{tag}] Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")

    return model