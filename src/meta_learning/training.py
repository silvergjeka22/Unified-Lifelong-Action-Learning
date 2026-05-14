import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna


from src.meta_learning.models import evaluate, distillation_loss, weighted_ce


# Rehearsal Training 
def train_rehearsal(
    model,
    teacher,            # frozen teacher model
    train_loader,       # new task training data
    val_loader,         # combined val
    replay_buffer,      # ReplayBuffer holding old task exemplars
    optimizer,
    device,
    num_old_classes,    # number of old-class logits used for distillation
    lambda_distill=0.3, # weight of the KD loss term
    epochs=5,
    new_repeat=8,      # times new task data is iterated per epoch
    kd=True,            # kd loss
    trial=None,         # optuna trial object
):
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss    = 0.0
        buffer_seeded = False  # seed the replay buffer only on the first pass

        for _ in range(new_repeat):
            for x_new, y_new in train_loader:
                x_new, y_new = x_new.to(device), y_new.to(device)

                # mix new samples with a replay batch of equal size
                x_old, y_old = replay_buffer.sample(len(x_new))
                if x_old is not None:
                    x = torch.cat([x_new, x_old.to(device)])
                    y = torch.cat([y_new, y_old.to(device)])
                else:
                    x, y = x_new, y_new  # no replay yet

                logits = model(x)
                loss   = ce(logits, y)

                # KD loss on old-class logits to prevent forgetting
                if kd:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss = loss + lambda_distill * distillation_loss(
                        logits[:, :num_old_classes],
                        t_logits[:, :num_old_classes],
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # new samples to the replay buffer
                if not buffer_seeded:
                    replay_buffer.add_batch(x_new.detach().cpu(), y_new.detach().cpu())

            buffer_seeded = True  # avoid adding new samples multiple times per epoch

        _, val_acc = evaluate(model, val_loader, device)

        # optuna for pruning during hyper-parameter search
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        print(
            f"[Rehearsal {'KD' if kd else 'no-KD'}] "
            f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | ValAcc: {val_acc:.4f}"
        )


# Reptile Training
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
    episode_buffer,         # EpisodeBuffer
    train_loader,           # NEW task train loader
    val_loader,
    device,
    teacher=None,
    num_old_classes=None,
    n_way=5,
    k_support=3,
    k_query=2,
    inner_lr=0.01,
    lstm_lr=1e-4,
    fc_lr=1e-3,
    inner_steps=5,
    epsilon=0.1,
    new_repeat=10,
    epochs=5,
    lambda_kd=0.3,
    lambda_qry=0.5,
    new_class_ids=None,
    new_weight=1.0,
    kd=False,
    tag="Reptile",
    trial=None,
):
    best_val     = 0.0
    use_weighted = (new_class_ids is not None and new_weight != 1.0)

    outer_opt = torch.optim.Adam(
        [
            {"params": model.lstm.parameters(), "lr": lstm_lr},
            {"params": model.fc.parameters(),   "lr": fc_lr},
        ],
        weight_decay=1e-4,
    )

    for epoch in range(epochs):
        model.train()
        qry_loss = torch.tensor(0.0, device=device)

        for _ in range(new_repeat):
            for x_new, y_new in train_loader:
                x_new, y_new = x_new.to(device), y_new.to(device)

                # OLD-CLASS branch: Reptile retention
                W_start = _snapshot(model)

                sup_x_old, sup_y_old, qry_x_old, qry_y_old = episode_buffer.sample_episode(
                    n_way, k_support, k_query, device
                )

                inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
                for _ in range(inner_steps):
                    logits = model(sup_x_old)
                    loss = (
                        weighted_ce(logits, sup_y_old, new_class_ids, new_weight, device)
                        if use_weighted
                        else F.cross_entropy(logits, sup_y_old)
                    )

                    if kd and teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(sup_x_old)
                        loss += lambda_kd * distillation_loss(
                            logits[:, :num_old_classes],
                            t_logits[:, :num_old_classes],
                        )

                    inner_opt.zero_grad()
                    loss.backward()
                    inner_opt.step()

                _reptile_update(model, W_start, epsilon)

                qry_logits = model(qry_x_old)
                qry_loss = (
                    weighted_ce(qry_logits, qry_y_old, new_class_ids, new_weight, device)
                    if use_weighted
                    else F.cross_entropy(qry_logits, qry_y_old)
                )
                outer_opt.zero_grad()
                (lambda_qry * qry_loss).backward()
                outer_opt.step()

                # NEW-CLASS branch: standard supervised SGD
                new_logits = model(x_new)
                new_loss = (
                    weighted_ce(new_logits, y_new, new_class_ids, new_weight, device)
                    if use_weighted
                    else F.cross_entropy(new_logits, y_new)
                )
                outer_opt.zero_grad()
                new_loss.backward()
                outer_opt.step()

        outer_opt.step()
        outer_opt.zero_grad()

        _, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc

        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        print(
            f"[{tag}] Epoch {epoch+1}/{epochs} | "
            f"ValAcc: {val_acc:.4f} | QryLoss: {qry_loss.item():.4f}"
        )

    print(f"  [{tag}] Best val acc : {best_val:.4f}")
    return model, best_val