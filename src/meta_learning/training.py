import torch
import torch.nn as nn
import torch.nn.functional as F


from src.meta_learning.models import evaluate, distillation_loss

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
    model, episode_buffer, train_loader,
    val_loader, device, teacher=None,
    num_old_classes=None, n_way=3,
    k_support=3, k_query=2, inner_lr=0.02,
    lstm_lr=0.0001, fc_lr=0.001, inner_steps=8,
    epsilon=0.1, new_repeat=5,
    epochs=5, lambda_kd=0.5, lambda_qry=0.5,
    kd=False, tag="Reptile"
):
    

    outer_opt = torch.optim.Adam([
        {"params": model.lstm.parameters(), "lr": lstm_lr},
        {"params": model.fc.parameters(), "lr": fc_lr},
    ], weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        for _ in range(new_repeat):
            for x_new, y_new in train_loader:
                x_new, y_new = x_new.to(device), y_new.to(device)
                
                # Snapshot for the Meta-Step
                W_start = _snapshot(model)
                sup_x, sup_y, qry_x, qry_y = episode_buffer.sample_episode(
                    n_way, k_support, k_query, device
                )
                
                # Inner Loop: Fast Adaptation
                inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
                for _ in range(inner_steps):
                    l_in = F.cross_entropy(model(sup_x), sup_y)
                    if kd and teacher is not None:
                        l_in += lambda_kd * distillation_loss(
                            model(sup_x)[:, :num_old_classes], 
                            teacher(sup_x)[:, :num_old_classes]
                        )
                    inner_opt.zero_grad(); l_in.backward(); inner_opt.step()

                # Meta-Update (The Secret Sauce)
                _reptile_update(model, W_start, epsilon)

                # Outer Update: Stability via Query Loss
                q_loss = F.cross_entropy(model(qry_x), qry_y)
                outer_opt.zero_grad(); (lambda_qry * q_loss).backward(); outer_opt.step()

                # New Data Update: Direct Supervised Learning
                n_loss = F.cross_entropy(model(x_new), y_new)
                outer_opt.zero_grad(); n_loss.backward(); outer_opt.step()

        _, val_acc = evaluate(model, val_loader, device)
        print(f"[{tag}] Epoch {epoch+1} | Val Acc: {val_acc:.4f}")
    return model
