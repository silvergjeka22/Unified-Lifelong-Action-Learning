import torch
import torch.nn as nn
import torch.nn.functional as F


def fresh_model(num_classes, cfg, device, lstm_hidden):
    m = EmbeddingLSTM(hidden_size=lstm_hidden, num_classes=num_classes).to(device)

    ckpt      = torch.load(cfg.RESNET50_PATH, map_location=device)
    lstm_only = {k: v for k, v in ckpt.items() if k.startswith("lstm.")}
    m.load_state_dict(lstm_only, strict=False)

    return m


def make_teacher(source_model):
    t = copy.deepcopy(source_model)
    t.eval()
    for p in t.parameters():
        p.requires_grad = False
    return t


def make_optimizer(model):
    return torch.optim.Adam([
        {"params": model.lstm.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(),   "lr": 1e-3},
    ], weight_decay=1e-4)


def distillation_loss(student_logits, teacher_logits, T=5.0):
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction='batchmean') * (T * T)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y      = x.to(device), y.to(device)
        logits    = model(x)
        loss_sum += ce(logits, y).item()
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum / max(len(loader), 1), correct / total if total else 0.0


def weighted_ce(logits, labels, new_class_ids, new_weight=1.0, device="cpu"):
    weights = torch.ones(len(labels), device=device)
    for i, lbl in enumerate(labels):
        if lbl.item() in new_class_ids:
            weights[i] = new_weight
    return (weights * F.cross_entropy(logits, labels, reduction='none')).mean()

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING — Naive
# ══════════════════════════════════════════════════════════════════════════════

def train_naive(model, train_loader, val_loader, optimizer, device,
                epochs=5, new_repeat=10):
    ce = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for _ in range(new_repeat):
            for x, y in train_loader:
                x, y   = x.to(device), y.to(device)
                loss   = ce(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        _, val_acc = evaluate(model, val_loader, device)
        print(f"[Naive]    Epoch {epoch+1}/{epochs} | Loss:{total_loss:.4f} | ValAcc:{val_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING — EWC
# ══════════════════════════════════════════════════════════════════════════════

class EWC:
    def __init__(self, model, loader, device, num_samples=200):
        self.params = {
            n: p.clone().detach()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.fisher = self._compute_fisher(model, loader, device, num_samples)

    def _compute_fisher(self, model, loader, device, num_samples):
        fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters() if p.requires_grad
        }
        model.train()
        ce, count = nn.CrossEntropyLoss(), 0

        for x, y in loader:
            if count >= num_samples:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            ce(model(x), y).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += x.size(0)

        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher

    def penalty(self, model):
        return sum(
            (self.fisher[n] * (p - self.params[n]) ** 2).sum()
            for n, p in model.named_parameters() if n in self.fisher
        )


def train_ewc(model, train_loader, val_loader, optimizer, device,
              ewc_obj, lambda_ewc=5000, epochs=5, new_repeat=10):
    ce = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for _ in range(new_repeat):
            for x, y in train_loader:
                x, y   = x.to(device), y.to(device)
                loss   = ce(model(x), y) + lambda_ewc * ewc_obj.penalty(model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        _, val_acc = evaluate(model, val_loader, device)
        print(f"[EWC]      Epoch {epoch+1}/{epochs} | Loss:{total_loss:.4f} | ValAcc:{val_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING — Rehearsal
# ══════════════════════════════════════════════════════════════════════════════

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
                        t_logits[:, :num_old_classes]
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if not buffer_seeded:
                    replay_buffer.add_batch(x_new.detach().cpu(), y_new.detach().cpu())

            buffer_seeded = True

        _, val_acc = evaluate(model, val_loader, device)
        print(f"[Rehearsal {'KD' if kd else 'no-KD'}] "
              f"Epoch {epoch+1}/{epochs} | Loss:{total_loss:.4f} | ValAcc:{val_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# REPTILE CORE
# ══════════════════════════════════════════════════════════════════════════════

def _snapshot(model):
    """Captures current parameter values for the Reptile meta-update."""
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def _reptile_update(model, W_start, epsilon):
    """
    Reptile meta-update:
    W_meta = W_meta + epsilon * (W_adapted - W_meta)
    """
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
    best_val = 0.0
    use_weighted = (new_class_ids is not None and new_weight != 1.0)

    for epoch in range(epochs):
        model.train()

        for _ in range(episodes):
            W_start = _snapshot(model)

            sup_x, sup_y, qry_x, qry_y = episode_buffer.sample_episode(
                n_way, k_support, k_query, device,
                new_class_bias=new_class_bias
            )

            inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)

            for _ in range(inner_steps):
                logits = model(sup_x)

                if use_weighted:
                    loss = weighted_ce(logits, sup_y, new_class_ids, new_weight, device)
                else:
                    loss = F.cross_entropy(logits, sup_y)

                if kd and teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(sup_x)
                    loss += lambda_kd * distillation_loss(
                        logits[:, :num_old_classes],
                        t_logits[:, :num_old_classes]
                    )

                inner_opt.zero_grad()
                loss.backward()
                inner_opt.step()

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