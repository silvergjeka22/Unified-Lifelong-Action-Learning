"""
Unified continual-learning training loop.

ONE function trains every arm — naive, replay, EWC, and any combination:

    buffer only   -> replay
    fisher only   -> EWC
    both          -> replay + EWC
    neither       -> naive

Why one loop matters
--------------------
An earlier version ran naive/replay through train_with_smart_replay (which selects the
best epoch by validation accuracy) and EWC through train_ewc (which does not). Two
consequences, both invalidating:

  1. Best-val selection stopped naive at epoch ~2 — before it had forgotten anything OR
     learned the new task (Base=97%, Task1=43%) — while EWC trained all 15 epochs. The
     arms were measuring different things.

  2. Selecting on a validation set that spans OLD classes is itself a form of memory. A
     "0 bytes" naive baseline that quietly consults base-class validation data is not a
     0-byte baseline.

So: fixed epochs, no selection, report the final model. Per-epoch validation is printed
for monitoring only and never touches the weights.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.cl.rehearsal import distillation_loss
from src.kd.utils import eval_head


# ── EWC key mapping ───────────────────────────────────────────────────────────
def fisher_key(fisher_tid: dict, name: str):
    """
    Map a raw head parameter name onto its Fisher entry.

    compute_fisher() runs on a ModelWrapper (it needs bare logits, but heads return
    (logits, proj)), so every stored key carries an 'original_model.' prefix that
    head.named_parameters() does not. Without this the penalty silently matches nothing
    and EWC degenerates to naive.
    """
    if name in fisher_tid:
        return name
    prefixed = "original_model." + name
    return prefixed if prefixed in fisher_tid else None


def ewc_penalty(head, fisher: dict, optpar: dict, device):
    """sum_tasks sum_params  F * (theta - theta*)^2"""
    pen = torch.zeros((), device=device)
    for tid in fisher:
        for name, prm in head.named_parameters():
            k = fisher_key(fisher[tid], name)
            if k is None:
                continue
            f_ = fisher[tid][k].to(device)
            o_ = optpar[tid][k].to(device)
            c  = (f_ * (prm - o_).pow(2)).sum()
            if not torch.isnan(c):
                pen = pen + c
    return pen


# ── Evaluation ────────────────────────────────────────────────────────────────
def eval_tasks_upto(head, cache: dict, t: int, device, split: str = "test"):
    """{task_idx: accuracy} for every task seen so far."""
    return {
        i: eval_head(head, cache[f"t{i}_{split}"][0], cache[f"t{i}_{split}"][1], device)
        for i in range(t + 1) if f"t{i}_{split}" in cache
    }


def format_row(row: dict, task_names: list):
    return "   ".join(f"{task_names[i]}={a:.2%}" for i, a in sorted(row.items()))


def emb_loader(s, y, batch_size=32, shuffle=False):
    """DataLoader over cached features — used for Fisher estimation."""
    return DataLoader(TensorDataset(s, y), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def eval_head_ce(head, s, y, device, batch_size=128):
    """(accuracy, cross-entropy) of a head on cached features."""
    head.eval()
    correct = total = 0
    loss_sum = 0.0
    for sb, yb in DataLoader(TensorDataset(s, y), batch_size=batch_size):
        sb, yb = sb.to(device), yb.to(device)
        logits, _ = head(sb, training=False)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        correct  += (logits.argmax(1) == yb).sum().item()
        total    += yb.size(0)
    return correct / max(total, 1), loss_sum / max(total, 1)


# ── The loop ──────────────────────────────────────────────────────────────────
def snapshot_teacher(head):
    """
    Frozen copy of the head as it stands *before* the current task — the LwF teacher.

    Must be re-taken at the start of EVERY task: the teacher for task t is the model
    after task t-1, not the original base model. Reusing a stale teacher quietly makes
    the distillation term useless.
    """
    teacher = copy.deepcopy(head)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


def train_cl_arm(head, new_s, new_y, device, epochs, lr, wd, batch_size,
                 label_smoothing=0.1, tag="arm", buffer=None,
                 fisher=None, optpar=None, ewc_lambda=0.0,
                 teacher=None, kd_lambda=0.0, kd_T=5.0, num_old_classes=None,
                 cache=None, monitor_t=None, task_names=None, log_every=5,
                 val_s=None, val_y=None, history=None):
    """
    Train one CL arm on the current task's cached features.

    Mechanisms are opt-in and compose freely:

        buffer            -> replay          (hard labels on stored exemplars)
        fisher            -> EWC             (anchor important weights)
        teacher           -> LwF             (soft targets on EVERY sample)
        buffer + teacher  -> iCaRL-style     (both)
        none              -> naive

    Why LwF is not redundant with replay
    ------------------------------------
    Replay supplies HARD labels for the handful of exemplars you stored. LwF supplies
    SOFT targets for every sample in the batch — including the new task's data, which
    you have a lot of. Those soft targets encode inter-class similarity ("this
    BrushingTeeth clip is somewhat ApplyLipstick-ish"), which is exactly the decision
    boundary that hard labels cannot express.

    Only the OLD-class logits are distilled: the teacher's head has no rows for the
    classes introduced by this task.

    Args:
        teacher         : frozen pre-task snapshot (see snapshot_teacher)
        kd_lambda, kd_T : distillation weight and temperature
        num_old_classes : head width before this task's expansion
        val_s, val_y    : optional held-out features, evaluated once per epoch
        history         : optional dict filled with per-epoch losses and accuracies,
                          in the shape plot_training_results expects

    Returns the head after a fixed number of epochs. No best-epoch selection.
    """
    head.to(device)
    params = [p for p in head.parameters() if p.requires_grad]
    opt    = optim.AdamW(params, lr=lr, weight_decay=wd)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)

    use_ewc = bool(fisher) and ewc_lambda > 0
    use_kd  = teacher is not None and kd_lambda > 0
    if use_kd:
        if not num_old_classes:
            raise ValueError("LwF needs num_old_classes (head width before expansion).")
        teacher.to(device).eval()

    for ep in range(1, epochs + 1):
        # Resample the replay mix every epoch so the buffer is seen from a fresh angle.
        if buffer is not None:
            b_s, _, b_y = buffer.sample_batch(len(new_s))
            s_ep = torch.cat([new_s, b_s]) if b_s is not None else new_s
            y_ep = torch.cat([new_y, b_y]) if b_y is not None else new_y
        else:
            s_ep, y_ep = new_s, new_y

        loader = DataLoader(TensorDataset(s_ep, y_ep), batch_size=batch_size, shuffle=True)
        head.train()
        tot_ce = tot_pen = tot_kd = 0.0
        correct = total = 0

        for sb, yb in loader:
            sb, yb = sb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = head(sb, training=True)
            ce   = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
            loss = ce

            if use_ewc:
                pen  = ewc_penalty(head, fisher, optpar, device)
                loss = loss + ewc_lambda * pen
                tot_pen += float(pen.detach())

            if use_kd:
                with torch.no_grad():
                    t_out = teacher(sb, training=False)
                    t_logits = t_out[0] if isinstance(t_out, (tuple, list)) else t_out
                n_old = min(num_old_classes, t_logits.shape[1], logits.shape[1])
                kd = distillation_loss(logits[:, :n_old], t_logits[:, :n_old], T=kd_T)
                loss = loss + kd_lambda * kd
                tot_kd += float(kd.detach())

            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot_ce  += ce.item()
            correct += (logits.argmax(1) == yb).sum().item()
            total   += yb.size(0)

        sched.step()

        if history is not None:
            nb = max(len(loader), 1)
            train_acc = correct / max(total, 1)
            val_acc, val_loss = eval_head_ce(head, val_s, val_y, device)
            history["train_losses"].append(tot_ce / nb)
            history["val_losses"].append(val_loss)
            history["train_accs"].append(train_acc)
            history["val_accs"].append(val_acc)
            history["best_val_acc"] = max(history["best_val_acc"], val_acc)
            print(f"Epoch [{ep}/{epochs}] | Train Acc: {train_acc:.4f} "
                  f"Loss: {tot_ce/nb:.4f} | Val Acc: {val_acc:.4f} Loss: {val_loss:.4f}")
            head.train()

        if ep % log_every == 0 or ep == 1 or ep == epochs:
            nb  = max(len(loader), 1)
            msg = f"  [{tag}] ep {ep:02d}/{epochs} | CE {tot_ce/nb:.4f}"
            if use_ewc:
                # Aim for EWC at 5-20% of CE. Much below that and the penalty is inert.
                msg += f" | EWC {ewc_lambda*tot_pen/nb:.4f}"
            if use_kd:
                msg += f" | KD {kd_lambda*tot_kd/nb:.4f}"
            if cache is not None and monitor_t is not None:
                r = eval_tasks_upto(head, cache, monitor_t, device, "val")
                names = task_names or [f"T{i}" for i in range(monitor_t + 1)]
                msg += " | val " + " ".join(f"{names[i][:2]}{i}={a:.0%}" for i, a in sorted(r.items()))
            print(msg)

    head.eval()
    return head


# ── The shared starting point ─────────────────────────────────────────────────
def train_base_head(cache, lstm_state, num_classes, device,
                    epochs=15, lr=5e-4, wd=0.03, batch_size=32, label_smoothing=0.1):
    """Train a fresh TemporalHead on task 0's cached features. Returns the head."""
    from src.models.temporal_head import make_temporal_head

    head = make_temporal_head(num_classes, lstm_state=lstm_state, device=device)
    train_s, train_y = cache["t0_train"]
    val_s,   val_y   = cache["t0_val"]
    head, _ = train_task(head, train_s, train_y, val_s, val_y, device,
                         epochs=epochs, lr=lr, wd=wd, batch_size=batch_size,
                         label_smoothing=label_smoothing, tag="base")
    return head


def load_or_train_base_head(path, cache, lstm_state, num_classes, device,
                            epochs=15, lr=5e-4, wd=0.03, batch_size=32,
                            label_smoothing=0.1):
    """
    The base head from `path`, training and saving it first if it is not there yet.

    Every arm must start from IDENTICAL weights or the comparison is meaningless, so
    the first notebook to run writes the checkpoint and the rest load it. Delete the
    file after changing the class configuration.
    """
    import os

    from src.models.temporal_head import make_temporal_head

    if os.path.exists(path):
        head = make_temporal_head(num_classes, lstm_state=lstm_state, device=device)
        head.load_state_dict(torch.load(path, map_location=device))
        head.eval()
        print(f"Loaded base head <- {path}")
        return head

    print(f"No base head at {path} — training it on task 0 now.")
    head = train_base_head(cache, lstm_state, num_classes, device, epochs, lr, wd,
                           batch_size, label_smoothing)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(head.state_dict(), path)
    head.eval()
    print(f"Saved base head -> {path}")
    return head


# ── Buffer helper ─────────────────────────────────────────────────────────────
def fill_buffer(buffer, feats, labels, class_names, global_c2i, strategy="hard"):
    """Add each class's exemplars to the replay buffer."""
    for c in class_names:
        idx  = global_c2i[c]
        mask = labels == idx
        if mask.sum() > 0:
            buffer.add_class(idx, feats[mask], strategy=strategy)
    return buffer


def buffer_bytes_per_class(buffer, per_class: int):
    """Bytes one class costs in the buffer — the x-axis of the Section 5 curve."""
    s, _, _ = buffer.get_all()
    if s is None:
        return 0
    return s.element_size() * s[0].nelement() * per_class


# ── Full task stream ──────────────────────────────────────────────────────────
def run_cl_stream(base_head, cache, cl_tasks, task_classes, task_names, num_after,
                  global_c2i, base_classes, device,
                  epochs, lr, wd, batch_size, label_smoothing,
                  arm="naive", replay_per_class=0, replay_strategy="hard",
                  use_ewc=False, ewc_lambda=0.0,
                  use_kd=False, kd_lambda=2.0, kd_T=5.0,
                  use_weight_align=False, log_every=5, verbose=True):
    """
    Run one arm across the whole task stream and return everything the compare
    notebook needs.

    Every arm goes through this ONE function, so lr/epochs/schedule/protocol are
    identical by construction — the mechanism flags are the only difference. That is
    what makes the per-method notebooks comparable even though they run separately.

    Flags:
        replay_per_class > 0 -> SmartReplay
        use_ewc              -> EWC
        use_kd               -> LwF
        use_weight_align     -> rescale new-class logit norms after each task
        (combinations are allowed: replay + kd = iCaRL-style)

    Returns dict: rows, head, bytes_per_class, buffer, arm
    """
    import copy as _copy

    from src.cl.ewc          import compute_fisher, pad_fisher_after_expand
    from src.cl.smart_replay import SmartReplayBuffer
    from src.models.temporal_head import expand_head, weight_align
    from src.utils.train     import ModelWrapper

    head = _copy.deepcopy(base_head).to(device)
    rows = [eval_tasks_upto(head, cache, 0, device)]

    base_s, base_y = cache["t0_train"]

    buffer = None
    if replay_per_class > 0:
        buffer = SmartReplayBuffer(max_per_class=replay_per_class)
        fill_buffer(buffer, base_s, base_y, base_classes, global_c2i, replay_strategy)
        if verbose:
            buffer.summary()

    fisher, optpar, wrapped = {}, {}, None
    if use_ewc:
        wrapped = ModelWrapper(head).to(device)
        if verbose:
            print("\n--- Fisher on base classes")
        compute_fisher(wrapped, emb_loader(base_s, base_y, batch_size), device, 0, fisher, optpar)

    for t in cl_tasks:
        if verbose:
            print(f"\n--- {task_names[t]}: +{len(task_classes[t])} -> head {num_after[t]}")
        new_s, new_y = cache[f"t{t}_train"]
        n_old = num_after[t - 1]

        # LwF teacher = the model as it stands BEFORE this task
        teacher = snapshot_teacher(head) if use_kd else None

        head = expand_head(head, num_after[t])
        if use_ewc:
            fisher, optpar = pad_fisher_after_expand(fisher, optpar, wrapped)

        head = train_cl_arm(
            head, new_s, new_y, device=device, epochs=epochs, lr=lr, wd=wd,
            batch_size=batch_size, label_smoothing=label_smoothing, tag=arm,
            buffer=buffer,
            fisher=fisher if use_ewc else None, optpar=optpar if use_ewc else None,
            ewc_lambda=ewc_lambda if use_ewc else 0.0,
            teacher=teacher, kd_lambda=kd_lambda if use_kd else 0.0,
            kd_T=kd_T, num_old_classes=n_old,
            cache=cache, monitor_t=t, task_names=task_names, log_every=log_every,
        )

        if use_weight_align:
            before = eval_tasks_upto(head, cache, t, device)[0]
            head = weight_align(head, n_old=n_old)
            if verbose:
                after = eval_tasks_upto(head, cache, t, device)[0]
                print(f"  weight_align: base {before:.2%} -> {after:.2%}")

        if buffer is not None:      # classes just learned become replayable
            fill_buffer(buffer, new_s, new_y, task_classes[t], global_c2i, replay_strategy)

        if use_ewc and t != cl_tasks[-1]:
            compute_fisher(wrapped, emb_loader(new_s, new_y, batch_size), device, t, fisher, optpar)

        row = eval_tasks_upto(head, cache, t, device)
        rows.append(row)
        if verbose:
            print("  -> " + format_row(row, task_names))

    bpc = buffer_bytes_per_class(buffer, replay_per_class) if buffer is not None else 0
    return {"rows": rows, "head": head, "bytes_per_class": int(bpc),
            "buffer": buffer, "arm": arm}


def save_arm_result(path, arm, rows, task_names, bytes_per_class, config, ceiling=None):
    """Write one arm's result so the compare notebook can collect it later."""
    import json

    from src.utils.metrics import (
        average_accuracy, backward_transfer, build_accuracy_matrix, forgetting_measure,
    )

    R = build_accuracy_matrix(rows)
    metrics = {
        "base_acc_final": float(R[-1][0]),
        "AA":             average_accuracy(R),
        "BWT":            backward_transfer(R),
        "forgetting":     forgetting_measure(R),
    }
    payload = {"arm": arm, "task_names": task_names, "matrix": R.tolist(),
               "metrics": metrics, "bytes_per_class": int(bytes_per_class),
               "ceiling": ceiling, "config": config}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n{arm}: AA={metrics['AA']:.2%}  BWT={metrics['BWT']:+.2%}  "
          f"base={metrics['base_acc_final']:.2%}  mem={bytes_per_class/1e3:.0f} KB/class")
    print(f"Saved -> {path}")
    return R, metrics


# ── Notebook-facing wrapper ───────────────────────────────────────────────────
def train_task(head, train_s, train_y, val_s, val_y, device,
               epochs=15, lr=5e-4, wd=0.03, batch_size=32, label_smoothing=0.1,
               buffer=None, fisher=None, optpar=None, ewc_lambda=0.0,
               teacher=None, kd_lambda=0.0, kd_T=5.0, num_old_classes=None,
               tag="task"):
    """
    Train one task and return (head, history).

    Same role and return shape as train_model in src/utils/train.py, so notebook
    cells look the same and plot_training_results works unchanged.

    All epochs run inside ONE train_cl_arm call. Calling it once per epoch instead
    rebuilds AdamW every epoch (throwing away the momentum estimates) and restarts
    the cosine schedule, so the learning rate never actually decays.
    """
    hist = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": [],
            "best_val_acc": 0.0}

    head = train_cl_arm(head, train_s, train_y, device=device, epochs=epochs, lr=lr,
                        wd=wd, batch_size=batch_size,
                        label_smoothing=label_smoothing, tag=tag, buffer=buffer,
                        fisher=fisher, optpar=optpar, ewc_lambda=ewc_lambda,
                        teacher=teacher, kd_lambda=kd_lambda, kd_T=kd_T,
                        num_old_classes=num_old_classes, log_every=10**9,
                        val_s=val_s, val_y=val_y, history=hist)

    return head, hist
