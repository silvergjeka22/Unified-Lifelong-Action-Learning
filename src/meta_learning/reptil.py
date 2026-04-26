import os
import copy
import random
from collections import defaultdict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.fine_tune.trainer import evaluate_model


class ReplayBufferBalanced:
    """
    Class-balanced episodic memory.

    Each class is capped at `max_per_class` samples independently.
    This prevents new tasks from evicting old-class samples (which happens
    in your current FIFO ReplayBuffer when total size is exceeded).

    API is intentionally compatible with your existing ReplayBuffer:
        buffer.add_batch(x, y)   ← same signature
        buffer.sample(n)         ← same signature, same return type
        len(buffer.data)         ← still works via .data property

    Parameters
    ----------
    max_per_class : int
        Maximum samples stored per class.  10 per class × 10 base classes
        = 100 base samples, same memory cost as your current limit=5 ×
        batch approach but class-balanced.
    """

    def __init__(self, max_per_class: int = 10):
        self.max_per_class = max_per_class
        self._store: dict = defaultdict(list)   # {class_idx: [(x_cpu, y_cpu)]}

   
    @property
    def data(self):
        return [s for samples in self._store.values() for s in samples]

    def add_batch(self, x, y):
        """
        Same signature as your existing ReplayBuffer.add_batch(x, y).
        Stores up to max_per_class samples per class on CPU.
        """
        for xi, yi in zip(x, y):
            c = yi.item() if hasattr(yi, "item") else int(yi)
            if len(self._store[c]) < self.max_per_class:
                self._store[c].append((xi.cpu(), yi.cpu()))

    def sample(self, batch_size: int):
        """
        Same signature as your existing ReplayBuffer.sample(batch_size).
        Returns (x_batch, y_batch) on CPU, or (None, None) if empty.
        """
        all_samples = self.data
        if not all_samples:
            return None, None
        chosen = random.sample(all_samples, min(batch_size, len(all_samples)))
        xs, ys = zip(*chosen)
        return torch.stack(xs), torch.stack([y if isinstance(y, torch.Tensor)
                                              else torch.tensor(y) for y in ys])

    def summary(self) -> dict:
        return {c: len(v) for c, v in sorted(self._store.items())}

def _inner_loss_rehearsal(
    output,
    target,
    adapted_model,
    replay_buffer,
    device,
    teacher=None,
    num_old_classes: int = None,
    lambda_distill: float = 1.0,
    T: float = 2.0,
    kd: bool = False,
):
    """
    L = CE(new_task_batch) + CE(replay_batch) [+ λ·KL if kd=True]
    """
    loss = F.cross_entropy(output, target)

    x_old, y_old = replay_buffer.sample(len(target))
    if x_old is None:
        return loss

    x_old      = x_old.to(device)
    y_old      = y_old.to(device)
    old_output = adapted_model(x_old)
    loss       = loss + F.cross_entropy(old_output, y_old)

    if kd and teacher is not None and num_old_classes is not None:
        with torch.no_grad():
            t_logits = teacher(x_old)
        s_old   = F.log_softmax(old_output[:, :num_old_classes] / T, dim=1)
        t_old   = F.softmax(t_logits[:, :num_old_classes]  / T, dim=1)
        loss_kd = F.kl_div(s_old, t_old, reduction="batchmean") * (T * T)
        loss    = loss + lambda_distill * loss_kd

    return loss


def maml_rehearsal_task(
    model,
    train_loader,               # renamed from support_loader to match your notebook
    val_loader,                 # renamed from query_loader to match your notebook
    replay_buffer,
    device,
    inner_steps: int   = 5,
    inner_lr: float    = 5e-3,
    meta_lr: float     = 0.3,   # Reptile ε
    meta_epochs: int   = 5,
    grad_clip: float   = 1.0,
    teacher            = None,
    num_old_classes: int   = None,
    lambda_distill: float  = 1.0,
    T: float               = 2.0,
    kd: bool               = False,
    task_name: str     = "MAML_Rehearsal",
    save_dir: str      = None,
):
    """
    Reptile continual learner with Rehearsal inner loop.

    Signature mirrors train_continual() so you can swap it in directly.

    One outer epoch:
      1. deep-copy θ  →  θ'
      2. run inner_steps passes of SGD on θ':  CE(new) + CE(replay) [+KD]
      3. Reptile update:  θ ← θ + ε·(θ'−θ)    ← only this touches θ
    """
    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    best_val_acc    = 0.0
    best_model_path = None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, f"best_{task_name}.pt")

    print(
        f"\n[Reptile+Rehearsal] '{task_name}' | "
        f"inner_steps={inner_steps} | inner_lr={inner_lr} | "
        f"ε={meta_lr} | epochs={meta_epochs} | "
        f"replay={len(replay_buffer)} samples | kd={kd}"
    )

    for epoch in range(meta_epochs):
        model.train()

        # fresh adapted copy
        adapted   = copy.deepcopy(model).to(device)
        inner_opt = optim.SGD(
            adapted.parameters(), lr=inner_lr, momentum=0.9, weight_decay=1e-4
        )

        total_loss, total_samples = 0.0, 0

        # inner loop: adapt copy on new task + replay
        for step in range(inner_steps):
            for x, y in tqdm(
                train_loader,
                desc=f"[{task_name}] Epoch {epoch+1}/{meta_epochs} "
                     f"inner step {step+1}/{inner_steps}",
                leave=False,
            ):
                x, y = x.to(device), y.to(device)
                inner_opt.zero_grad()

                output = adapted(x)
                loss   = _inner_loss_rehearsal(
                    output          = output,
                    target          = y,
                    adapted_model   = adapted,
                    replay_buffer   = replay_buffer,
                    device          = device,
                    teacher         = teacher,
                    num_old_classes = num_old_classes,
                    lambda_distill  = lambda_distill,
                    T               = T,
                    kd              = kd,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapted.parameters(), grad_clip)
                inner_opt.step()

                total_loss    += loss.item() * x.size(0)
                total_samples += x.size(0)

        avg_inner_loss = total_loss / max(total_samples, 1)

        # Reptile outer update: θ ← θ + ε·(θ'−θ)
        with torch.no_grad():
            for p_meta, p_adapt in zip(model.parameters(), adapted.parameters()):
                if p_meta.requires_grad:
                    p_meta.data.add_(meta_lr * (p_adapt.data - p_meta.data))

        # evaluate meta-weights 
        train_acc, train_loss = evaluate_model(model, train_loader, device)
        val_acc,   val_loss   = evaluate_model(model, val_loader,   device)

        train_accs.append(train_acc);    val_accs.append(val_acc)
        train_losses.append(train_loss); val_losses.append(val_loss)

        print(
            f"  Epoch {epoch+1}/{meta_epochs} | "
            f"Train={train_acc:.4f}  Val={val_acc:.4f}  "
            f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  "
            f"InnerLoss={avg_inner_loss:.4f}"
        )

        if save_dir and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    → Saved best (Val={best_val_acc:.4f})")

    return train_accs, val_accs, train_losses, val_losses