"""
Few-shot class-incremental learning (FSCIL) stream with replay + LwF.

Novel classes arrive a task at a time, each with only a few clips. A replay buffer plus an LwF
teacher keep the old classes, and the backbone is frozen so few-shot fitting moves only the LSTM +
head (old-class features are preserved). run_fscil is called twice - once from the pretrained init
(rehearsal only) and once from a MAML-meta-trained init (MAML + rehearsal) - to show what MAML adds.
"""

import gc
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

import src.config.config as cfg
from src.cl.rehearsal import ReplayBuffer, fill_buffer, snapshot_teacher, train_continual
from src.models.backbones import expand_classifier
from src.training.train import test_model


def fewshot_task_data(train_ds, test_ds, stream_classes, class_to_idx, task_size):
    """Partition the stream datasets into incremental tasks of task_size classes each, in order.
    Returns (task_train_subsets, task_test_subsets, task_sizes, task_class_lists). The train set is
    already few-shot (capped at preprocess time); this only splits it by class into tasks."""
    task_class_lists, train_subsets, test_subsets, task_sizes = [], [], [], []
    for start in range(0, len(stream_classes), task_size):
        chunk = stream_classes[start:start + task_size]
        ids   = set(class_to_idx[c] for c in chunk)
        tr    = [i for i, (_, y) in enumerate(train_ds.samples) if y in ids]
        te    = [i for i, (_, y) in enumerate(test_ds.samples) if y in ids]
        train_subsets.append(Subset(train_ds, tr))
        test_subsets.append(Subset(test_ds, te))
        task_sizes.append(len(chunk))
        task_class_lists.append(chunk)
    return train_subsets, test_subsets, task_sizes, task_class_lists


def _freeze_backbone(model):
    """Freeze the backbone so few-shot fitting only moves the LSTM + head (and stores no backbone
    activations for backward - keeps eval memory small on a free-Colab T4)."""
    for p in model.backbone.parameters():
        p.requires_grad_(False)


def run_fscil(model, base_train_loader, base_test_set, task_train_subsets, task_test_subsets,
              task_sizes, n_base, device, replay_per_class=5, epochs=3, lr=1e-4,
              lambda_distill=cfg.LAMBDA_DISTILL, T=cfg.KD_TEMPERATURE):
    """Run the incremental few-shot stream and return (model, accs). accs[0] is accuracy on the base
    classes before any task; accs[i] is accuracy on ALL classes seen through task i. The backbone is
    frozen; each task few-shot-fits the LSTM + head with replay + LwF, so old classes are retained."""
    model = model.to(device)
    _freeze_backbone(model)

    cap    = replay_per_class * (n_base + sum(task_sizes) + 5)
    buffer = ReplayBuffer(max_size=cap)
    fill_buffer(buffer, base_train_loader, per_class=replay_per_class)

    seen_tests  = [base_test_set]
    base_loader = DataLoader(ConcatDataset(seen_tests), batch_size=cfg.BATCH_SIZE, num_workers=2)
    acc0, _, _  = test_model(model, base_loader, device)
    accs        = [acc0]

    n_old = n_base
    for i in range(len(task_train_subsets)):
        teacher = snapshot_teacher(model)
        model   = expand_classifier(model, n_old + task_sizes[i]).to(device)
        _freeze_backbone(model)
        train_loader = DataLoader(task_train_subsets[i], batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(task_test_subsets[i],  batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
        train_continual(model, train_loader, val_loader, device, buffer=buffer, teacher=teacher,
                        num_old_classes=n_old, lambda_distill=lambda_distill, T=T,
                        epochs=epochs, lr=lr)
        fill_buffer(buffer, train_loader, per_class=replay_per_class)
        seen_tests.append(task_test_subsets[i])
        n_old += task_sizes[i]
        loader     = DataLoader(ConcatDataset(seen_tests), batch_size=cfg.BATCH_SIZE, num_workers=2)
        acc, _, _  = test_model(model, loader, device)
        accs.append(acc)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
    return model, accs
