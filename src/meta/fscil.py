"""
Few-shot class-incremental learning (FSCIL): add novel classes a task at a time from only a few
clips each, without forgetting the base classes.

Two classifiers live here:
  - run_fscil        : a gradient-trained growing head with replay + LwF. On a frozen backbone with
                       only a few shots the replay/LwF pull drowns the tiny new-class signal, so the
                       new classes barely learn - kept for reference.
  - run_fscil_proto  : a nearest-class-mean PROTOTYPE classifier (the standard FSCIL fix). Each class
                       is the mean feature of its few clips; classify by nearest prototype. No
                       gradient, no replay, no forgetting - and it actually learns the new classes.

meta_train_protonet meta-trains the features (ProtoNet) so prototypes are more separable - the
metric-learning cousin of MAML that pairs with the prototype classifier.
"""

import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset

import src.config.config as cfg
from src.cl.rehearsal import ReplayBuffer, fill_buffer, snapshot_teacher, train_continual
from src.models.backbones import expand_classifier
from src.training.train import test_model
from src.meta.episodic import sample_episode


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


def class_prototypes(model, loader, device):
    """Return {class_id: mean 256-d feature} over a loader - each class's prototype. The backbone is
    frozen, so a class's prototype is a fixed point that never drifts (no forgetting)."""
    sums, counts = {}, {}
    model.eval()
    with torch.no_grad():
        for clips, y in loader:
            feats = model.features(clips.to(device))
            for i in range(y.size(0)):
                c = int(y[i])
                sums[c]   = feats[i] if c not in sums else sums[c] + feats[i]
                counts[c] = counts.get(c, 0) + 1
    return {c: sums[c] / counts[c] for c in sums}


def proto_accuracy(model, loader, protos, device):
    """Nearest-prototype accuracy (cosine similarity) over a loader given {class_id: prototype}."""
    ids = sorted(protos.keys())
    bank = F.normalize(torch.stack([protos[c] for c in ids]).to(device), dim=1)
    id_tensor = torch.tensor(ids, device=device)
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for clips, y in loader:
            feats = F.normalize(model.features(clips.to(device)), dim=1)
            pred  = id_tensor[(feats @ bank.t()).argmax(1)]
            correct += (pred == y.to(device)).sum().item()
            total   += y.size(0)
    return correct / max(total, 1)


def run_fscil_proto(model, base_train_loader, base_test_set, task_train_subsets, task_test_subsets, device):
    """Prototype-based FSCIL. Build a mean-feature prototype per class from its (few) clips, add each
    task's prototypes incrementally, and classify by nearest prototype. No gradient, no replay, no
    forgetting (base prototypes from the frozen backbone never change). Returns accs where accs[0] is
    the base accuracy and accs[i] is accuracy on ALL classes seen through task i."""
    model = model.to(device)
    protos = class_prototypes(model, base_train_loader, device)
    seen_tests  = [base_test_set]
    base_loader = DataLoader(ConcatDataset(seen_tests), batch_size=cfg.BATCH_SIZE, num_workers=2)
    accs = [proto_accuracy(model, base_loader, protos, device)]
    for i in range(len(task_train_subsets)):
        tl = DataLoader(task_train_subsets[i], batch_size=cfg.BATCH_SIZE, num_workers=2)
        protos.update(class_prototypes(model, tl, device))
        seen_tests.append(task_test_subsets[i])
        loader = DataLoader(ConcatDataset(seen_tests), batch_size=cfg.BATCH_SIZE, num_workers=2)
        accs.append(proto_accuracy(model, loader, protos, device))
        torch.cuda.empty_cache()
    return accs


def meta_train_protonet(model, clips, labels, class_pool, n_way, k_shot, k_query,
                        epochs, episodes, lr, device, freeze_backbone=True):
    """ProtoNet episodic meta-training - the metric-learning cousin of MAML that PAIRS with a
    prototype classifier. Each episode builds prototypes from the support set, classifies the query
    by cosine similarity to them, and backprops the CE loss into the (LSTM) features so classes
    become more separable. Returns (model, history). freeze_backbone keeps the strong ResNet50."""
    model.to(device)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    history = {"query_accs": []}
    for epoch in range(epochs):
        accs = []
        for _ in range(episodes):
            sx, sy, qx, qy = sample_episode(clips, labels, class_pool, n_way, k_shot, k_query, device)
            model.train()
            sf = F.normalize(model.features(sx), dim=1)
            qf = F.normalize(model.features(qx), dim=1)
            protos = torch.stack([sf[sy == c].mean(0) for c in range(n_way)])
            protos = F.normalize(protos, dim=1)
            logits = qf @ protos.t()
            loss = F.cross_entropy(logits, qy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            accs.append((logits.argmax(1) == qy).float().mean().item())
        history["query_accs"].append(sum(accs) / len(accs))
        print(f"[protonet] epoch {epoch+1}/{epochs} | meta query acc {history['query_accs'][-1]:.4f}")
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(True)
    return model, history
