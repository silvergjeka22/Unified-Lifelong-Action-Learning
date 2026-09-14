"""
Episodic few-shot meta-learning on real clips (first-order MAML).

Meta-training samples many N-way K-shot tasks from a pool of UCF101 clips so the model learns an
initialisation that adapts quickly to new classes. adapt_and_eval is the single adaptation every study
arm uses; the arms differ only in the init (plain vs MAML) and whether replay / LwF are on.
"""

import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config.config as cfg
from src.models.backbones import expand_classifier
from src.cl.rehearsal import train_continual
from src.training.train import evaluate_model


def build_pool(datasets, per_class):
    """Load up to per_class clips per class from one or more UCF101Clips datasets into RAM tensors.
    Returns (clips, labels) so episodes are sampled fast without re-reading disk."""
    by_class = {}
    for ds in datasets:
        for path, label in ds.samples:
            by_class.setdefault(label, []).append(path)
    clips, labels = [], []
    for label, paths in by_class.items():
        for p in paths[:per_class]:
            clips.append(torch.load(p))
            labels.append(label)
    return torch.stack(clips), torch.tensor(labels)


def make_task(clips, labels, class_ids, k_shot, k_query, device):
    """Build one support/query split from the given class ids, relabelled 0..n-1.
    Returns support_x, support_y, query_x, query_y."""
    sx, sy, qx, qy = [], [], [], []
    for new_label, c in enumerate(class_ids):
        idx  = (labels == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx))]
        pick = perm[:k_shot + k_query]
        sx.append(clips[pick[:k_shot]]);            sy += [new_label] * k_shot
        qx.append(clips[pick[k_shot:]]);            qy += [new_label] * (len(pick) - k_shot)
    return (torch.cat(sx).to(device), torch.tensor(sy).to(device),
            torch.cat(qx).to(device), torch.tensor(qy).to(device))


def sample_episode(clips, labels, class_pool, n_way, k_shot, k_query, device):
    """One random N-way K-shot episode from the pool."""
    return make_task(clips, labels, random.sample(class_pool, n_way), k_shot, k_query, device)


def _adapt(model, sx, sy, inner_lr, inner_steps):
    """Copy the model and take inner_steps of SGD on the support set (only the trainable params -
    a frozen backbone stays put). Returns the copy."""
    adapted = copy.deepcopy(model)
    params = [p for p in adapted.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=inner_lr, momentum=0.9)
    adapted.train()
    for _ in range(inner_steps):
        opt.zero_grad()
        F.cross_entropy(adapted(sx), sy).backward()
        opt.step()
    return adapted


def few_shot_adapt(model, sx, sy, inner_lr, inner_steps):
    """Return a copy of the model adapted to the support set (a few SGD steps)."""
    return _adapt(model, sx, sy, inner_lr, inner_steps)


def meta_train_maml(model, clips, labels, class_pool, n_way, k_shot, k_query,
                    epochs, episodes, inner_lr, inner_steps, meta_lr, device,
                    freeze_backbone=False, unfreeze_layer4=False, train_last=0):
    """First-order MAML: adapt on support, take the query gradient at the adapted weights, and apply it
    to the meta-weights. Returns (model, history).

    freeze_backbone with train_last=N keeps only the LAST N backbone blocks trainable (plus LSTM + head):
    the lower blocks stay frozen so the OLD classes' features are preserved, while the upper blocks
    reshape for the new task. unfreeze_layer4 is the ResNet-only shorthand (layer4 = backbone[-2]) and is
    ignored when train_last > 0. On return the same partial freeze is restored, so no-meta and MAML adapt
    the same parameters at meta-test - a fair comparison."""
    model.to(device)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        if train_last > 0:
            for child in list(model.backbone)[-train_last:]:
                for p in child.parameters():
                    p.requires_grad_(True)
        elif unfreeze_layer4:
            for p in model.backbone[-2].parameters():
                p.requires_grad_(True)
    meta_params = [p for p in model.parameters() if p.requires_grad]
    meta_opt = torch.optim.Adam(meta_params, lr=meta_lr)
    history = {"query_accs": []}
    for epoch in range(epochs):
        accs = []
        for _ in range(episodes):
            sx, sy, qx, qy = sample_episode(clips, labels, class_pool, n_way, k_shot, k_query, device)
            adapted = _adapt(model, sx, sy, inner_lr, inner_steps)
            loss  = F.cross_entropy(adapted(qx), qy)
            grads = torch.autograd.grad(loss, [p for p in adapted.parameters() if p.requires_grad])
            meta_opt.zero_grad()
            for p, g in zip(meta_params, grads):
                p.grad = g.detach()
            meta_opt.step()
            with torch.no_grad():
                accs.append((adapted(qx).argmax(1) == qy).float().mean().item())
        history["query_accs"].append(sum(accs) / len(accs))
        print(f"[maml] epoch {epoch+1}/{epochs} | meta query acc {history['query_accs'][-1]:.4f}")
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        if train_last > 0:
            for child in list(model.backbone)[-train_last:]:
                for p in child.parameters():
                    p.requires_grad_(True)
        else:
            for p in model.backbone[-2].parameters():
                p.requires_grad_(True)
    return model, history


def recalibrate_head(model, loader, n_classes, device, epochs=3, lr=1e-3):
    """Re-fit a FRESH n_classes linear head to the model's CURRENT features (backbone + LSTM frozen), so
    the head matches a backbone that meta-training has shifted. Meta-training moves the features but
    leaves the old head calibrated to the OLD features, which collapses old-class accuracy; a short linear
    probe on the old-class data restores it without touching the meta-learned features. This only puts the
    meta init on equal footing with the plain init. All parameters are left trainable on return. Returns
    the model."""
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, n_classes).to(device)
    opt = torch.optim.Adam(model.fc.parameters(), lr=lr)
    model.eval()
    for _ in range(epochs):
        for clips, y in loader:
            clips, y = clips.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(clips), y).backward()
            opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def adapt_and_eval(model, train_loader, base_test_loader, new_test_loader, n_base, n_new, device,
                   buffer=None, teacher=None, epochs=3, lr=1e-4,
                   lambda_distill=cfg.LAMBDA_DISTILL, T=cfg.KD_TEMPERATURE, verbose=True, track=True):
    """The ONE adaptation every study arm uses: grow the head to n_base + n_new and fully fine-tune on the
    new classes for `epochs` epochs, with optional replay (buffer) and LwF (teacher). Arms differ only in
    the init (plain vs MAML) and whether buffer / teacher are passed.

    Returns (model, new_acc, base_acc, new_curve, base_curve): the adapted model (for a confusion matrix),
    the final new-class and base accuracy, and - when track is True - the per-epoch new/base test accuracy
    (index 0 = before adapting) for the adaptation-curve plot. verbose=False hides the per-epoch log (used
    by the shots sweep so it prints one line per shot); track=False skips the per-epoch evaluation."""
    model = model.to(device)
    model = expand_classifier(model, n_base + n_new).to(device)
    if track:
        history = []
        track_loaders = {"new": new_test_loader, "base": base_test_loader}
        train_continual(model, train_loader, new_test_loader, device, buffer=buffer, teacher=teacher,
                        num_old_classes=n_base, lambda_distill=lambda_distill, T=T, epochs=epochs, lr=lr,
                        track_loaders=track_loaders, track_history=history, verbose=verbose)
        new_curve  = [round(d["new"], 4)  for d in history]
        base_curve = [round(d["base"], 4) for d in history]
        new_acc, base_acc = new_curve[-1], base_curve[-1]
    else:
        train_continual(model, train_loader, new_test_loader, device, buffer=buffer, teacher=teacher,
                        num_old_classes=n_base, lambda_distill=lambda_distill, T=T, epochs=epochs, lr=lr,
                        verbose=verbose)
        new_acc,  _ = evaluate_model(model, new_test_loader,  device)
        base_acc, _ = evaluate_model(model, base_test_loader, device)
        new_curve, base_curve = [], []
    return model, new_acc, base_acc, new_curve, base_curve


def proto_accuracy(model, train_loader, test_loader, device):
    """Nearest-class-mean accuracy from the model's frozen features: build one prototype (mean feature)
    per class from train_loader, then classify each test clip by the nearest prototype (cosine). A higher
    score means the features separate these classes better - a probe of what meta-training reshaped."""
    model = model.to(device).eval()
    sums, counts = {}, {}
    with torch.no_grad():
        for clips, y in train_loader:
            feats = model.features(clips.to(device))
            for i in range(y.size(0)):
                c = int(y[i])
                sums[c] = feats[i] if c not in sums else sums[c] + feats[i]
                counts[c] = counts.get(c, 0) + 1
    ids = sorted(sums)
    protos = F.normalize(torch.stack([sums[c] / counts[c] for c in ids]), dim=1)
    ids_t = torch.tensor(ids, device=device)
    correct = total = 0
    with torch.no_grad():
        for clips, y in test_loader:
            feats = F.normalize(model.features(clips.to(device)), dim=1)
            pred = ids_t[(feats @ protos.t()).argmax(1)]
            correct += (pred.cpu() == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)
