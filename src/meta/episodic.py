"""
Episodic few-shot meta-learning on real clips: Reptile and first-order MAML.

Meta-training samples many N-way K-shot tasks from a pool of UCF101 clips so the model
learns an initialisation that adapts quickly. Meta-test then adapts that init to a new
few-shot task (e.g. a shifted YouTube domain) and reports query accuracy.
"""

import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config.config as cfg
from src.models.backbones import expand_classifier
from src.cl.rehearsal import train_continual, distillation_loss
from src.training.train import test_model


def build_pool(datasets, per_class):
    """Load up to per_class clips per class from one or more UCF101Clips datasets into RAM
    tensors. Returns (clips, labels) so episodes are sampled fast without re-reading disk."""
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


def make_video_disjoint_task(clips, labels, video_ids, class_ids, k_shot, device):
    """One VIDEO-DISJOINT few-shot task. For each class: SUPPORT = k_shot clips from ONE of its
    videos; QUERY = the class's other clips, each flagged new=True if it comes from a DIFFERENT
    video than the support (held-out) or new=False if from the same support video. Labels are
    relabelled 0..n-1. Returns (sx, sy, qx, qy, q_new). A class with only one video contributes
    no 'new' query."""
    sx, sy, qx, qy, q_new = [], [], [], [], []
    for new_label, c in enumerate(class_ids):
        c_idx   = (labels == c).nonzero(as_tuple=True)[0]
        c_vids  = video_ids[c_idx]
        uniq    = c_vids.unique()
        sup_vid = uniq[torch.randint(len(uniq), (1,)).item()].item()

        same = c_idx[c_vids == sup_vid]
        same = same[torch.randperm(len(same))]
        sup   = same[:k_shot]                    # support
        old_q = same[k_shot:]                    # same video, unseen clips
        new_q = c_idx[c_vids != sup_vid]         # held-out videos

        sx.append(clips[sup]);   sy += [new_label] * len(sup)
        qx.append(clips[old_q]); qy += [new_label] * len(old_q); q_new += [False] * len(old_q)
        qx.append(clips[new_q]); qy += [new_label] * len(new_q); q_new += [True]  * len(new_q)

    return (torch.cat(sx).to(device), torch.tensor(sy).to(device),
            torch.cat(qx).to(device), torch.tensor(qy).to(device),
            torch.tensor(q_new).to(device))


def few_shot_old_new(model, sx, sy, qx, qy, q_new, inner_lr, inner_steps, n_classes, batch_size=8):
    """Adapt on the support set, then return (old_per_class, new_per_class) query accuracy lists:
    'old' = clips from the support video (seen scene), 'new' = clips from held-out videos (the real
    generalisation test). nan for a class with no clips in that bucket. The query is scored in
    mini-batches so a large query set does not OOM a heavy backbone (e.g. ResNet50)."""
    adapted = few_shot_adapt(model, sx, sy, inner_lr, inner_steps)
    adapted.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, qx.size(0), batch_size):
            preds.append(adapted(qx[i:i + batch_size]).argmax(1))
    correct = torch.cat(preds) == qy
    old_pc, new_pc = [], []
    for c in range(n_classes):
        cls = qy == c
        old_mask = cls & (~q_new)
        new_mask = cls & q_new
        old_pc.append(correct[old_mask].float().mean().item() if int(old_mask.sum()) > 0 else float("nan"))
        new_pc.append(correct[new_mask].float().mean().item() if int(new_mask.sum()) > 0 else float("nan"))
    return old_pc, new_pc


def _adapt(model, sx, sy, inner_lr, inner_steps):
    """Copy the model and take inner_steps of SGD on the support set (only the trainable
    params - a frozen backbone stays put). Returns the copy."""
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


def few_shot_eval(model, sx, sy, qx, qy, inner_lr, inner_steps):
    """Adapt on the support set, then return query accuracy - the few-shot test."""
    adapted = few_shot_adapt(model, sx, sy, inner_lr, inner_steps)
    adapted.eval()
    with torch.no_grad():
        return (adapted(qx).argmax(1) == qy).float().mean().item()


def few_shot_per_class(model, sx, sy, qx, qy, inner_lr, inner_steps, n_classes):
    """Adapt on the support set, then return per-class query accuracy as a list indexed by the
    relabelled class 0..n_classes-1 (nan for a class with no query items). The overall accuracy
    is the mean of this list when the query set is class-balanced."""
    adapted = few_shot_adapt(model, sx, sy, inner_lr, inner_steps)
    adapted.eval()
    with torch.no_grad():
        preds = adapted(qx).argmax(1)
    accs = []
    for c in range(n_classes):
        mask = qy == c
        accs.append((preds[mask] == c).float().mean().item() if int(mask.sum()) > 0 else float("nan"))
    return accs


def adapt_new_and_retain(model, sx, sy, qx, qy, old_head, old_loader, inner_lr, inner_steps,
                         device, batch_size=8):
    """Few-shot adapt to a NEW task, then return (new_acc, old_retention):
      new_acc       - query accuracy on the NEW classes (did it learn them from K shots),
      old_retention - the ORIGINAL old-class head applied to the ADAPTED backbone over the OLD
                      test set (how much it still remembers; the drop from the pre-adapt value is
                      the forgetting).
    Query and old test are scored in mini-batches so a heavy backbone does not OOM."""
    adapted = few_shot_adapt(model, sx, sy, inner_lr, inner_steps)
    adapted.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, qx.size(0), batch_size):
            preds.append(adapted(qx[i:i + batch_size]).argmax(1))
    new_acc = (torch.cat(preds) == qy).float().mean().item()

    old_head = old_head.to(device)
    correct = total = 0
    with torch.no_grad():
        for clips, y in old_loader:
            clips, y = clips.to(device), y.to(device)
            correct += (old_head(adapted.features(clips)).argmax(1) == y).sum().item()
            total   += y.size(0)
    return new_acc, correct / max(total, 1)


def meta_train_maml(model, clips, labels, class_pool, n_way, k_shot, k_query,
                    epochs, episodes, inner_lr, inner_steps, meta_lr, device,
                    freeze_backbone=False, unfreeze_layer4=False, train_last=0):
    """First-order MAML: adapt on support, take the query gradient at the adapted weights,
    and apply it to the meta-weights. Returns (model, history).

    freeze_backbone: meta-learn only the trainable params (LSTM + head, plus part of the backbone as
    below). train_last: keep the LAST train_last backbone blocks trainable (the general lever - freeze
    the lower blocks so their features, which the OLD classes rely on, are preserved, while the upper
    blocks reshape for the new task); this works for any backbone (ResNet or the MobileNet student).
    unfreeze_layer4: the ResNet-only shorthand for train_last on layer4 (backbone[-2] = [...layer4,
    avgpool]); ignored when train_last > 0. Freezing the WHOLE backbone leaves only the small LSTM,
    which barely moves the features (a flat meta-train-vs-plain prototype score is the symptom), so
    train_last is the recommended setting for a weak backbone. On return the requires_grad state is set
    back to the same partial freeze, so the no-meta and MAML arms adapt the SAME parameters at meta-test
    - a fair comparison."""
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
                    p.requires_grad_(True)      # keep the last train_last blocks trainable at meta-test
        else:
            for p in model.backbone[-2].parameters():
                p.requires_grad_(True)          # restore constructor state: only layer4 trainable
    return model, history


def recalibrate_head(model, loader, n_classes, device, epochs=3, lr=1e-3):
    """Re-fit a FRESH n_classes linear head to the model's CURRENT features (backbone + LSTM frozen),
    so the head matches a backbone that meta-training has shifted. Meta-training moves the features but
    leaves the old head calibrated to the OLD features, which collapses old-class accuracy; a short
    linear probe on the old-class data restores it without touching the meta-learned features. This only
    puts the meta init on equal footing with the plain init (whose head already matches its backbone).
    All parameters are left trainable on return, so a later full fine-tune adapts the whole model.
    Returns the model."""
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


def adapt_and_eval(model, train_loader, val_loader, base_test_loader, new_test_loader,
                   n_base, n_new, device, buffer=None, teacher=None, epochs=5, lr=1e-4,
                   lambda_distill=cfg.LAMBDA_DISTILL, T=cfg.KD_TEMPERATURE, freeze_backbone=True):
    """Grow the head to n_base + n_new, adapt to the new classes for `epochs` epochs with train_continual
    (optional replay buffer + LwF teacher), then return (model, new_acc, base_retention, new_curve,
    base_curve): the adapted model, its accuracy on the new EXAM classes and on the old BASE classes
    through the same grown head, and the per-epoch new/base test accuracy (index 0 = before adapting) so
    the adaptation can be plotted - a rising new_curve shows the classes being learned, a falling
    base_curve shows forgetting. The model is returned so it can be reused (per-class accuracy, a
    confusion matrix) without re-training. The study arms call this with the same adaptation, differing
    only in the init (plain vs MAML) and whether replay/LwF are on. freeze_backbone=True keeps a strong
    backbone (ResNet50) fixed and tunes only LSTM + head; False fully fine-tunes a small backbone (the
    MobileNet student) so a weak backbone has room to actually learn the new classes."""
    model = model.to(device)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    model = expand_classifier(model, n_base + n_new).to(device)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    track = []
    track_loaders = {"new": new_test_loader, "base": base_test_loader}
    train_continual(model, train_loader, val_loader, device, buffer=buffer, teacher=teacher,
                    num_old_classes=n_base, lambda_distill=lambda_distill, T=T, epochs=epochs, lr=lr,
                    track_loaders=track_loaders, track_history=track)
    new_curve  = [d["new"]  for d in track]
    base_curve = [d["base"] for d in track]
    new_acc,  _, _ = test_model(model, new_test_loader,  device)
    base_acc, _, _ = test_model(model, base_test_loader, device)
    return model, new_acc, base_acc, new_curve, base_curve


def fair_adapt_eval(model, support_x, support_y, base_test_loader, new_test_loader, n_base, n_new,
                    device, buffer=None, teacher=None, inner_lr=0.01, inner_steps=5,
                    lambda_distill=cfg.LAMBDA_DISTILL, T=cfg.KD_TEMPERATURE, chunk=16,
                    freeze_backbone=True, return_model=False):
    """ONE matched adaptation used by EVERY arm, so the comparison is fully fair: grow the head to
    n_base + n_new, keep layer4 + LSTM + head trainable, and take inner_steps few-step SGD updates on
    the new-class support (global labels), with OPTIONAL replay (buffer of old exemplars mixed into
    each step) and OPTIONAL LwF (teacher distilling the old-class logits, with the distillation weight
    ramped 0 -> lambda_distill so new classes can learn before base protection tightens). Across arms only the init
    (plain vs MAML) and whether replay/LwF are on differ - same head, same steps, same lr, same
    support. chunk bounds the per-forward batch so many new classes still fit a free T4.
    freeze_backbone=True tunes only layer4 + LSTM + head (a strong ResNet50 backbone); False fully
    fine-tunes a small backbone (the MobileNet student), giving a weak backbone room to adapt. Returns
    (new_acc, base_retention), or (new_acc, base_retention, adapted_model) when return_model=True (for
    a confusion matrix off the few-shot model)."""
    model = model.to(device)
    model = expand_classifier(model, n_base + n_new).to(device)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        for p in model.backbone[-2].parameters():
            p.requires_grad_(True)                   # layer4 trainable (same as the no-meta init)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=inner_lr, momentum=0.9)
    if teacher is not None:
        teacher = teacher.to(device).eval()
    model.train()
    for step in range(inner_steps):
        x, y = support_x, support_y
        if buffer is not None:
            xb, yb = buffer.sample(support_x.size(0))
            if xb is not None:
                x = torch.cat([support_x, xb.to(device)]); y = torch.cat([support_y, yb.to(device)])
        # smooth LwF: ramp the distillation weight 0 -> lambda_distill across the steps, so the new
        # classes get to learn first and base protection tightens gradually (a fixed heavy lambda
        # from step 1 crushes the fresh new-class head before it can move).
        lam = lambda_distill * (step + 1) / inner_steps
        n = x.size(0)
        opt.zero_grad()
        # accumulate the gradient over chunks so a large support+replay batch (many new classes) does
        # not OOM a free-Colab T4 - each chunk's mean loss is weighted by its share of the full batch.
        for i in range(0, n, chunk):
            xc, yc = x[i:i + chunk], y[i:i + chunk]
            w = xc.size(0) / n
            logits = model(xc)
            loss = F.cross_entropy(logits, yc)
            if teacher is not None:
                with torch.no_grad():
                    t = teacher(xc)
                loss = loss + lam * distillation_loss(logits[:, :n_base], t[:, :n_base], T)
            (loss * w).backward()
        opt.step()
    new_acc,  _, _ = test_model(model, new_test_loader,  device)
    base_acc, _, _ = test_model(model, base_test_loader, device)
    if return_model:
        return new_acc, base_acc, model
    return new_acc, base_acc
