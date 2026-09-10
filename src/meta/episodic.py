"""
Episodic few-shot meta-learning on real clips: Reptile and first-order MAML.

Meta-training samples many N-way K-shot tasks from a pool of UCF101 clips so the model
learns an initialisation that adapts quickly. Meta-test then adapts that init to a new
few-shot task (e.g. a shifted YouTube domain) and reports query accuracy.
"""

import copy
import random
import torch
import torch.nn.functional as F


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


def meta_train_maml(model, clips, labels, class_pool, n_way, k_shot, k_query,
                    epochs, episodes, inner_lr, inner_steps, meta_lr, device,
                    freeze_backbone=False):
    """First-order MAML: adapt on support, take the query gradient at the adapted weights,
    and apply it to the meta-weights. Returns (model, history).

    freeze_backbone: meta-learn only the LSTM + head and keep the (pretrained/distilled)
    backbone fixed. Meta-training a strong backbone can DEGRADE it, so freezing keeps the good
    features and just learns a fast-adapting head. requires_grad is restored before returning,
    so meta-test adaptation is the same full-model adaptation as the no-meta arm."""
    model.to(device)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
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
            p.requires_grad_(True)              # restore -> meta-test adapts the full model, like no-meta
    return model, history
