"""Few-shot meta-learning on cached features: Reptile and first-order MAML."""

import copy
import random

import torch
import torch.nn.functional as F


def sample_episode(feats, labels, class_pool, n_way, k_shot, k_query, device):
    """Sample one n-way episode. Returns support_x, support_y, query_x, query_y relabelled 0..n_way-1."""
    classes = random.sample(class_pool, n_way)
    sx, sy, qx, qy = [], [], [], []
    for new_label, c in enumerate(classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx))]
        pick = perm[:k_shot + k_query]
        sx.append(feats[pick[:k_shot]])
        sy.append(torch.full((k_shot,), new_label, dtype=torch.long))
        if k_query > 0:
            qx.append(feats[pick[k_shot:k_shot + k_query]])
            qy.append(torch.full((k_query,), new_label, dtype=torch.long))
    support_x = torch.cat(sx).to(device)
    support_y = torch.cat(sy).to(device)
    if k_query > 0:
        query_x = torch.cat(qx).to(device)
        query_y = torch.cat(qy).to(device)
        return support_x, support_y, query_x, query_y
    return support_x, support_y, None, None


def inner_adapt(head, support_x, support_y, inner_lr, inner_steps):
    """Copy the head and take inner_steps SGD steps on the support set. Returns the adapted copy."""
    adapted = copy.deepcopy(head)
    adapted.train()
    params = [p for p in adapted.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=inner_lr)
    for step in range(inner_steps):
        opt.zero_grad()
        logits, _ = adapted(support_x, training=True)
        loss = F.cross_entropy(logits, support_y)
        loss.backward()
        opt.step()
    return adapted


def train_reptile(head, feats, labels, class_pool, n_way, k_shot, epochs, episodes,
                  inner_lr, inner_steps, epsilon, device):
    """Meta-train a head with Reptile: move the meta-weights toward each episode's adapted weights."""
    head.to(device)
    meta_params = [p for p in head.parameters() if p.requires_grad]
    for epoch in range(1, epochs + 1):
        for episode in range(episodes):
            support_x, support_y, _, _ = sample_episode(feats, labels, class_pool, n_way, k_shot, 0, device)
            adapted = inner_adapt(head, support_x, support_y, inner_lr, inner_steps)
            adapted_params = [p for p in adapted.parameters() if p.requires_grad]
            for mp, ap in zip(meta_params, adapted_params):
                mp.data.add_(epsilon * (ap.data - mp.data))
        print(f"[reptile] epoch {epoch}/{epochs}")
    head.eval()
    return head


def train_maml(head, feats, labels, class_pool, n_way, k_shot, k_query, epochs, episodes,
               inner_lr, inner_steps, meta_lr, device):
    """Meta-train a head with first-order MAML: optimise the post-adaptation query loss."""
    head.to(device)
    meta_params = [p for p in head.parameters() if p.requires_grad]
    meta_opt = torch.optim.Adam(meta_params, lr=meta_lr)
    for epoch in range(1, epochs + 1):
        for episode in range(episodes):
            support_x, support_y, query_x, query_y = sample_episode(feats, labels, class_pool, n_way, k_shot, k_query, device)
            adapted = inner_adapt(head, support_x, support_y, inner_lr, inner_steps)
            adapted_params = [p for p in adapted.parameters() if p.requires_grad]
            logits, _ = adapted(query_x, training=True)
            loss = F.cross_entropy(logits, query_y)
            grads = torch.autograd.grad(loss, adapted_params)
            meta_opt.zero_grad()
            for mp, g in zip(meta_params, grads):
                mp.grad = g.detach()
            meta_opt.step()
        print(f"[maml] epoch {epoch}/{epochs}")
    head.eval()
    return head


def few_shot_eval(head, feats, labels, class_pool, n_way, k_shot, k_query,
                  inner_lr, inner_steps, episodes, device):
    """Mean query accuracy over episodes after k-shot adaptation from the head."""
    accs = []
    for episode in range(episodes):
        support_x, support_y, query_x, query_y = sample_episode(feats, labels, class_pool, n_way, k_shot, k_query, device)
        adapted = inner_adapt(head, support_x, support_y, inner_lr, inner_steps)
        adapted.eval()
        with torch.no_grad():
            logits, _ = adapted(query_x, training=False)
            acc = (logits.argmax(1) == query_y).float().mean().item()
        accs.append(acc)
    return sum(accs) / max(len(accs), 1)
