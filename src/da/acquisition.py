import torch
import torch.nn.functional as F


@torch.no_grad()
def entropy_score(model, clips, device, batch_size=16):
    """Predictive entropy per clip. Higher = the model is more uncertain."""
    model.eval()
    scores = []
    for i in range(0, len(clips), batch_size):
        logits = model(clips[i:i + batch_size].to(device))
        p = F.softmax(logits, dim=1)
        scores.append((-(p * p.clamp(min=1e-9).log()).sum(1)).cpu())
    return torch.cat(scores)


@torch.no_grad()
def margin_score(model, clips, device, batch_size=16):
    """Negative top-2 softmax gap per clip. Higher = more uncertain."""
    model.eval()
    scores = []
    for i in range(0, len(clips), batch_size):
        p = F.softmax(model(clips[i:i + batch_size].to(device)), dim=1)
        top2 = p.topk(2, dim=1).values
        scores.append((-(top2[:, 0] - top2[:, 1])).cpu())
    return torch.cat(scores)


def select_top_k(model, clips, labels, k_per_class, device, strategy="entropy"):
    """
    Score a labelled pool and keep the top-k most informative clips per class.
    Returns (selected_clips, selected_labels).
    """
    if strategy == "entropy":
        scores = entropy_score(model, clips, device)
    elif strategy == "margin":
        scores = margin_score(model, clips, device)
    else:
        scores = torch.rand(len(clips))

    keep = []
    for c in labels.unique().tolist():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        order = idx[scores[idx].argsort(descending=True)][:k_per_class]
        keep.append(order)
        print(f"  class {c}: {len(idx)} clips -> selected {len(order)}  [{strategy}]")
    keep = torch.cat(keep)
    return clips[keep], labels[keep]
