"""
Active Learning acquisition functions for domain adaptation.

All functions operate on pre-extracted student embeddings and a trained
EmbeddingHead — no raw video frames needed.

Functions:
    entropy_score    — higher = more uncertain (preferred for AL)
    margin_score     — higher = more uncertain
    coreset_score    — higher = more diverse / underrepresented
    select_top_k     — unified interface: score + select top-K per class
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def entropy_score(head, embs: torch.Tensor, device, batch_size: int = 64) -> torch.Tensor:
    """
    Predictive entropy: H = -Σ p(y|x) log p(y|x).
    High entropy → model is confused → good candidate for labelling.
    """
    head.eval()
    scores = []
    for i in range(0, len(embs), batch_size):
        s = embs[i : i + batch_size].to(device)
        logits, _ = head(s, training=False)
        p = F.softmax(logits, dim=1)
        H = -(p * p.clamp(min=1e-9).log()).sum(dim=1)
        scores.append(H.cpu())
    return torch.cat(scores)


@torch.no_grad()
def margin_score(head, embs: torch.Tensor, device, batch_size: int = 64) -> torch.Tensor:
    """
    Margin uncertainty: gap between top-2 softmax scores.
    Small margin → model is uncertain between two classes.
    Negated so higher = more uncertain (consistent with entropy_score).
    """
    head.eval()
    scores = []
    for i in range(0, len(embs), batch_size):
        s = embs[i : i + batch_size].to(device)
        logits, _ = head(s, training=False)
        top2, _   = F.softmax(logits, dim=1).topk(2, dim=1)
        scores.append(-(top2[:, 0] - top2[:, 1]).cpu())
    return torch.cat(scores)


def coreset_score(embs: torch.Tensor) -> torch.Tensor:
    """
    Coreset diversity score: distance of each sample to the class prototype.
    High score → sample is far from the mean → maximises coverage.
    No model needed.
    """
    mu    = embs.mean(0)
    dists = (embs - mu).norm(dim=1)
    return dists


def select_top_k(
    embs_by_class: dict,
    strategy: str = "entropy",
    k: int = 10,
    head=None,
    device=None,
) -> dict:
    """
    Score and select top-K samples per class using the chosen strategy.

    Args:
        embs_by_class : {class_name: (N, D) tensor of student embeddings}
        strategy      : "entropy" | "margin" | "coreset" | "random"
        k             : number of samples to select per class
        head          : EmbeddingHead (required for entropy/margin)
        device        : torch.device (required for entropy/margin)

    Returns:
        {class_name: list of selected indices into embs_by_class[class_name]}
    """
    selected = {}

    for cls_name, embs in embs_by_class.items():
        n = embs.shape[0]

        if strategy == "entropy":
            scores = entropy_score(head, embs, device)
        elif strategy == "margin":
            scores = margin_score(head, embs, device)
        elif strategy == "coreset":
            scores = coreset_score(embs)
        else:  # random
            scores = torch.rand(n)

        top_k         = min(k, n)
        top_idx       = scores.argsort(descending=True)[:top_k].tolist()
        selected[cls_name] = top_idx

        print(f"  {cls_name:<20}: {n} clips -> selected {top_k}  "
              f"[{strategy}]  max={scores.max():.3f} min={scores.min():.3f}")

    return selected
