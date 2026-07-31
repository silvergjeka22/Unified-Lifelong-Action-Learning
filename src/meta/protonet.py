import torch
import torch.nn.functional as F


@torch.no_grad()
def proto_eval(head, support_s, support_y, query_s, query_y, device, metric="euclidean"):
    """
    Prototypical-network few-shot accuracy on frozen features. Returns accuracy (float).

    Embeds the support and query clips through the head's frozen LSTM, builds one
    prototype (mean embedding) per class, and assigns each query to its nearest
    prototype. No gradient adaptation happens - this is the SimpleCIL-style metric
    baseline for a frozen backbone, and the counterpoint to Reptile's inner-loop SGD.
    """
    head.eval()
    se = head.embed(support_s.to(device)).cpu()
    qe = head.embed(query_s.to(device)).cpu()

    classes = sorted(support_y.unique().tolist())
    protos  = torch.stack([se[support_y == c].mean(0) for c in classes])

    if metric == "cosine":
        sims     = F.normalize(qe, dim=1) @ F.normalize(protos, dim=1).t()
        pred_idx = sims.argmax(1)
    else:
        pred_idx = torch.cdist(qe, protos).argmin(1)

    proto_labels = torch.tensor(classes)
    preds        = proto_labels[pred_idx]
    return (preds == query_y).float().mean().item()
