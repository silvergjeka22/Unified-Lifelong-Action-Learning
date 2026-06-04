import torch
import numpy as np


def remap_teacher_checkpoint(raw_ckpt: dict) -> dict:
    """
    Remap legacy ResNet key names to backbone.* and drop num_batches_tracked buffers.
    Handles checkpoints saved from ResNet50LSTM (resnet.*) for ResNet50LSTMTeacher (backbone.*).
    """
    remapped, n_dropped = {}, 0
    for k, v in raw_ckpt.items():
        if k.endswith("num_batches_tracked"):
            n_dropped += 1
            continue
        remapped["backbone." + k[len("resnet."):] if k.startswith("resnet.") else k] = v
    print(f"  Dropped {n_dropped} num_batches_tracked buffer(s).")
    return remapped


# ── Recording containers ──────────────────────────────────────────────────────
results_t1, results_t2 = {}, {}


@torch.no_grad()
def evaluate(model, loader, device):
    """Simple accuracy + predictions sweep over a loader."""
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        if isinstance(logits, tuple):
            logits = logits[0]
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y)
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc    = float((preds == labels).mean())
    return preds, acc


def record(
    name,
    model,
    results_dict,
    old_loader,
    new_loader,
    all_loader,
    device,
    embed_loader=None,
    class_names=None,
    W0=None,
):
    """
    Evaluate model on old/new/combined loaders, optionally extract embeddings
    and weight deltas, and store everything in results_dict[name].
    """
    model.eval()

    _, old_acc = evaluate(model, old_loader, device)
    _, new_acc = evaluate(model, new_loader, device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in all_loader:
            logits = model(x.to(device))
            if isinstance(logits, tuple):
                logits = logits[0]
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y)
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_acc    = float((all_preds == all_labels).mean())

    # Optional UMAP/t-SNE embeddings
    embeddings, emb_labels = None, None
    if embed_loader is not None:
        emb_list, lab_list = [], []
        with torch.no_grad():
            for x, y in embed_loader:
                out = model(x.to(device))
                h   = out[1] if isinstance(out, tuple) else out
                emb_list.append(h.cpu())
                lab_list.append(y)
        embeddings = torch.cat(emb_list).numpy()
        emb_labels = torch.cat(lab_list).numpy()

    # Weight delta per layer
    weight_delta = {}
    if W0 is not None:
        for k, v in model.state_dict().items():
            weight_delta[k] = (v.cpu().float() - W0[k].cpu().float()).norm().item()

    results_dict[name] = {
        "old_acc":      old_acc,
        "new_acc":      new_acc,
        "all_acc":      all_acc,
        "preds":        all_preds,
        "labels":       all_labels,
        "embeddings":   embeddings,
        "emb_labels":   emb_labels,
        "weight_delta": weight_delta,
        "class_names":  class_names,
    }

    print(f"  [{name}] Old: {old_acc:.4f} | New: {new_acc:.4f} | All: {all_acc:.4f}")
