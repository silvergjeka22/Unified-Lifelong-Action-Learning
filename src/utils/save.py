import torch

from src.meta_learning.training import evaluate


# recording meta-learning results
results_t1, results_t2 = {}, {}


def record(
    name,
    model,
    results_dict,
    old_loader,           # T0 test (T1) or T0+T1 test (T2)
    new_loader,           # T1 test (T1) or T2 test (T2)
    all_loader,           # combined test — primary CL metric
    device,
    embed_loader=None,    # any loader for UMAP embeddings (val, test, or None to skip)
    class_names=None,     # list of all class name strings seen so far
    W0=None,              # pre-training weight snapshot for ∆W
):
    model.eval()

    # accuracies for old and new tasks
    _, old_acc = evaluate(model, old_loader, device)
    _, new_acc = evaluate(model, new_loader, device)

    # single pass over all_loader — confusion matrix + combined accuracy
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in all_loader:
            logits = model(x.to(device))
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y)
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # derive all_acc directly from predictions — no second pass needed
    all_acc = float((all_preds == all_labels).mean())

    # UMAP embeddings — extracted from embed_loader (explicit, any split)
    embeddings, emb_labels = None, None
    if embed_loader is not None:
        emb_list, lab_list = [], []
        with torch.no_grad():
            for x, y in embed_loader:
                h = model.get_embedding(x.to(device))
                emb_list.append(h.cpu())
                lab_list.append(y)
        embeddings = torch.cat(emb_list).numpy()
        emb_labels = torch.cat(lab_list).numpy()

    # weight delta per layer (only when W0 is passed)
    weight_delta = {}
    if W0 is not None:
        for k, v in model.state_dict().items():
            weight_delta[k] = (v.cpu().float() - W0[k].cpu().float()).norm().item()

    # store everything
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

    print(
        f"  [{name}] "
        f"Old: {old_acc:.4f} | New: {new_acc:.4f} | All: {all_acc:.4f}"
    )