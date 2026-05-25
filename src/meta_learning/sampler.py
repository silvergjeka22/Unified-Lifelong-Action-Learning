import random
import torch


def sample_episode(s_embs, t_embs, labels, n_way, k_support, k_query, device):
    by_class = {}
    for i, y in enumerate(labels.tolist()):
        by_class.setdefault(int(y), []).append(i)

    classes = list(by_class.keys())
    n_way = min(n_way, len(classes))
    chosen = random.sample(classes, n_way)

    sup_s, sup_t, sup_y = [], [], []
    qry_s, qry_y = [], []

    for cls in chosen:
        idxs = by_class[cls]
        need = k_support + k_query

        if len(idxs) >= need:
            picked = random.sample(idxs, need)
        else:
            picked = idxs[:] + random.choices(idxs, k=need - len(idxs))

        for i in picked[:k_support]:
            sup_s.append(s_embs[i])
            sup_t.append(t_embs[i])
            sup_y.append(cls)

        for i in picked[k_support:]:
            qry_s.append(s_embs[i])
            qry_y.append(cls)

    return (
        torch.stack(sup_s).to(device),
        torch.stack(sup_t).to(device),
        torch.tensor(sup_y, dtype=torch.long).to(device),
        torch.stack(qry_s).to(device),
        torch.tensor(qry_y, dtype=torch.long).to(device),
    )


def limit_to_k_per_class(s_embs, t_embs, labels, k_per_class=20, seed=42):
    rng = random.Random(seed)
    by_class = {}

    for i, y in enumerate(labels.tolist()):
        by_class.setdefault(int(y), []).append(i)

    keep = []
    for cls in sorted(by_class.keys()):
        idxs = by_class[cls][:]
        rng.shuffle(idxs)
        keep.extend(idxs[:k_per_class])

    keep = torch.tensor(keep, dtype=torch.long)
    return s_embs[keep], t_embs[keep], labels[keep]