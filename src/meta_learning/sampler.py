import random
import torch


def sample(
    s_embs,
    t_embs,
    labels,
    n_way,
    k_sup,
    k_qry,
    device,
):
    lbl: dict[int, list[int]] = {}
    for i, l in enumerate(labels.tolist()):
        lbl.setdefault(int(l), []).append(i)
    valid = list(lbl.keys())

    n_way  = min(n_way, len(valid))
    chosen = random.sample(valid, n_way)

    sup_s, sup_t, sup_y, qry_s, qry_y = [], [], [], [], []

    for l in chosen:
        idxs = lbl[l]
        need = k_sup + k_qry
        sel  = random.sample(idxs, min(need, len(idxs)))
        while len(sel) < need:
            sel += random.choices(idxs, k=need - len(sel))

        for i in sel[:k_sup]:
            sup_s.append(s_embs[i])
            sup_t.append(t_embs[i])
            sup_y.append(l)
        for i in sel[k_sup:]:
            qry_s.append(s_embs[i])
            qry_y.append(l)

    return (
        torch.stack(sup_s).to(device),
        torch.stack(sup_t).to(device),
        torch.tensor(sup_y, dtype=torch.long).to(device),
        torch.stack(qry_s).to(device),
        torch.tensor(qry_y, dtype=torch.long).to(device),
    )