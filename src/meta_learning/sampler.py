import random
import torch


class EmbeddingEpisodeSampler:
    """
    Samples N-way K-shot episodes directly from pre-extracted embedding tensors,
    avoiding repeated backbone forward passes during meta-training.

    Args:
        s_embs  (Tensor): Student embeddings  [N, student_hidden]
        t_embs  (Tensor): Teacher embeddings  [N, teacher_hidden]
        labels  (Tensor): Class indices        [N]
    """

    def __init__(self, s_embs: torch.Tensor, t_embs: torch.Tensor, labels: torch.Tensor):
        self.s_embs = s_embs
        self.t_embs = t_embs
        self.labels = labels

        self._lbl: dict[int, list[int]] = {}
        for i, lbl in enumerate(labels.tolist()):
            self._lbl.setdefault(int(lbl), []).append(i)
        self.valid = list(self._lbl.keys())

    def sample(
        self,
        n_way: int,
        k_sup: int,
        k_qry: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """
        Draw one episode.

        Returns:
            sup_s  — support student embeddings  [n_way * k_sup, D_s]
            sup_t  — support teacher embeddings  [n_way * k_sup, D_t]
            sup_y  — support labels              [n_way * k_sup]
            qry_s  — query   student embeddings  [n_way * k_qry, D_s]
            qry_y  — query   labels              [n_way * k_qry]
        """
        n_way  = min(n_way, len(self.valid))
        chosen = random.sample(self.valid, n_way)

        sup_s, sup_t, sup_y, qry_s, qry_y = [], [], [], [], []

        for lbl in chosen:
            idxs = self._lbl[lbl]
            need = k_sup + k_qry
            sel  = random.sample(idxs, min(need, len(idxs)))
            while len(sel) < need:
                sel += random.choices(idxs, k=need - len(sel))

            for i in sel[:k_sup]:
                sup_s.append(self.s_embs[i])
                sup_t.append(self.t_embs[i])
                sup_y.append(lbl)
            for i in sel[k_sup:]:
                qry_s.append(self.s_embs[i])
                qry_y.append(lbl)

        return (
            torch.stack(sup_s).to(device),
            torch.stack(sup_t).to(device),
            torch.tensor(sup_y, dtype=torch.long).to(device),
            torch.stack(qry_s).to(device),
            torch.tensor(qry_y, dtype=torch.long).to(device),
        )