import random
import torch

class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.data = []

    def add_batch(self, x, y):
        for i in range(len(x)):
            lbl = y[i].item() if torch.is_tensor(y[i]) else int(y[i])
            self.data.append((x[i].cpu(), lbl))
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]

    def add_from_loader(self, loader, max_per_class=None):
        counts = {}
        for x, y in loader:
            for i in range(x.size(0)):
                lbl = y[i].item()
                if max_per_class is not None and counts.get(lbl, 0) >= max_per_class:
                    continue
                self.data.append((x[i].cpu(), lbl))
                counts[lbl] = counts.get(lbl, 0) + 1

        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]

    def sample(self, batch_size):
        if not self.data:
            return None, None
        picked = random.sample(self.data, min(batch_size, len(self.data)))
        x, y = zip(*picked)
        return torch.stack(x), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.data)


class EpisodeBuffer:
    def __init__(self):
        self.data = {}
        self.new_classes = set()

    def add_from_loader(self, loader, mark_new=False, max_per_class=None):
        counts = {}
        added = set()

        for x, y in loader:
            for i in range(x.size(0)):
                lbl = y[i].item()
                if max_per_class is not None and counts.get(lbl, 0) >= max_per_class:
                    continue
                self.data.setdefault(lbl, []).append(x[i].cpu())
                counts[lbl] = counts.get(lbl, 0) + 1
                added.add(lbl)

        if mark_new:
            self.new_classes.update(added)

    def available_classes(self):
        return list(self.data.keys())

    def sample_episode(self, n_way, k_support, k_query, device, new_class_bias=1):
        old_cls = [c for c in self.data if c not in self.new_classes]
        new_cls = list(self.new_classes)

        pool = old_cls + new_cls * new_class_bias
        random.shuffle(pool)

        seen, chosen = set(), []
        for c in pool:
            if c not in seen:
                seen.add(c)
                chosen.append(c)
            if len(chosen) == n_way:
                break

        if len(chosen) < n_way:
            rem = [c for c in self.data if c not in seen]
            if len(rem) > 0:
                chosen += random.sample(rem, min(n_way - len(chosen), len(rem)))

        if len(chosen) == 0:
            raise ValueError("EpisodeBuffer is empty.")

        sup_x, sup_y, qry_x, qry_y = [], [], [], []

        for cls in chosen:
            cls_samples = self.data[cls]
            need = k_support + k_query

            if len(cls_samples) >= need:
                picked = random.sample(cls_samples, need)
            else:
                picked = cls_samples + random.choices(cls_samples, k=need - len(cls_samples))

            for s in picked[:k_support]:
                sup_x.append(s)
                sup_y.append(cls)
            for q in picked[k_support:]:
                qry_x.append(q)
                qry_y.append(cls)

        return (
            torch.stack(sup_x).to(device),
            torch.tensor(sup_y, dtype=torch.long).to(device),
            torch.stack(qry_x).to(device),
            torch.tensor(qry_y, dtype=torch.long).to(device)
        )

    def __len__(self):
        return sum(len(v) for v in self.data.values())