import random
import torch

class ReplayBuffer:
    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.data = []

    def add_from_loader(self, loader, max_per_class=8):
        counts = {}
        for x_batch, y_batch in loader:
            for i in range(x_batch.size(0)):
                lbl = y_batch[i].item()
                if counts.get(lbl, 0) >= max_per_class:
                    continue
                self.data.append((x_batch[i].cpu(), lbl))
                counts[lbl] = counts.get(lbl, 0) + 1

        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]

    def sample(self, batch_size):
        if not self.data:
            return None, None
        picked = random.sample(self.data, min(batch_size, len(self.data)))
        xs, ys = zip(*picked)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    def __len__(self):
        return len(self.data)


class EpisodeBuffer:
    def __init__(self):
        self.data = {}           # {label: [tensor, ...]}
        self.new_classes = set()

    def add_from_loader(self, loader, mark_new=False, max_per_class=8):
        counts = {lbl: len(v) for lbl, v in self.data.items()}
        added = set()
        for x_batch, y_batch in loader:
            for i in range(x_batch.size(0)):
                lbl = y_batch[i].item()
                if counts.get(lbl, 0) >= max_per_class:
                    continue
                self.data.setdefault(lbl, []).append(x_batch[i].cpu())
                counts[lbl] = counts.get(lbl, 0) + 1
                added.add(lbl)
        if mark_new:
            self.new_classes.update(added)

    def sample_episode(self, n_way, k_support, k_query, device, new_class_bias=3):
        # bias class selection toward new classes
        old_cls = [c for c in self.data if c not in self.new_classes]
        new_cls = list(self.new_classes & self.data.keys())
        pool = old_cls + new_cls * new_class_bias
        random.shuffle(pool)

        # pick n_way unique classes
        seen, chosen = set(), []
        for c in pool:
            if c not in seen:
                seen.add(c)
                chosen.append(c)
            if len(chosen) == n_way:
                break

        if not chosen:
            raise ValueError("EpisodeBuffer is empty — call add_from_loader before training.")

        # build support / query splits
        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        for cls in chosen:
            samples = self.data[cls]
            need = k_support + k_query
            picked = random.sample(samples, need) if len(samples) >= need \
                     else samples + random.choices(samples, k=need - len(samples))
            sup_x += picked[:k_support];  sup_y += [cls] * k_support
            qry_x += picked[k_support:];  qry_y += [cls] * k_query

        return (
            torch.stack(sup_x).to(device),
            torch.tensor(sup_y, dtype=torch.long).to(device),
            torch.stack(qry_x).to(device),
            torch.tensor(qry_y, dtype=torch.long).to(device),
        )

    def __len__(self):
        return sum(len(v) for v in self.data.values())