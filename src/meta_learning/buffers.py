import random
import torch

# update if more new calsses
REP_N_WAY = 3
LIMIT = 8

class ReplayBuffer:
    def __init__(self, max_size=2000):
        self.max_size   = max_size
        self.data       = []            # flat list
        self._by_class  = {}           # balanced sampling

    def add_from_loader(self, loader, max_per_class=LIMIT):
        counts = {lbl: len(items) for lbl, items in self._by_class.items()}

        for x_batch, y_batch in loader:
            for i in range(x_batch.size(0)):
                lbl = y_batch[i].item()
                if counts.get(lbl, 0) >= max_per_class:
                    continue
                tensor = x_batch[i].detach().cpu()
                self.data.append((tensor, lbl))
                self._by_class.setdefault(lbl, []).append(tensor)
                counts[lbl] = counts.get(lbl, 0) + 1

        self._trim()

    def add_batch(self, x, y, max_per_class=LIMIT):
        counts = {lbl: len(items) for lbl, items in self._by_class.items()}

        for i in range(len(x)):
            lbl = y[i].item() if torch.is_tensor(y[i]) else int(y[i])
            if counts.get(lbl, 0) >= max_per_class:
                continue
            tensor = x[i].detach().cpu()
            self.data.append((tensor, lbl))
            self._by_class.setdefault(lbl, []).append(tensor)
            counts[lbl] = counts.get(lbl, 0) + 1

        self._trim()

    def sample(self, batch_size, balanced=True):
        if not self.data or batch_size <= 0:
            return None, None

        if not balanced:
            picked = random.sample(self.data, min(batch_size, len(self.data)))
        else:
            classes    = list(self._by_class.keys())
            per_class  = max(1, batch_size // len(classes))
            picked     = []

            for cls in classes:
                items = self._by_class[cls]
                take  = min(per_class, len(items))
                chosen = (random.sample(items, take)
                          if len(items) >= take
                          else random.choices(items, k=take))
                picked.extend([(t, cls) for t in chosen])

            if len(picked) < batch_size:
                extra = random.sample(self.data,
                                      min(batch_size - len(picked), len(self.data)))
                picked.extend(extra)

            picked = picked[:batch_size]

        xs, ys = zip(*picked)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    def available_classes(self):
        return list(self._by_class.keys())

    def _trim(self):
        if len(self.data) > self.max_size:
            drop = len(self.data) - self.max_size
            removed = self.data[:drop]
            self.data = self.data[drop:]
            for tensor, lbl in removed:
                if lbl in self._by_class and tensor in self._by_class[lbl]:
                    self._by_class[lbl].remove(tensor)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        counts = {lbl: len(v) for lbl, v in self._by_class.items()}
        return f"ReplayBuffer(total={len(self.data)}, classes={counts})"

class EpisodeBuffer:
    def __init__(self):
        self.data           = {}        # {label: [tensor, ...]}
        self.new_classes    = set()     # labels that arrived in the current task
        self._live_counts   = {}        # per-epoch live-ingestion quota

    def add_from_loader(self, loader, mark_new=False, max_per_class=LIMIT):
        counts = {lbl: len(v) for lbl, v in self.data.items()}
        added  = set()

        for x_batch, y_batch in loader:
            for i in range(x_batch.size(0)):
                lbl = y_batch[i].item()
                if counts.get(lbl, 0) >= max_per_class:
                    continue
                self.data.setdefault(lbl, []).append(x_batch[i].detach().cpu())
                counts[lbl] = counts.get(lbl, 0) + 1
                added.add(lbl)

        if mark_new:
            self.new_classes.update(added)

    def add_batch(self, x, y, max_per_class=LIMIT):
        for i in range(x.size(0)):
            lbl = y[i].item() if torch.is_tensor(y[i]) else int(y[i])
            if self._live_counts.get(lbl, 0) >= max_per_class:
                continue
            self.data.setdefault(lbl, []).append(x[i].detach().cpu())
            self._live_counts[lbl] = self._live_counts.get(lbl, 0) + 1

    def reset_live_counts(self):
        self._live_counts = {}

    def sample_episode(self, n_way, k_support, k_query, device,
                       new_class_bias=3):
        old_cls = [c for c in self.data if c not in self.new_classes]
        new_cls = list(self.new_classes & self.data.keys())

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
            remaining = [c for c in self.data if c not in seen]
            chosen   += random.sample(remaining,
                                      min(n_way - len(chosen), len(remaining)))

        if not chosen:
            raise ValueError(
                "EpisodeBuffer is empty — call add_from_loader before training."
            )

        sup_x, sup_y, qry_x, qry_y = [], [], [], []

        for cls in chosen:
            samples = self.data[cls]
            need    = k_support + k_query

            if len(samples) >= need:
                picked = random.sample(samples, need)
            else:
                picked = samples + random.choices(samples, k=need - len(samples))

            for s in picked[:k_support]:
                sup_x.append(s)
                sup_y.append(cls)
            for q in picked[k_support:]:
                qry_x.append(q)
                qry_y.append(cls)

        return (
            torch.stack(sup_x).to(device),
            torch.tensor(sup_y,  dtype=torch.long).to(device),
            torch.stack(qry_x).to(device),
            torch.tensor(qry_y,  dtype=torch.long).to(device),
        )

    def available_classes(self):
        return list(self.data.keys())

    def __len__(self):
        return sum(len(v) for v in self.data.values())

    def __repr__(self):
        counts = {lbl: len(v) for lbl, v in self.data.items()}
        new    = list(self.new_classes)
        return f"EpisodeBuffer(total={len(self)}, new_classes={new}, per_class={counts})"

def build_task1_buffers(exemplar_train_loader, device=None):
    replay_t1       = ReplayBuffer(max_size=2000)
    replay_t1_nokd  = ReplayBuffer(max_size=2000)

    for buf in (replay_t1, replay_t1_nokd):
        buf.add_from_loader(exemplar_train_loader, max_per_class=LIMIT)

    episode_t1       = EpisodeBuffer()
    episode_t1_nokd  = EpisodeBuffer()

    for buf in (episode_t1, episode_t1_nokd):
        buf.add_from_loader(exemplar_train_loader,
                            mark_new=False,
                            max_per_class=LIMIT)

    n_way_eff_t1 = min(REP_N_WAY, len(episode_t1.available_classes()))

    print(f"[T1 buffers] {replay_t1}")
    print(f"[T1 episode] {episode_t1}")
    print(f"[T1] Effective n_way = {n_way_eff_t1}")

    return replay_t1, replay_t1_nokd, episode_t1, episode_t1_nokd, n_way_eff_t1


def build_task2_buffers(exemplar_train_loader, task1_train_loader, device=None):
    replay_t2       = ReplayBuffer(max_size=2000)
    replay_t2_nokd  = ReplayBuffer(max_size=2000)

    for buf in (replay_t2, replay_t2_nokd):
        buf.add_from_loader(exemplar_train_loader, max_per_class=LIMIT)
        buf.add_from_loader(task1_train_loader,    max_per_class=LIMIT)

    episode_t2       = EpisodeBuffer()
    episode_t2_nokd  = EpisodeBuffer()

    for buf in (episode_t2, episode_t2_nokd):
        buf.add_from_loader(exemplar_train_loader,
                            mark_new=False,
                            max_per_class=LIMIT)
        buf.add_from_loader(task1_train_loader,
                            mark_new=True,       # important
                            max_per_class=LIMIT)

    n_way_eff_t2 = min(REP_N_WAY, len(episode_t2.available_classes()))

    print(f"[T2 buffers] {replay_t2}")
    print(f"[T2 episode] {episode_t2}")
    print(f"[T2] Effective n_way = {n_way_eff_t2}")

    return replay_t2, replay_t2_nokd, episode_t2, episode_t2_nokd, n_way_eff_t2