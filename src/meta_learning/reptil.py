import os
import copy
import random
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def check_backbone(model_path, data_root, class_list, num_classes, device,
                   batch_size=8, num_workers=2, task_label="Task 0"):

    model = ResNet50LSTM(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(
        UCF101Clips(f"{data_root}/test", class_to_idx={c: i for i, c in enumerate(class_list)}),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb   = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += yb.size(0)

    acc         = correct / total if total else 0.0
    lstm_hidden = torch.load(model_path, map_location="cpu")["lstm.weight_hh_l0"].shape[1]

    print(f"  [{task_label}] Accuracy: {acc:.4f} ({correct}/{total}) | LSTM hidden: {lstm_hidden}")
    return model, acc, lstm_hidden


class _ResNet50Trunk(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.resnet    = copy.deepcopy(full_model.resnet)
        self.resnet.fc = nn.Flatten(start_dim=1)

    def forward(self, frames):
        return self.resnet(frames)


def extract_embeddings(clip_root, emb_root, trunk, device):
    trunk.eval()
    for dirpath, _, filenames in os.walk(clip_root):
        pt_files = sorted(f for f in filenames if f.endswith(".pt"))
        if not pt_files:
            continue
        rel  = os.path.relpath(dirpath, clip_root)
        odir = os.path.join(emb_root, rel)
        os.makedirs(odir, exist_ok=True)
        for fname in tqdm(pt_files, desc=f"  emb {rel}", leave=False):
            clip = torch.load(os.path.join(dirpath, fname))
            with torch.no_grad():
                emb = trunk(clip.to(device))
            assert emb.dim() == 2 and emb.shape[1] == 2048
            torch.save(emb.cpu(), os.path.join(odir, fname))


class EmbeddingLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=256,
                 num_layers=1, num_classes=16, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.drop(h_n[-1]))


class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.data     = []

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
                if max_per_class and counts.get(lbl, 0) >= max_per_class:
                    continue
                self.data.append((x[i].cpu(), lbl))
                counts[lbl] = counts.get(lbl, 0) + 1
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]

    def sample(self, batch_size):
        if not self.data:
            return None, None
        picked = random.sample(self.data, min(batch_size, len(self.data)))
        x, y   = zip(*picked)
        return torch.stack(x), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.data)


def fresh_replay_buffer(exemplar_train_loader, limit):
    buf = ReplayBuffer(max_size=1000)
    buf.add_from_loader(exemplar_train_loader, max_per_class=limit)
    return buf


def fresh_replay_buffer_t2(exemplar_train_loader, task1_train_loader, limit):
    buf = ReplayBuffer(max_size=1000)
    buf.add_from_loader(exemplar_train_loader, max_per_class=limit)
    buf.add_from_loader(task1_train_loader,    max_per_class=limit)
    return buf


class EpisodeBuffer:
    def __init__(self):
        self.data        = {}
        self.new_classes = set()

    def add_batch(self, x, y):
        for i in range(len(x)):
            lbl = y[i].item() if torch.is_tensor(y[i]) else int(y[i])
            self.data.setdefault(lbl, []).append(x[i].cpu())

    def add_from_loader(self, loader, mark_new=False):
        added = set()
        for x, y in loader:
            self.add_batch(x, y)
            for lbl in y.tolist():
                added.add(lbl)
        if mark_new:
            self.new_classes.update(added)

    def available_classes(self):
        return list(self.data.keys())

    def sample_episode(self, n_way, k_support, k_query, device, new_class_bias=1):
        old_cls = [c for c in self.data if c not in self.new_classes]
        new_cls = list(self.new_classes)
        pool    = old_cls + new_cls * new_class_bias
        random.shuffle(pool)

        seen, chosen = set(), []
        for c in pool:
            if c not in seen:
                seen.add(c)
                chosen.append(c)
            if len(chosen) == n_way:
                break

        if len(chosen) < n_way:
            rem     = [c for c in self.data if c not in seen]
            chosen += random.sample(rem, min(n_way - len(chosen), len(rem)))

        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        for cls in chosen:
            pool_s = self.data[cls]
            need   = k_support + k_query
            picked = random.sample(pool_s, min(need, len(pool_s)))
            if len(picked) < need:
                picked = picked + random.choices(pool_s, k=need - len(picked))
            for c in picked[:k_support]:
                sup_x.append(c)
                sup_y.append(cls)
            for c in picked[k_support:]:
                qry_x.append(c)
                qry_y.append(cls)

        return (
            torch.stack(sup_x).to(device),
            torch.tensor(sup_y, dtype=torch.long).to(device),
            torch.stack(qry_x).to(device),
            torch.tensor(qry_y, dtype=torch.long).to(device)
        )

    def __len__(self):
        return sum(len(v) for v in self.data.values())