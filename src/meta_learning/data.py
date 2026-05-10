import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import BATCH_SIZE

class UCF101Embeddings(Dataset):
    def __init__(self, root, class_to_idx):
        self.samples = []
        for cls_name, idx in class_to_idx.items():
            d = os.path.join(root, cls_name)
            if not os.path.isdir(d): continue
            for f in sorted(os.listdir(d)):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(d, f), idx))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        t = torch.load(path)
        if t.dim() != 2 or t.shape[1] != 2048:
            raise RuntimeError(f"Bad embedding {tuple(t.shape)} at {path}")
        return t, label


def _dl(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)