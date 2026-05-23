import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import config as cfg

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
        return t, label


def _dl(ds, shuffle=False):
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=2)