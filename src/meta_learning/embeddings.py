import copy
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from src.meta_learning.models import fresh_model


class ResNet50Trunk(nn.Module):
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
            src  = os.path.join(dirpath, fname)
            clip = torch.load(src)
            with torch.no_grad():
                emb = trunk(clip.to(device))
            assert emb.dim() == 2 and emb.shape[1] == 2048
            torch.save(emb.cpu(), os.path.join(odir, fname))