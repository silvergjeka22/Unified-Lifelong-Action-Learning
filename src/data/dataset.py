#  Run from notebook: %run /content/src/data/dataset.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from glob import glob
from config.config import SELECTED_CLASSES, OUTPUT_ROOT, BATCH_SIZE


class UCF101Clips(Dataset):
    """
    PyTorch Dataset for preprocessed UCF101 clips (.pt tensors).

    Usage
    dataset = UCF101Clips(root_dir, split="train")
    loader  = DataLoader(dataset, batch_size=4, shuffle=True)

    # visualize first clip from a batch
    for clips, labels in loader:
        UCF101Clips.show_frame(clips, labels)
        break
    """

    def __init__(self, root_dir=OUTPUT_ROOT, split="train"):
        self.root    = os.path.join(root_dir, split)
        self.classes = sorted(os.listdir(self.root))
        self.samples = []

        for cls_idx, cls_name in enumerate(self.classes):
            clips = glob(os.path.join(self.root, cls_name, "*.pt"))
            for clip in clips:
                self.samples.append((clip, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = torch.load(path)   # [T, C, H, W]
        return clip, label

    @staticmethod
    def show_frame(clips, labels, clip_idx=0, frame_idx=0):
        """
        Visualize a single frame from a batch of clips.

        Args:
            clips     : tensor [B, T, C, H, W] from the DataLoader
            labels    : tensor [B] from the DataLoader
            clip_idx  : which clip in the batch to show (default 0)
            frame_idx : which frame in the clip to show  (default 0)
        """
        first_clip  = clips[clip_idx]              # [T, C, H, W]
        first_frame = first_clip[frame_idx]        # [C, H, W]

        frame_np = first_frame.permute(1, 2, 0).numpy()  # [H, W, C]
        frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min())

        plt.figure(figsize=(5, 5))
        plt.imshow(frame_np)
        plt.title(f"Class Label: {labels[clip_idx].item()}", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
