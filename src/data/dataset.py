# Run from notebook: %run /content/src/data/dataset.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from glob import glob


class ClipDataset(Dataset):
    """
    PyTorch Dataset for preprocessed UCF101 clips (.pt tensors).

    Args:
        root         : path to a split folder, e.g. processed_data_base/test
        classes      : explicit list of class names to load (preserves global label order)
        label_offset : add this integer to every label (used for continual-learning tasks)

    Usage:
        ds     = ClipDataset(root="./UCF101/processed_data_base/test", classes=SELECTED_CLASSES)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        for clips, labels in loader:
            ClipDataset.show_frame(clips, labels)
            break
    """

    def __init__(self, root: str, classes: list, label_offset: int = 0):
        self.root         = root
        self.label_offset = label_offset
        self.classes      = sorted(classes)          # deterministic order
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples      = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for clip_path in glob(os.path.join(cls_dir, "*.pt")):
                self.samples.append((clip_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = torch.load(path)          # [T, C, H, W]
        return clip, label + self.label_offset

    @staticmethod
    def show_frame(clips, labels, clip_idx=0, frame_idx=0):
        """Visualize one frame from a DataLoader batch."""
        frame = clips[clip_idx][frame_idx].permute(1, 2, 0).numpy()
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        plt.figure(figsize=(5, 5))
        plt.imshow(frame)
        plt.title(f"Class Label: {labels[clip_idx].item()}", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.show()