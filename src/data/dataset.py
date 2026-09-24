import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset
from glob import glob


def per_class_subset(dataset, k):
    """The first k clips of every class of a UCF101Clips dataset, as a Subset (nothing is loaded)"""
    counts, keep = {}, []
    for i, (_, label) in enumerate(dataset.samples):
        if counts.get(label, 0) < k:
            keep.append(i)
            counts[label] = counts.get(label, 0) + 1
    return Subset(dataset, keep)


class UCF101Clips(Dataset):
    def __init__(self, root_dir, class_to_idx):
        self.root = root_dir
        self.class_to_idx = class_to_idx # The Global index
        self.classes = sorted(os.listdir(self.root))
        self.samples = []

        for cls_name in self.classes:
            # Look up the GLOBAL ID instead of using a local loop index
            if cls_name in self.class_to_idx:
                cls_idx = self.class_to_idx[cls_name]

                clips = glob(os.path.join(self.root, cls_name, "*.pt"))
                for clip in clips:
                    self.samples.append((clip, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = torch.load(path)
        return clip, label

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