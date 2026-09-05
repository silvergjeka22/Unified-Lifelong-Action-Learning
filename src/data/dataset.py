# Run from notebook: %run /content/src/data/dataset.py

import os
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, ConcatDataset
from glob import glob


class UCF101Clips(Dataset):
    def __init__(self, root_dir, class_to_idx):
        self.root = root_dir
        self.class_to_idx = class_to_idx # The Global Phonebook
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


def dataset_labels(dataset):
    """A list of labels aligned to dataset indices, read without loading any clip tensor."""
    if hasattr(dataset, "samples"):
        return [label for _, label in dataset.samples]
    if isinstance(dataset, ConcatDataset):
        labels = []
        for d in dataset.datasets:
            labels.extend(dataset_labels(d))
        return labels
    return [dataset[i][1] for i in range(len(dataset))]


def few_shot_subset(dataset, k, seed=42):
    """A Subset of `dataset` with at most k clips per class, chosen deterministically."""
    labels = dataset_labels(dataset)
    by_class = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)

    rng = random.Random(seed)
    keep = []
    for label in sorted(by_class):
        idxs = list(by_class[label])
        rng.shuffle(idxs)
        keep.extend(idxs[:k])
    keep.sort()
    return Subset(dataset, keep)