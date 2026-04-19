import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from data.dataset import ClipDataset
from config.config import (
    SELECTED_CLASSES, TASK_1, TASK_2, TASK_3, TASK_4,
    CLASSES_20, CLASSES_30, CLASSES_40, CLASSES_50,
    BATCH_SIZE,
    BASE_ROOT, TASK1_ROOT, TASK2_ROOT, TASK3_ROOT, TASK4_ROOT,
)


def make_loader(root, classes, split, label_offset=0, shuffle=None):
    """Build a DataLoader for one (root, classes, split) combination."""
    ds = ClipDataset(
        root=os.path.join(root, split),
        classes=classes,
        label_offset=label_offset,
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=2,
        pin_memory=True,
    )