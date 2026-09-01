import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import gaussian_blur

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def domain_corrupt(frames, blur_sigma=1.2, brightness=0.75, contrast=0.8, downscale=0.5):
    """
    Simulate a real-world domain shift on ImageNet-normalised frames (N, 3, H, W):
    brightness / contrast change, resolution loss, and blur. Returns re-normalised frames.
    """
    mean, std = _MEAN.to(frames.device), _STD.to(frames.device)
    x = (frames * std + mean).clamp(0, 1)
    x = (((x - 0.5) * contrast + 0.5) * brightness).clamp(0, 1)
    H, W = x.shape[-2:]
    x = F.interpolate(x, scale_factor=downscale, mode="bilinear", align_corners=False)
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    k = max(3, int(2 * round(blur_sigma) + 1))
    x = gaussian_blur(x, kernel_size=[k, k], sigma=[blur_sigma, blur_sigma])
    return (x.clamp(0, 1) - mean) / std


class ShiftedClips(Dataset):
    """Wraps a clip dataset and returns each clip domain-corrupted - the shifted domain."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        clip, label = self.base[idx]
        return domain_corrupt(clip), label
