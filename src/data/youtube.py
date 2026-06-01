"""
YouTube video downloader and clip extractor.

Provides the same 16-frame clip extraction pipeline as UCF101,
so YouTube clips produce embeddings in a comparable space.

Requirements:
    pip install yt-dlp opencv-python
"""

import os
import subprocess
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms

# Same spatial transform as UCF101 (matches config.py)
YT_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def download_video(url: str, out_path: str, max_height: int = 480) -> bool:
    """
    Download a single YouTube video using yt-dlp.

    Args:
        url        : YouTube URL
        out_path   : full output path including filename (.mp4)
        max_height : cap resolution for speed (default 480p)

    Returns:
        True if successful, False otherwise.
    """
    if os.path.exists(out_path):
        print(f"  [skip] {os.path.basename(out_path)} already exists")
        return True

    fmt    = f"bestvideo[ext=mp4][height<={max_height}]+bestaudio[ext=m4a]/best[ext=mp4]"
    result = subprocess.run(
        ["yt-dlp", "--format", fmt, "--merge-output-format", "mp4",
         "--output", out_path, "--quiet", "--no-warnings", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [!] Download failed: {result.stderr[:200]}")
        return False

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  [OK] {os.path.basename(out_path)}  ({size_mb:.1f} MB)")
    return True


def extract_clips(
    video_path: str,
    clip_len: int = 16,
    n_clips: int = 8,
    transform=None,
) -> list:
    """
    Extract n_clips evenly-spaced clips of clip_len frames from a video.

    Args:
        video_path : path to .mp4 file
        clip_len   : frames per clip (default 16, same as UCF101)
        n_clips    : number of clips to extract per video
        transform  : torchvision transform applied per frame (default YT_TRANSFORM)

    Returns:
        List of (T, C, H, W) tensors, one per clip. Empty list on failure.
    """
    if transform is None:
        transform = YT_TRANSFORM

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < clip_len:
        cap.release()
        print(f"  [!] {video_path}: too short ({total} frames < {clip_len})")
        return []

    starts = np.linspace(0, max(total - clip_len, 0), n_clips, dtype=int)
    clips  = []

    for start in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        frames = []
        for _ in range(clip_len):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(Image.fromarray(frame)))
        if len(frames) == clip_len:
            clips.append(torch.stack(frames))   # (T, C, H, W)

    cap.release()
    return clips


def download_and_extract(
    youtube_clips: dict,
    raw_dir: str,
    class_to_idx: dict,
    clip_len: int = 16,
    n_clips: int = 8,
) -> tuple:
    """
    Download and extract clips for multiple classes.

    Args:
        youtube_clips : {class_name: youtube_url}
        raw_dir       : directory to save downloaded .mp4 files
        class_to_idx  : {class_name: label_index}
        clip_len      : frames per clip
        n_clips       : clips per video

    Returns:
        clips_by_class  : {class_name: list of (T,C,H,W) tensors}
        skipped_classes : list of class names that failed
    """
    os.makedirs(raw_dir, exist_ok=True)
    clips_by_class  = {}
    skipped_classes = []

    for cls_name, url in youtube_clips.items():
        if cls_name not in class_to_idx:
            print(f"  [!] {cls_name} not in class_to_idx — skipping")
            skipped_classes.append(cls_name)
            continue

        out_path = os.path.join(raw_dir, f"{cls_name}.mp4")
        ok       = download_video(url, out_path)
        if not ok:
            skipped_classes.append(cls_name)
            continue

        clips = extract_clips(out_path, clip_len=clip_len, n_clips=n_clips)
        if not clips:
            skipped_classes.append(cls_name)
            continue

        clips_by_class[cls_name] = clips
        print(f"  {cls_name:<22}: {len(clips)} clips  {clips[0].shape}")

    print(f"\nTotal YouTube clips: {sum(len(v) for v in clips_by_class.values())}")
    if skipped_classes:
        print(f"Skipped: {skipped_classes}")

    return clips_by_class, skipped_classes
