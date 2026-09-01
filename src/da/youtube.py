import os
import random
import subprocess
import torch
from PIL import Image

import src.config.config as cfg
from src.data.preprocessing import extract_frames


def _download(url, out_path):
    """Fetch one video with yt-dlp. Works for archive.org direct URLs and YouTube."""
    subprocess.run(["yt-dlp", "-q", "-o", out_path, url], check=True)


def _clips_from_video(video_path, n_clips, clip_len):
    """Cut n_clips fixed-length clips from one video and apply the spatial transform."""
    frames = extract_frames(video_path)
    clips = []
    for _ in range(n_clips):
        if len(frames) > clip_len:
            start = random.randint(0, len(frames) - clip_len)
            window = frames[start:start + clip_len]
        else:
            window = frames + [frames[-1]] * (clip_len - len(frames))
        tensor = torch.stack([cfg.spatial_transform(Image.fromarray(f)) for f in window])
        clips.append(tensor)
    return clips


def download_youtube_clips(youtube_clips, class_to_idx, n_clips=40, raw_dir=None, clip_len=None):
    """
    Download each {class: url} video and cut it into n_clips clips. Returns
    (clips, labels) as tensors, ready to run through the model. Real out-of-domain data.
    """
    raw_dir  = raw_dir or cfg.YT_RAW_DIR
    clip_len = clip_len or cfg.CLIP_LEN
    os.makedirs(raw_dir, exist_ok=True)

    all_clips, all_labels = [], []
    for cls, url in youtube_clips.items():
        if cls not in class_to_idx:
            continue
        path = os.path.join(raw_dir, f"{cls}.mp4")
        if not os.path.exists(path):
            _download(url, path)
        clips = _clips_from_video(path, n_clips, clip_len)
        all_clips.extend(clips)
        all_labels.extend([class_to_idx[cls]] * len(clips))
        print(f"  {cls:<20}: {len(clips)} clips")

    return torch.stack(all_clips), torch.tensor(all_labels)
