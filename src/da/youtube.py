import os
import random
import subprocess
import torch
from PIL import Image

import src.config.config as cfg
from src.data.preprocessing import extract_frames


def _download(url, out_path):
    """Fetch one video with yt-dlp. Returns True on success, False if the URL fails (dead link).
    Removes any partial file on failure so a re-run starts clean."""
    ok = subprocess.run(["yt-dlp", "-q", "-o", out_path, url]).returncode == 0
    if not ok and os.path.exists(out_path):
        os.remove(out_path)
    return ok


def _clips_from_video(video_path, n_clips, clip_len, max_frames=None):
    """Cut n_clips fixed-length clips from one video and apply the spatial transform.
    Only the first max_frames frames are read, so a long video does not blow up RAM."""
    frames = extract_frames(video_path, max_frames=max_frames)
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


def download_youtube_clips(youtube_clips, class_to_idx, n_clips=40, raw_dir=None, clip_len=None,
                           max_frames=None):
    """
    Download each {class: url} video and cut it into n_clips clips. Returns
    (clips, labels) as tensors, ready to run through the model. Real out-of-domain data.

    max_frames caps frames read per video (defaults to cfg.YT_MAX_FRAMES) so a long clip
    does not exhaust RAM and crash the kernel.
    """
    raw_dir    = raw_dir or cfg.YT_RAW_DIR
    clip_len   = clip_len or cfg.CLIP_LEN
    max_frames = max_frames or cfg.YT_MAX_FRAMES
    os.makedirs(raw_dir, exist_ok=True)

    all_clips, all_labels = [], []
    for cls, url in youtube_clips.items():
        if cls not in class_to_idx:
            continue
        path = os.path.join(raw_dir, f"{cls}.mp4")
        # a dead URL is skipped, not fatal - the task just uses the classes that download
        if not os.path.exists(path) and not _download(url, path):
            print(f"  {cls:<20}: download failed - skipping")
            continue
        clips = _clips_from_video(path, n_clips, clip_len, max_frames)
        all_clips.extend(clips)
        all_labels.extend([class_to_idx[cls]] * len(clips))
        print(f"  {cls:<20}: {len(clips)} clips")

    return torch.stack(all_clips), torch.tensor(all_labels)
