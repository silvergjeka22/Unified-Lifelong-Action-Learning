import os
import subprocess
import torch
from PIL import Image

import src.config.config as cfg
from src.data.preprocessing import sample_spread_clips


def _download(url, out_path):
    """Fetch one video with yt-dlp (archive.org, YouTube, ...). Prefers a small ~480p version for
    speed. Returns True on success, False on failure (removes any partial file). Note: YouTube on
    cloud IPs (Colab) is often bot-blocked - archive.org or a local file is more reliable."""
    ok = subprocess.run(["yt-dlp", "-q", "-S", "res:480", "-o", out_path, url]).returncode == 0
    if not ok and os.path.exists(out_path):
        os.remove(out_path)
    return ok


def _clips_from_video(video_path, n_clips, clip_len):
    """Cut n_clips short clips whose START positions are spread across the WHOLE video (so a long
    video's action is not missed), and apply the spatial transform. RAM-safe (seeks per clip)."""
    clips = []
    for window in sample_spread_clips(video_path, n_clips, clip_len):
        clips.append(torch.stack([cfg.spatial_transform(Image.fromarray(f)) for f in window]))
    return clips


def download_youtube_clips(youtube_clips, class_to_idx, clips_per_video=20, raw_dir=None,
                           clip_len=None):
    """
    Download EVERY video for each class and cut clips_per_video clips from each. Returns
    (clips, labels, video_ids): video_ids tags which SOURCE video each clip came from, so a
    video-disjoint few-shot split can hold out whole videos. Real out-of-domain data.

    youtube_clips: {class_name: [url, ...]} (a single string is also accepted). A dead URL is
    skipped (the class just gets fewer videos). Clip start positions are spread across each
    whole video (sample_spread_clips), so the action is captured even in long videos.
    """
    raw_dir  = raw_dir or cfg.YT_RAW_DIR
    clip_len = clip_len or cfg.CLIP_LEN
    os.makedirs(raw_dir, exist_ok=True)

    all_clips, all_labels, all_videos = [], [], []
    video_id = 0
    for cls, urls in youtube_clips.items():
        if cls not in class_to_idx:
            continue
        if isinstance(urls, str):
            urls = [urls]
        n_videos = 0
        for j, src in enumerate(urls):
            if os.path.exists(src):
                path = src                       # a local video file - use it directly (most reliable)
            else:
                path = os.path.join(raw_dir, f"{cls}_{j}.mp4")
                # a dead URL is skipped, not fatal - the class just gets fewer videos
                if not os.path.exists(path) and not _download(src, path):
                    print(f"  {cls} video {j}: download failed - skipping")
                    continue
            clips = _clips_from_video(path, clips_per_video, clip_len)
            all_clips.extend(clips)
            all_labels.extend([class_to_idx[cls]] * len(clips))
            all_videos.extend([video_id] * len(clips))
            video_id  += 1
            n_videos  += 1
            print(f"  {cls} video {j}: {len(clips)} clips")
        print(f"  {cls:<16}: {n_videos} video(s)")

    return torch.stack(all_clips), torch.tensor(all_labels), torch.tensor(all_videos)
