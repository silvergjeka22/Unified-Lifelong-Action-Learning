import os
import subprocess
import torch
from PIL import Image

import src.config.config as cfg
from src.data.preprocessing import sample_spread_clips


def _download(url, out_path):
    """Download a video as an H.264 MP4"""
    ok = subprocess.run(["yt-dlp", "-q", "-S", "res:480", "--recode-video", "mp4",
                         "-o", out_path, url]).returncode == 0
    if not ok and os.path.exists(out_path):
        os.remove(out_path)
    return ok


def _clips_from_video(video_path, n_clips, clip_len):
    """Create clips spread across the video and apply the spatial transform"""
    clips = []
    for window in sample_spread_clips(video_path, n_clips, clip_len):
        clips.append(torch.stack([cfg.spatial_transform(Image.fromarray(f)) for f in window]))
    return clips


def download_youtube_clips(youtube_clips, class_to_idx, clips_per_video=20, raw_dir=None,
                           clip_len=None):
    """
    Download videos, split them into clips, and return the clips and labels

    ``youtube_clips`` maps class names to URLs. A single URL is also accepted.
    Invalid URLs are skipped, and clips are sampled across each video.
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
                path = src                       # a local video file use it directly (most reliable)
            else:
                path = os.path.join(raw_dir, f"{cls}_{j}.mp4")
                # a dead URL is skipped, not fatal the class just gets fewer videos
                if not os.path.exists(path) and not _download(src, path):
                    print(f"  {cls} video {j}: download failed - skipping")
                    continue
            clips = _clips_from_video(path, clips_per_video, clip_len)
            if not clips:
                print(f"  {cls} video {j}: 0 clips (could not decode) - skipping")
                continue
            all_clips.extend(clips)
            all_labels.extend([class_to_idx[cls]] * len(clips))
            all_videos.extend([video_id] * len(clips))
            video_id  += 1
            n_videos  += 1
            print(f"  {cls} video {j}: {len(clips)} clips")
        print(f"  {cls:<16}: {n_videos} usable video(s)")

    if not all_clips:
        raise RuntimeError("No readable clips the downloaded videos could not be decoded "
                           "Clear the raw dir and re run so they re download as H.264 mp4")
    return torch.stack(all_clips), torch.tensor(all_labels), torch.tensor(all_videos)


def video_disjoint_split(clips, labels, video_ids, holdout_per_class=1):
    """Split clips by video, keeping the last videos of each class for testing"""
    adapt_idx, test_idx = [], []
    for c in labels.unique():
        c_pos  = (labels == c).nonzero(as_tuple=True)[0]
        c_vids = video_ids[c_pos].unique().tolist()
        held   = set(c_vids[-holdout_per_class:])
        for i in c_pos.tolist():
            if video_ids[i].item() in held:
                test_idx.append(i)
            else:
                adapt_idx.append(i)
    adapt_idx = torch.tensor(adapt_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx, dtype=torch.long)
    return clips[adapt_idx], labels[adapt_idx], clips[test_idx], labels[test_idx]
