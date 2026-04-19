# Run from notebook: %run /content/src/data/preprocessing.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2, torch, random
from PIL import Image
from tqdm import tqdm
import config.config as cfg


def extract_frames(video_path, target_fps=cfg.FRAME_RATE):
    """Extract frames from a video at target fps. Returns list of RGB arrays."""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_interval = max(int(original_fps // target_fps), 1)
    frames, count = [], 0
    success, frame = cap.read()
    while success:
        if count % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, frame = cap.read()
        count += 1
    cap.release()
    return frames


def temporal_sample(frames, clip_len=cfg.CLIP_LEN):
    """Sample a fixed-length clip; pads with last frame if too short."""
    if len(frames) >= clip_len:
        start = random.randint(0, len(frames) - clip_len)
        return frames[start : start + clip_len]
    last = frames[-1]
    while len(frames) < clip_len:
        frames.append(last)
    return frames


def save_clip_tensor(frames, output_path):
    """Apply spatial transforms and save clip as [T, C, H, W] .pt tensor."""
    processed = [cfg.spatial_transform(Image.fromarray(f)) for f in frames]
    torch.save(torch.stack(processed, dim=0), output_path)


def preprocess_dataset(splits=None):
    """
    Preprocess UCF101 using the current cfg.SELECTED_CLASSES and cfg.OUTPUT_ROOT.

    Args:
        splits: list of splits to process. Defaults to ["train", "val", "test"].

    Examples:
        preprocess_dataset(splits=["test"])
        preprocess_dataset()                # all three splits
    """
    if splits is None:
        splits = ["train", "val", "test"]

    print(f"Preprocessing UCF101\n  from : {cfg.DATASET_ROOT}\n  to   : {cfg.OUTPUT_ROOT}\n  splits: {splits}\n")
    os.makedirs(cfg.OUTPUT_ROOT, exist_ok=True)

    for split in splits:
        split_in  = os.path.join(cfg.DATASET_ROOT, split)
        split_out = os.path.join(cfg.OUTPUT_ROOT, split)
        os.makedirs(split_out, exist_ok=True)
        print(f"-> {split}")

        for cls in cfg.SELECTED_CLASSES:
            cls_in  = os.path.join(split_in, cls)
            cls_out = os.path.join(split_out, cls)
            if not os.path.isdir(cls_in):
                print(f"   Warning: '{cls}' not found in '{split}', skipping.")
                continue
            os.makedirs(cls_out, exist_ok=True)

            videos = sorted(
                os.path.join(cls_in, v)
                for v in os.listdir(cls_in) if v.endswith(".avi")
            )
            for vid_path in tqdm(videos, desc=f"  {cls} [{split}]"):
                frames = extract_frames(vid_path)
                if not frames:
                    print(f"   Skipping empty video: {vid_path}")
                    continue
                vid_name = os.path.splitext(os.path.basename(vid_path))[0]
                save_clip_tensor(temporal_sample(frames), os.path.join(cls_out, f"{vid_name}.pt"))

    print("\nDone! Saved to:", cfg.OUTPUT_ROOT)