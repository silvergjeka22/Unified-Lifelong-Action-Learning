#  PREPROCESSING
#  Run from notebook: %run /content/src/data/preprocessing.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2, torch, shutil, random
from PIL import Image
from tqdm import tqdm
from config.config import (
    DATASET_ROOT, OUTPUT_ROOT, SELECTED_CLASSES,
    FRAME_RATE, CLIP_LEN, spatial_transform
)


def extract_frames(video_path, target_fps=FRAME_RATE):
    """Extract frames from a video at target fps. Returns list of RGB arrays."""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = target_fps

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


def temporal_sample(frames, clip_len=CLIP_LEN):
    """Sample a fixed-length clip; pads with last frame if video is too short."""
    num_frames = len(frames)
    if num_frames >= clip_len:
        start = random.randint(0, num_frames - clip_len)
        return frames[start:start + clip_len]
    last_frame = frames[-1]
    while len(frames) < clip_len:
        frames.append(last_frame)
    return frames


def save_clip_tensor(frames, output_path):
    """Apply spatial transforms and save clip as a .pt tensor [T, C, H, W]."""
    processed = [spatial_transform(Image.fromarray(f)) for f in frames]
    torch.save(torch.stack(processed, dim=0), output_path)


def preprocess_dataset(splits=None):
    """
    Preprocess UCF101 dataset.

    Args:
        splits: list of splits to process. Default: ["train", "val", "test"]
                Examples:
                  preprocess_dataset()                        # all splits
                  preprocess_dataset(splits=["train"])        # train only
                  preprocess_dataset(splits=["train", "val"]) # train + val
    """
    if splits is None:
        splits = ["train", "val", "test"]

    print(f"Preprocessing UCF101\n  from   : {DATASET_ROOT}\n  to     : {OUTPUT_ROOT}\n  splits : {splits}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for split_name in splits:
        print(f"-> Split: {split_name}")
        split_in  = os.path.join(DATASET_ROOT, split_name)
        split_out = os.path.join(OUTPUT_ROOT,  split_name)
        os.makedirs(split_out, exist_ok=True)

        for cls in SELECTED_CLASSES:
            cls_in  = os.path.join(split_in,  cls)
            cls_out = os.path.join(split_out, cls)

            if not os.path.isdir(cls_in):
                print(f"   Warning: '{cls}' not found in '{split_name}', skipping.")
                continue

            os.makedirs(cls_out, exist_ok=True)
            videos = sorted(
                os.path.join(cls_in, v)
                for v in os.listdir(cls_in)
                if v.endswith(".avi")
            )

            for vid_path in tqdm(videos, desc=f"  {cls} [{split_name}]"):
                frames = extract_frames(vid_path)
                if not frames:
                    print(f"   Skipping empty video: {vid_path}")
                    continue

                sampled  = temporal_sample(frames)
                vid_name = os.path.splitext(os.path.basename(vid_path))[0]
                save_clip_tensor(sampled, os.path.join(cls_out, f"{vid_name}.pt"))

    print("\nDone! Processed data saved at:", OUTPUT_ROOT)
