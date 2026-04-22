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


def preprocess_dataset(
    target_classes=None,
    splits=None,
    input_root=None,
    output_root=None,
    max_samples=None
):
    """
    Unified preprocessing for UCF101. 
    Defaults to global cfg values but allows flexible overrides.
    """
    # 1. Setup Defaults from cfg
    target_classes = target_classes or cfg.SELECTED_CLASSES
    splits         = splits or ["train", "val", "test"]
    input_root     = input_root or cfg.DATASET_ROOT
    output_root    = output_root or cfg.OUTPUT_ROOT

    print(f"\n--- Preprocessing UCF101 ---")
    print(f"From: {input_root}")
    print(f"To  : {output_root}")
    print(f"Classes: {len(target_classes)} | Limit: {max_samples if max_samples else 'Full'}")

    os.makedirs(output_root, exist_ok=True)

    for split in splits:
        split_input_path  = os.path.join(input_root, split)
        split_output_path = os.path.join(output_root, split)
        os.makedirs(split_output_path, exist_ok=True)

        print(f"\nProcessing split: {split}")

        for cls in target_classes:
            class_input_path  = os.path.join(split_input_path, cls)
            class_output_path = os.path.join(split_output_path, cls)

            if not os.path.isdir(class_input_path):
                print(f"  Warning: Class '{cls}' not found in {split}, skipping.")
                continue

            os.makedirs(class_output_path, exist_ok=True)

            # Get and sort video files
            videos = [v for v in os.listdir(class_input_path) if v.endswith(".avi")]
            videos.sort()

            # Apply memory saver limit (typically for training sets)
            if max_samples is not None and split == "train":
                videos = videos[:max_samples]

            for vid_name in tqdm(videos, desc=f"  {cls} [{split}]", leave=False):
                vid_path = os.path.join(class_input_path, vid_name)
                output_filename = os.path.splitext(vid_name)[0] + ".pt"
                output_path = os.path.join(class_output_path, output_filename)

                # Skip if already exists (Resume capability)
                if os.path.exists(output_path):
                    continue

                try:
                    # Processing Pipeline
                    frames = extract_frames(vid_path)
                    if not frames:
                        continue

                    sampled_frames = temporal_sample(frames)
                    save_clip_tensor(sampled_frames, output_path)

                except Exception as e:
                    print(f"\n  Error processing {vid_name}: {e}")

    print(f"\nDone! Dataset ready at: {output_root}")