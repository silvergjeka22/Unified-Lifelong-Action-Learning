import os
import cv2, torch, random
from PIL import Image
from tqdm import tqdm
import src.config.config as cfg


def extract_frames(video_path, target_fps=cfg.FRAME_RATE, max_frames=None):
    """Read a video at the target fps and return its RGB frames (at most max_frames)."""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_interval = max(int(original_fps // target_fps), 1)
    frames, count = [], 0
    success, frame = cap.read()
    while success:
        if count % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames is not None and len(frames) >= max_frames:
                break
        success, frame = cap.read()
        count += 1
    cap.release()
    return frames


def temporal_sample(frames, clip_len=cfg.CLIP_LEN):
    """Sample a fixed- ength clip pads with last frame if too short"""
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


def sample_spread_clips(video_path, n_clips, clip_len=cfg.CLIP_LEN, target_fps=cfg.FRAME_RATE):
    """Return n_clips clips spread over the video, reading one clip at a time."""
    cap = cv2.VideoCapture(video_path)
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step         = max(int(original_fps // target_fps), 1)     # about target_fps frames per second
    last_start   = max(total - clip_len * step, 0)

    windows = []
    for i in range(n_clips):
        start = int(last_start * i / max(n_clips - 1, 1)) if last_start > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        window = []
        while len(window) < clip_len:
            ok, frame = cap.read()
            if not ok:
                break
            window.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for _ in range(step - 1):
                cap.read()
        if not window:
            continue
        while len(window) < clip_len:
            window.append(window[-1])
        windows.append(window)

    cap.release()
    return windows


def check_group_leakage(target_classes=None, input_root=None, splits=None):
    """Find UCF101 video groups that occur in more than one split"""
    target_classes = target_classes or cfg.SELECTED_CLASSES
    splits         = splits or ["train", "val", "test"]
    input_root     = input_root or cfg.DATASET_ROOT

    groups = {}
    n_videos = 0
    for split in splits:
        for cls in target_classes:
            class_path = os.path.join(input_root, split, cls)
            if not os.path.isdir(class_path):
                continue
            for vid in os.listdir(class_path):
                if not vid.endswith(".avi"):
                    continue
                n_videos += 1
                key = "_".join(vid.split("_")[:3])
                groups.setdefault(key, set()).add(split)

    leaking = {}
    for key in groups:
        if len(groups[key]) > 1:
            leaking[key] = sorted(groups[key])

    print(f"classes checked : {len(target_classes)}")
    print(f"videos          : {n_videos}")
    print(f"groups          : {len(groups)}")
    print(f"groups in >1 split: {len(leaking)}")

    if leaking:
        pct = 100.0 * len(leaking) / max(len(groups), 1)
        print(f"\nLEAKAGE: {pct:.1f}% of groups span multiple splits.")
        print("Clips from one source video are in both train and test, so every")
        print("accuracy and the probe ceiling are inflated. Re-split by group.")
        for key in sorted(leaking)[:10]:
            print(f"  {key:<34} {leaking[key]}")
    else:
        print("\nOK - no group spans multiple splits. The split is group-disjoint.")

    return leaking


def leakage_example(class_name, input_root=None, splits=None):
    """Table of one class's groups in the shipped split, leaking groups first."""
    import pandas as pd

    splits     = splits or ["train", "val", "test"]
    input_root = input_root or cfg.DATASET_ROOT

    rows = []
    for split in splits:
        class_path = os.path.join(input_root, split, class_name)
        if not os.path.isdir(class_path):
            continue
        for vid in sorted(os.listdir(class_path)):
            if not vid.endswith(".avi"):
                continue
            key = "_".join(vid.split("_")[:3])
            rows.append({"group": key, "clip": vid, "split": split})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    n_splits = df.groupby("group")["split"].transform("nunique")
    df["group_leaks"] = n_splits > 1
    return df.sort_values(["group_leaks", "group", "split"],
                          ascending=[False, True, True]).reset_index(drop=True)


def build_group_split(target_classes=None, input_root=None, splits=None,
                      train_groups=18, val_groups=3):
    """Group-disjoint split: every group (source video) goes to one split. Returns {split: {class: [paths]}}."""
    target_classes = target_classes or cfg.SELECTED_CLASSES
    splits         = splits or ["train", "val", "test"]
    input_root     = input_root or cfg.DATASET_ROOT

    split_map = {"train": {}, "val": {}, "test": {}}

    for cls in target_classes:
        by_group = {}
        for split in splits:
            class_path = os.path.join(input_root, split, cls)
            if not os.path.isdir(class_path):
                continue
            for vid in os.listdir(class_path):
                if not vid.endswith(".avi"):
                    continue
                key = "_".join(vid.split("_")[:3])
                by_group.setdefault(key, []).append(os.path.join(class_path, vid))

        keys = sorted(by_group)
        train_keys = keys[:train_groups]
        val_keys   = keys[train_groups:train_groups + val_groups]
        test_keys  = keys[train_groups + val_groups:]

        split_map["train"][cls] = [p for k in train_keys for p in sorted(by_group[k])]
        split_map["val"][cls]   = [p for k in val_keys   for p in sorted(by_group[k])]
        split_map["test"][cls]  = [p for k in test_keys  for p in sorted(by_group[k])]

    return split_map


def one_video_per_group(split_map, split="train"):
    """Keep the first video of every group, so each clip comes from a different source video."""
    for cls, paths in split_map[split].items():
        seen, keep = set(), []
        for p in paths:
            key = "_".join(os.path.basename(p).split("_")[:3])
            if key not in seen:
                seen.add(key)
                keep.append(p)
        split_map[split][cls] = keep
    return split_map


def describe_group_split(split_map, target_classes=None):
    """Print clip counts per split and confirm the group assignment is disjoint."""
    target_classes = target_classes or sorted(split_map["train"])

    totals = {}
    for split in ["train", "val", "test"]:
        totals[split] = sum(len(split_map[split].get(c, [])) for c in target_classes)
    grand = sum(totals.values())

    print(f"{'split':<8}{'clips':>8}{'percent':>10}{'clips/class':>13}")
    print("-" * 39)
    for split in ["train", "val", "test"]:
        pct = 100.0 * totals[split] / max(grand, 1)
        per = totals[split] / max(len(target_classes), 1)
        print(f"{split:<8}{totals[split]:>8}{pct:>9.1f}%{per:>13.1f}")
    print("-" * 39)
    print(f"{'TOTAL':<8}{grand:>8}{'':>10}{grand / max(len(target_classes), 1):>13.1f}")

    seen = {}
    clashes = 0
    for split in ["train", "val", "test"]:
        for cls in target_classes:
            for path in split_map[split].get(cls, []):
                key = "_".join(os.path.basename(path).split("_")[:3])
                if key in seen and seen[key] != split:
                    clashes += 1
                seen[key] = split

    print(f"\ngroups: {len(seen)}   groups in >1 split: {clashes}")
    print("OK - group-disjoint." if clashes == 0 else "STILL LEAKING - do not proceed.")
    return clashes


def preprocess_group_split(split_map, target_classes=None, output_root=None, max_samples=None,
                           max_test_samples=None):
    """Save the clips of a split as .pt files.

    max_samples caps the train clips per class, max_test_samples the val/test clips (None = all).
    """
    target_classes = target_classes or sorted(split_map["train"])
    output_root    = output_root or cfg.OUTPUT_ROOT

    print(f"\n--- Preprocessing (group-disjoint) ---")
    print(f"To  : {output_root}")
    print(f"Classes: {len(target_classes)} | Train limit: {max_samples if max_samples else 'Full'}")

    os.makedirs(output_root, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_output_path = os.path.join(output_root, split)
        os.makedirs(split_output_path, exist_ok=True)
        print(f"\nProcessing split: {split}")

        for cls in target_classes:
            paths = split_map[split].get(cls, [])
            # cap the clips per class
            if max_samples is not None and split == "train":
                paths = paths[:max_samples]
            elif max_test_samples is not None and split in ("val", "test"):
                paths = paths[:max_test_samples]
            if not paths:
                continue
            class_output_path = os.path.join(split_output_path, cls)
            os.makedirs(class_output_path, exist_ok=True)

            for vid_path in tqdm(paths, desc=f"  {cls} [{split}]", leave=False):
                vid_name = os.path.basename(vid_path)
                output_filename = os.path.splitext(vid_name)[0] + ".pt"
                output_path = os.path.join(class_output_path, output_filename)

                if os.path.exists(output_path):
                    continue

                frames = extract_frames(vid_path)
                if not frames:
                    continue
                save_clip_tensor(temporal_sample(frames), output_path)

    print(f"\nDone. Clips ready at: {output_root}")


def plot_split_leakage(target_classes=None, input_root=None, train_groups=18, val_groups=3):
    """Bar chart of leaking groups: shipped split vs group-disjoint split. Returns the counts."""
    import matplotlib.pyplot as plt

    target_classes = target_classes or cfg.SELECTED_CLASSES
    input_root     = input_root or cfg.DATASET_ROOT
    splits         = ["train", "val", "test"]

    # shipped split: which splits each group's clips fall into
    shipped = {}
    for split in splits:
        for cls in target_classes:
            d = os.path.join(input_root, split, cls)
            if not os.path.isdir(d):
                continue
            for vid in os.listdir(d):
                if vid.endswith(".avi"):
                    shipped.setdefault("_".join(vid.split("_")[:3]), set()).add(split)
    total        = len(shipped)
    shipped_leak = sum(1 for v in shipped.values() if len(v) > 1)

    # group-disjoint split: assign whole groups, then recount
    split_map = build_group_split(target_classes, input_root, splits, train_groups, val_groups)
    seen = {}
    for split in splits:
        for cls in target_classes:
            for path in split_map[split].get(cls, []):
                seen.setdefault("_".join(os.path.basename(path).split("_")[:3]), set()).add(split)
    dj_total = len(seen)
    dj_leak  = sum(1 for v in seen.values() if len(v) > 1)

    labels = ["Shipped split", "Group-disjoint split"]
    clean  = [total - shipped_leak, dj_total - dj_leak]
    leak   = [shipped_leak, dj_leak]
    x      = range(len(labels))

    plt.figure(figsize=(7, 5))
    plt.bar(x, clean, 0.55, label="clean (one split)",       color="#1D9E75")
    plt.bar(x, leak,  0.55, bottom=clean, label="leaking (multi-split)", color="#D85A30")
    for i in x:
        pct = 100 * leak[i] / max(total, 1)
        plt.text(i, clean[i] + leak[i] + max(total, 1) * 0.01,
                 f"{leak[i]} leak ({pct:.0f}%)", ha="center", fontsize=11)
    plt.xticks(list(x), labels)
    plt.ylabel("groups (source videos)")
    plt.ylim(0, max(total, 1) * 1.15)
    plt.title(f"Data leakage across splits\n{total} groups, {len(target_classes)} classes",
              fontweight="bold")
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.show()

    print(f"shipped split      : {shipped_leak}/{total} groups leak ({100*shipped_leak/max(total,1):.0f}%)")
    print(f"group-disjoint split: {dj_leak}/{dj_total} groups leak ({100*dj_leak/max(dj_total,1):.0f}%)")
    return {"total": total, "shipped_leaking": shipped_leak, "disjoint_leaking": dj_leak}