"""
Dataset loader for fine-tuning the clip classifier on UCF-Crime (or any
similarly video-level-labeled dataset).

Primary path: build a (video_path, class) sample list from a manifest CSV
produced by `prepare_data.py`, which was written specifically to work with
the *actual* UCF-Crime download layout (Anomaly-Videos-Part-1..4,
Training-Normal-Videos-Part-1..2, Testing_Normal_Videos, etc.) without
requiring you to copy/reorganize any of the (large) video files:

    python prepare_data.py --root data
    python train.py --manifest data/manifest.csv ...

UCF-Crime ships with only VIDEO-LEVEL labels for training (it does not tell
you which exact seconds are anomalous, let alone which bounding box). This
loader therefore does the standard weakly-supervised thing: every clip
sampled from an anomalous video inherits that video's label, and every clip
from a normal video is labeled "Normal". This is the same assumption used in
the original Sultani et al. MIL paper -- expect some label noise on
anomalous videos (the anomaly usually only occupies part of the video), and
factor that into how much you trust a single window's prediction (hence the
AlertStabilizer in alert_logic.py requiring several consecutive positive
windows before treating it as real).
"""

import csv
import os
import random
from typing import List, Tuple

import av
import numpy as np
import torch
from torch.utils.data import Dataset

import config


def _read_clip(video_path: str, clip_len: int, frame_size: int) -> np.ndarray:
    """Uniformly sample `clip_len` frames across the whole video and resize them."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames or int(stream.duration * stream.time_base * stream.average_rate)
    total_frames = max(total_frames, clip_len)

    indices = sorted(random.sample(range(total_frames), min(clip_len, total_frames)))
    while len(indices) < clip_len:            # pad if the video is shorter than clip_len
        indices.append(indices[-1])

    frames = []
    idx_set = set(indices)
    for i, frame in enumerate(container.decode(stream)):
        if i in idx_set:
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
        if len(frames) >= clip_len:
            break
    container.close()

    while len(frames) < clip_len:              # safety pad
        frames.append(frames[-1])

    import cv2
    frames = [cv2.resize(f, (frame_size, frame_size)) for f in frames]
    return np.stack(frames, axis=0)            # [T, H, W, C]


class UCFCrimeClipDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str]], clip_len: int = config.CLIP_LEN,
                 frame_size: int = config.FRAME_SIZE, clips_per_video: int = 1):
        """
        samples: list of (video_path, class_name) tuples, class_name in config.CRIME_CLASSES
        clips_per_video: how many random clips to sample per video per epoch
                         (repeats entries so long anomalous videos get more coverage)
        """
        self.samples = []
        for path, cls in samples:
            self.samples.extend([(path, cls)] * clips_per_video)
        self.clip_len = clip_len
        self.frame_size = frame_size
        self.label_to_idx = {c: i for i, c in enumerate(config.CRIME_CLASSES)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        clip = _read_clip(path, self.clip_len, self.frame_size)   # [T, H, W, C], uint8 RGB
        label = self.label_to_idx[cls]
        return torch.from_numpy(clip), label


def build_samples_from_manifest(manifest_csv: str) -> List[Tuple[str, str]]:
    """
    Read (path, class) pairs written by prepare_data.py. This is the
    recommended way to load the real UCF-Crime download -- it works
    directly against the nested Anomaly-Videos-Part-X / Training-Normal-...
    folders without requiring you to copy or reorganize any video files.
    """
    samples = []
    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path, cls = row["path"], row["class"]
            if cls not in config.CRIME_CLASSES:
                continue
            if not os.path.isfile(path):
                continue
            samples.append((path, cls))
    return samples


def build_samples_from_folder(root: str) -> List[Tuple[str, str]]:
    """
    Walk data/UCF_Crimes/videos/<ClassName>/*.mp4 and build (path, class_name) pairs.
    Folder name is matched (case-insensitively, ignoring underscores) against
    config.CRIME_CLASSES; anything under a folder containing "normal" maps to "Normal".
    """
    samples = []
    for class_dir in sorted(os.listdir(root)):
        full_dir = os.path.join(root, class_dir)
        if not os.path.isdir(full_dir):
            continue

        normalized = class_dir.lower().replace("_", "").replace("-", "")
        if "normal" in normalized:
            mapped = "Normal"
        else:
            mapped = next(
                (c for c in config.CRIME_CLASSES if c.lower() == normalized.replace("training", "").replace("testing", "")),
                None,
            )
        if mapped is None:
            print(f"[dataset] Skipping unrecognized folder: {class_dir}")
            continue

        for fname in os.listdir(full_dir):
            if fname.lower().endswith((".mp4", ".avi", ".mkv")):
                samples.append((os.path.join(full_dir, fname), mapped))

    return samples
