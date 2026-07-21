"""
Keeps a rolling window of cropped frames for every active track_id, so that
once CLIP_LEN frames are available we can hand a clip to the classifier.

This is what bridges "per-frame detection" -> "per-clip classification".
"""

from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np

import config


def _expand_and_clip_box(xyxy: np.ndarray, frame_shape, margin: float) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    x1 -= w * margin
    x2 += w * margin
    y1 -= h * margin
    y2 += h * margin
    H, W = frame_shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return np.array([x1, y1, x2, y2], dtype=int)


class TrackBuffer:
    def __init__(self, clip_len: int = config.CLIP_LEN, frame_size: int = config.FRAME_SIZE):
        self.clip_len = clip_len
        self.frame_size = frame_size
        self.buffers: Dict[int, deque] = {}
        self.frames_since_last_infer: Dict[int, int] = {}

    def update(self, frame: np.ndarray, track_id: int, xyxy: np.ndarray) -> None:
        box = _expand_and_clip_box(xyxy, frame.shape, config.BOX_MARGIN)
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        crop = cv2.resize(crop, (self.frame_size, self.frame_size))

        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.clip_len)
            self.frames_since_last_infer[track_id] = 0

        self.buffers[track_id].append(crop)
        self.frames_since_last_infer[track_id] += 1

    def ready(self, track_id: int) -> bool:
        """True once we have a full clip AND enough new frames have accrued since the last inference."""
        buf = self.buffers.get(track_id)
        if buf is None or len(buf) < self.clip_len:
            return False
        return self.frames_since_last_infer[track_id] >= config.CLIP_STRIDE

    def get_clip(self, track_id: int) -> Optional[List[np.ndarray]]:
        if track_id not in self.buffers:
            return None
        self.frames_since_last_infer[track_id] = 0
        return list(self.buffers[track_id])

    def prune(self, active_track_ids: set) -> None:
        """Drop buffers for tracks that disappeared (left frame / occluded / lost by tracker)."""
        stale = [tid for tid in self.buffers if tid not in active_track_ids]
        for tid in stale:
            del self.buffers[tid]
            del self.frames_since_last_infer[tid]
