"""
Step 1 + 2 of the pipeline: object detection (person / vehicle) and
multi-object tracking so each detected box gets a stable track_id across frames.

Uses Ultralytics YOLO, which ships ByteTrack out of the box via `.track()`.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

import config


@dataclass
class TrackedBox:
    track_id: int
    cls_name: str
    xyxy: np.ndarray   # [x1, y1, x2, y2] in pixel coords
    conf: float


class PersonVehicleDetector:
    """Thin wrapper: frame in -> list of TrackedBox out."""

    def __init__(self, weights: str = config.YOLO_WEIGHTS, device: str = config.DEVICE):
        self.model = YOLO(weights)
        self.device = device

    def track(self, frame: np.ndarray) -> List[TrackedBox]:
        """
        Run detection + tracking on a single BGR frame.
        Returns only the classes we care about (person / vehicle types).
        """
        results = self.model.track(
            frame,
            persist=True,                      # keep track state across calls
            tracker=config.TRACKER_CFG,
            classes=list(config.DETECT_CLASSES.keys()),
            conf=config.DETECT_CONF_THRESHOLD,
            device=self.device,
            verbose=False,
        )

        boxes_out: List[TrackedBox] = []
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return boxes_out

        r = results[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for box, tid, conf, cls_id in zip(xyxy, ids, confs, clss):
            cls_name = config.DETECT_CLASSES.get(int(cls_id), "object")
            boxes_out.append(TrackedBox(track_id=int(tid), cls_name=cls_name, xyxy=box, conf=float(conf)))

        return boxes_out
