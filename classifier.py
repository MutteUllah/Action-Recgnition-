"""
Step 4 of the pipeline: given a short clip (list of cropped frames around a
tracked person/vehicle), classify it into one of the crime categories.

Backbone: VideoMAE (HuggingFace transformers). Swap `MODEL_NAME` for a
SlowFast/X3D implementation later if you need faster inference; the
`ClipClassifier` interface (clip in -> (label, score) out) stays the same.
"""

from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

import config

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"  # starting checkpoint; replace with your fine-tuned one


class ClipClassifier:
    def __init__(self, num_classes: int = len(config.CRIME_CLASSES),
                 checkpoint: str = None, device: str = config.DEVICE):
        self.device = device
        self.processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)

        self.model = VideoMAEForVideoClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,   # we replace the head with our own class count
        )

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(state["model_state_dict"], strict=False)

        self.model.to(self.device).eval()
        self.labels = config.CRIME_CLASSES

    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> Tuple[str, float]:
        """
        frames: list of CLIP_LEN BGR numpy frames (already cropped/resized region).
        Returns (predicted_label, confidence).
        """
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        inputs = self.processor(frames_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        logits = self.model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1)[0]
        top_idx = int(torch.argmax(probs).item())
        return self.labels[top_idx], float(probs[top_idx].item())

    @torch.no_grad()
    def predict_all(self, frames: List[np.ndarray]) -> dict:
        """Return the full class -> probability distribution (useful for logging/debugging)."""
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        inputs = self.processor(frames_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        logits = self.model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        return {label: float(p) for label, p in zip(self.labels, probs)}
