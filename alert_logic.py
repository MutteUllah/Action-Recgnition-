"""
Turns raw per-window classifier predictions into stable alerts.
Requires a class to be predicted with high confidence for N consecutive
classification windows before it is treated as a real event -- this is what
stops a single noisy frame from firing an alarm.
"""

from collections import deque, defaultdict

import config


class AlertStabilizer:
    def __init__(self, sustain: int = config.ALERT_SUSTAIN_WINDOWS,
                 threshold: float = config.ALERT_SCORE_THRESHOLD):
        self.sustain = sustain
        self.threshold = threshold
        self.history = defaultdict(lambda: deque(maxlen=sustain))

    def update(self, track_id: int, label: str, score: float) -> str:
        """
        Feed the latest (label, score) for a track. Returns the *displayed*
        label: either the raw label (if it just crossed the sustain threshold
        and is alert-worthy) or "Normal" otherwise.
        """
        is_positive = label in config.ALERT_CLASSES and score >= self.threshold
        self.history[track_id].append(label if is_positive else "Normal")

        hist = self.history[track_id]
        if len(hist) == self.sustain and len(set(hist)) == 1 and hist[0] != "Normal":
            return hist[0]
        return "Normal"

    def prune(self, active_track_ids: set) -> None:
        stale = [tid for tid in self.history if tid not in active_track_ids]
        for tid in stale:
            del self.history[tid]
