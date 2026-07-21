"""
Central configuration for the CCTV crime-detection pipeline.
Edit paths / thresholds here rather than hunting through every file.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

YOLO_WEIGHTS = os.path.join(CHECKPOINT_DIR, "yolo11n.pt")          # auto-downloaded on first run
CLASSIFIER_CKPT = os.path.join(CHECKPOINT_DIR, "videomae_crime.pt")  # produced by train.py

# ---------------------------------------------------------------------------
# Detector / tracker
# ---------------------------------------------------------------------------
DETECT_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO ids
DETECT_CONF_THRESHOLD = 0.4
TRACKER_CFG = "bytetrack.yaml"          # ships with ultralytics

# ---------------------------------------------------------------------------
# Temporal clip / classifier
# ---------------------------------------------------------------------------
CLIP_LEN = 16                # frames fed to the classifier per inference window
CLIP_STRIDE = 4               # how many new frames before we re-run classification
FRAME_SIZE = 224              # spatial resize for the classifier input
BOX_MARGIN = 0.25             # extra context padding around a track's box (fraction of box size)

# UCF-Crime style label set (edit to match the classes you actually train on)
CRIME_CLASSES = [
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccident",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]

# Which of the classes above we treat as "alert-worthy" vs. background/normal
ALERT_CLASSES = {c for c in CRIME_CLASSES if c != "Normal"}
ALERT_SCORE_THRESHOLD = 0.55
ALERT_SUSTAIN_WINDOWS = 3     # require N consecutive positive windows before raising an alert (reduces flicker/false positives)

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEVICE = "cuda" if os.environ.get("FORCE_CPU", "0") != "1" else "cpu"
VIDEO_SOURCE = 0               # 0 = webcam; or path/RTSP URL, e.g. "rtsp://user:pass@ip:554/stream"
DISPLAY = True
SAVE_OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "outputs", "annotated.mp4")
