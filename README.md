# Real-Time CCTV Crime Detection

Detects **violence, robbery, accident, vandalism, and other anomalies** in a live
CCTV/video stream, draws a bounding box around the person/vehicle involved,
and labels it with the predicted event.

## Pipeline

```
CCTV stream
   -> detector.py       : YOLO11 detects person/vehicle + ByteTrack assigns track IDs
   -> track_buffer.py   : builds a rolling clip of cropped frames per track
   -> classifier.py     : VideoMAE classifies the clip into a crime category
   -> alert_logic.py    : requires several consecutive positive windows before
                           flagging an alert (reduces false positives/flicker)
   -> pipeline.py        : draws the box + label, shows/save the annotated video
```

Training side:

```
dataset.py  : loads UCF-Crime-style folders, weak video-level labels
train.py    : fine-tunes VideoMAE's classifier head (+ optionally backbone)
```

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate      # or conda
pip install -r requirements.txt
```

GPU strongly recommended for both training and real-time inference. Set
`FORCE_CPU=1` as an environment variable to force CPU mode (slow, for testing only).

## 2. Get a dataset

Recommended: **UCF-Crime** (13 real-world anomaly classes matching this
project's label set: Abuse, Arrest, Arson, Assault, Burglary, Explosion,
Fighting, RoadAccident, Robbery, Shooting, Shoplifting, Stealing, Vandalism).

- Official page: https://www.crcv.ucf.edu/projects/real-world/
- **Important caveat:** UCF-Crime only has *video-level* labels for training
  (it tells you the video is "Robbery", not which second/box). Training here
  therefore treats every clip sampled from an anomalous video as that video's
  class — this is standard weakly-supervised practice (same assumption used
  in the original UCF-Crime paper's MIL method), but it does introduce label
  noise since the anomaly usually isn't present for the whole video. This is
  why `alert_logic.py` waits for several consecutive positive windows before
  raising a real alert instead of trusting a single prediction.

Optional supplements:
- **CamNuvem** — a much larger robbery-only dataset, useful if robbery
  detection quality matters most to you.
- **XD-Violence** — has an actual bounding-box-annotated test subset (from a
  2025 follow-up paper) if you want to validate spatial localization quality,
  though its class list is narrower (no explicit "robbery"/"vandalism").

### Real download layout (what you actually get from UNC-Charlotte)

The official download extracts into something like this — **do not
reorganize it by hand**, `prepare_data.py` (next step) handles it as-is:

```
data/
  Anomaly-Videos-Part-1/          # Abuse, Arrest, Arson, Assault
  Anomaly-Videos-Part-2/          # Burglary, Explosion, Fighting, RoadAccidents
  Anomaly-Videos-Part-3/          # Robbery, Shooting, Shoplifting
  Anomaly-Videos-Part-4/          # Stealing, Vandalism
  Training-Normal-Videos-Part-1/
  Training-Normal-Videos-Part-2/
  Testing_Normal_Videos/
  Normal_Videos_for_Event_Recognition/
  UCF_Crimes-Train-Test-Split/
  Temporal_Anomaly_Annotation_for_Testing_Videos.txt
  ReadMe-Anomaly-Detection.txt
```

Put (or symlink) that whole tree under this project's `data/` folder, then run:

```bash
python prepare_data.py --root data
```

This will:
1. Auto-unzip anything still zipped (e.g. `Testing_Normal_Videos.zip`) in place.
2. Recursively walk every folder and find every video file.
3. Infer each video's class from its parent folder name (matched against
   `config.CRIME_CLASSES`; any folder with "normal" in the name → `Normal`).
4. Write `data/manifest.csv` (columns: `path,class`) and print a per-class
   video count so you can immediately sanity-check nothing was missed or
   miscategorized.

No video files are copied or moved — the manifest just records paths, so
this works fine even with the ~4.5 GB+ zip files you already have.

## 3. Fine-tune the classifier

```bash
python train.py --manifest data/manifest.csv \
                 --epochs 10 --batch_size 4 --freeze_backbone
```

- Start with `--freeze_backbone` (trains only the classification head —
  fast, works with limited data). Once that's stable, drop the flag to fine-tune
  the whole network for a few more epochs at a lower `--lr` (e.g. `1e-5`).
- Best checkpoint is saved to `checkpoints/videomae_crime.pt`.

## 4. Run real-time inference

```bash
# Webcam
python pipeline.py --source 0 --checkpoint checkpoints/videomae_crime.pt

# Video file
python pipeline.py --source path/to/cctv_clip.mp4 --checkpoint checkpoints/videomae_crime.pt

# RTSP CCTV stream
python pipeline.py --source rtsp://user:pass@192.168.1.10:554/stream1 \
                    --checkpoint checkpoints/videomae_crime.pt --save outputs/annotated.mp4
```

Press `q` in the display window to stop. Each tracked person/vehicle gets a
box colored green (normal) or red (alert-worthy), labeled with the predicted
class and confidence.

## Performance notes / next steps

- **VideoMAE-base is not real-time on modest hardware.** For deployment on
  weaker GPUs or edge devices, swap it for **X3D-S** or **MoViNet-A1** —
  `classifier.py`'s `ClipClassifier` interface (clip in -> label,score out)
  is deliberately backbone-agnostic, so you can drop in a different model
  without touching `pipeline.py`.
- `config.py` centralizes every tunable knob: clip length/stride, box margin,
  confidence thresholds, alert sustain window, which classes count as
  "alert-worthy".
- Current tracker is Ultralytics' built-in ByteTrack. If you need to survive
  longer occlusions (e.g. someone briefly leaves frame during a robbery),
  consider swapping in a stronger ReID-based tracker.
- For genuinely per-box (not per-scene) ground truth, you'll eventually want
  to hand-label a subset of your own footage — the weak-label approach here
  gets you a working system quickly, but supervised fine-tuning on even a
  few hundred manually boxed/labeled clips will materially improve precision.
