"""
Streamlit demo for the CCTV crime-detection pipeline.

Flow: user uploads a short video -> detection+tracking (YOLO11 + ByteTrack)
-> rolling per-track clip buffer -> VideoMAE classification per ready clip
-> alert stabilization across consecutive windows -> draw boxes/labels ->
show the annotated output video.

Designed for Streamlit Community Cloud (free, CPU-only):
- device hardcoded to "cpu" for the same reason as the Gradio version:
  config.py's DEVICE logic defaults to "cuda" unless FORCE_CPU=1 is set,
  which is risky to rely on on a host with no GPU at all.
- input videos are capped in duration and frame count so a visitor can't
  queue up something that takes forever on CPU.
"""

import os
import tempfile
import time
from collections import deque, defaultdict

import cv2
import streamlit as st

import config
from detector import PersonVehicleDetector
from track_buffer import TrackBuffer
from classifier import ClipClassifier

# ---------------------------------------------------------------------------
# Tunables for the hosted demo
# ---------------------------------------------------------------------------
DEVICE = "cpu"
MAX_INPUT_SECONDS = 12
PROCESS_EVERY_N_FRAMES = 1
BOX_COLOR_NORMAL = (80, 200, 80)   # BGR
BOX_COLOR_ALERT = (40, 40, 230)    # BGR

st.set_page_config(page_title="CCTV Crime Detection Demo", layout="wide")


class AlertStabilizer:
    """
    Minimal stand-in for alert_logic.py (not available at build time).
    Requires config.ALERT_SUSTAIN_WINDOWS consecutive matching positive
    predictions above config.ALERT_SCORE_THRESHOLD before a track_id is
    treated as a confirmed alert.
    """

    def __init__(self, sustain_windows: int = config.ALERT_SUSTAIN_WINDOWS):
        self.sustain_windows = sustain_windows
        self.history = defaultdict(lambda: deque(maxlen=sustain_windows))

    def update(self, track_id: int, label: str, score: float):
        is_positive = label in config.ALERT_CLASSES and score >= config.ALERT_SCORE_THRESHOLD
        self.history[track_id].append(label if is_positive else "Normal")
        hist = self.history[track_id]
        if len(hist) == self.sustain_windows and len(set(hist)) == 1 and hist[0] != "Normal":
            return hist[0]
        return None

    def prune(self, active_track_ids: set):
        stale = [tid for tid in self.history if tid not in active_track_ids]
        for tid in stale:
            del self.history[tid]


@st.cache_resource(show_spinner="Loading detector...")
def load_detector():
    return PersonVehicleDetector(weights=config.YOLO_WEIGHTS, device=DEVICE)


@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    checkpoint = config.CLASSIFIER_CKPT if os.path.exists(config.CLASSIFIER_CKPT) else None
    clf = ClipClassifier(
        num_classes=len(config.CRIME_CLASSES),
        checkpoint=checkpoint,
        device=DEVICE,
    )
    return clf, checkpoint is not None


def process_video(video_path, detector, classifier, progress_bar, status_text):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open that video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = max(1, min(total_frames_in_file, int(MAX_INPUT_SECONDS * fps)))

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    track_buffer = TrackBuffer(clip_len=config.CLIP_LEN, frame_size=config.FRAME_SIZE)
    stabilizer = AlertStabilizer()
    last_label_for_track = {}
    alerts_seen = {}

    start_time = time.time()
    frame_idx = 0
    tracked_boxes = []

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
            tracked_boxes = detector.track(frame)
            active_ids = set()

            for tb in tracked_boxes:
                active_ids.add(tb.track_id)
                track_buffer.update(frame, tb.track_id, tb.xyxy)

                if track_buffer.ready(tb.track_id):
                    clip = track_buffer.get_clip(tb.track_id)
                    label, score = classifier.predict(clip)
                    is_alert = label in config.ALERT_CLASSES and score >= config.ALERT_SCORE_THRESHOLD
                    last_label_for_track[tb.track_id] = (label, score, is_alert)

                    confirmed = stabilizer.update(tb.track_id, label, score)
                    if confirmed:
                        alerts_seen[tb.track_id] = confirmed

            track_buffer.prune(active_ids)
            stabilizer.prune(active_ids)

        for tb in tracked_boxes:
            x1, y1, x2, y2 = tb.xyxy.astype(int)
            label, score, is_alert = last_label_for_track.get(tb.track_id, ("...", 0.0, False))
            confirmed_label = alerts_seen.get(tb.track_id)

            color = BOX_COLOR_ALERT if (is_alert or confirmed_label) else BOX_COLOR_NORMAL
            display_text = f"ID{tb.track_id} {tb.cls_name}"
            if confirmed_label:
                display_text += f" | ALERT: {confirmed_label}"
            elif label != "...":
                display_text += f" | {label} ({score:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, display_text, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )

        writer.write(frame)
        frame_idx += 1
        progress_bar.progress(frame_idx / max_frames)
        status_text.text(f"Processing frame {frame_idx}/{max_frames}")

    cap.release()
    writer.release()
    elapsed = time.time() - start_time

    if alerts_seen:
        summary_lines = [f"Track {tid}: {label}" for tid, label in alerts_seen.items()]
        summary = "**Confirmed alerts:**\n\n" + "\n\n".join(summary_lines)
    else:
        summary = "No stabilized alerts triggered in the processed segment."
    summary += f"\n\nProcessed {frame_idx} frames in {elapsed:.1f}s on CPU."

    return out_path, summary


def main():
    st.title("🎥 CCTV Crime Detection — Live Demo")
    st.markdown(
        f"""
        Upload a short clip (max **{MAX_INPUT_SECONDS}s** processed). The pipeline detects
        people/vehicles, tracks them across frames, classifies each track's behavior into one
        of {len(config.CRIME_CLASSES)} categories, and raises a stabilized alert once a category
        is predicted consistently across consecutive windows.

        Running on **CPU** (free tier) — expect roughly real-time-or-slower processing, not
        live streaming.
        """
    )

    detector = load_detector()
    classifier, has_checkpoint = load_classifier()

    if not has_checkpoint:
        st.warning(
            f"No fine-tuned checkpoint found at `{config.CLASSIFIER_CKPT}`. Running with an "
            "**untrained** classification head — predicted labels are not meaningful until you "
            "run `train.py` and add the checkpoint to `checkpoints/`."
        )

    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.video(uploaded_file)

        if st.button("Run detection", type="primary"):
            tmp_in = tempfile.mktemp(suffix=os.path.splitext(uploaded_file.name)[1])
            with open(tmp_in, "wb") as f:
                f.write(uploaded_file.getbuffer())

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            out_path, summary = process_video(
                tmp_in, detector, classifier, progress_bar, status_text
            )
            status_text.empty()

            if out_path is None:
                st.error(summary)
            else:
                with col2:
                    st.subheader("Annotated output")
                    st.video(out_path)
                st.markdown(summary)


if __name__ == "__main__":
    main()
