"""
End-to-end real-time pipeline.

    CCTV stream -> [detector.py: detect + track person/vehicle]
                -> [track_buffer.py: build per-track clip]
                -> [classifier.py: label the clip]
                -> [alert_logic.py: stabilize the label across time]
                -> draw box + label on frame, optionally raise an alert

Run:
    python pipeline.py --source 0                     # webcam
    python pipeline.py --source path/to/video.mp4
    python pipeline.py --source rtsp://user:pass@ip/stream
"""

import argparse
import time

import cv2

import config
from detector import PersonVehicleDetector
from classifier import ClipClassifier
from track_buffer import TrackBuffer
from alert_logic import AlertStabilizer

COLOR_NORMAL = (80, 200, 80)
COLOR_ALERT = (0, 0, 255)


def draw_box(frame, xyxy, label, score, is_alert):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    color = COLOR_ALERT if is_alert else COLOR_NORMAL
    thickness = 3 if is_alert else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text = f"{label} {score:.2f}" if score is not None else label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def run(source, display=True, save_path=None, checkpoint=None):
    detector = PersonVehicleDetector()
    classifier = ClipClassifier(checkpoint=checkpoint)
    buffer_mgr = TrackBuffer()
    stabilizer = AlertStabilizer()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    # last known (label, score) per track so boxes still show something between
    # classification windows, instead of blanking out every other frame
    last_result = {}

    print("Pipeline started. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        tracks = detector.track(frame)
        active_ids = {t.track_id for t in tracks}

        for t in tracks:
            buffer_mgr.update(frame, t.track_id, t.xyxy)

            if buffer_mgr.ready(t.track_id):
                clip = buffer_mgr.get_clip(t.track_id)
                label, score = classifier.predict(clip)
                stable_label = stabilizer.update(t.track_id, label, score)
                last_result[t.track_id] = (stable_label, score, label != "Normal")

            stable_label, score, is_alert = last_result.get(t.track_id, ("Analyzing...", None, False))
            draw_box(frame, t.xyxy, f"{t.cls_name}#{t.track_id}: {stable_label}", score, is_alert)

        buffer_mgr.prune(active_ids)
        stabilizer.prune(active_ids)

        fps_now = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(frame, f"FPS: {fps_now:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if writer:
            writer.write(frame)
        if display:
            cv2.imshow("CCTV Crime Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=config.VIDEO_SOURCE,
                         help="0 for webcam, or a file path / RTSP URL")
    parser.add_argument("--save", default=None, help="optional path to save annotated output video")
    parser.add_argument("--checkpoint", default=None, help="path to fine-tuned classifier checkpoint")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    run(source, display=not args.no_display, save_path=args.save, checkpoint=args.checkpoint)
