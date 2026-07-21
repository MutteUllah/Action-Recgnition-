"""
Turns the *actual* UCF-Crime download (as distributed by Sultani/Chen/Shah,
UNC-Charlotte) into a manifest CSV the training script can use directly --
no need to copy or reorganize the (very large) video files.

Your download typically looks like this after extracting the zips:

    data/
      Anomaly-Videos-Part-1/
        Abuse/Abuse001_x264.mp4 ...
        Arrest/...
        Arson/...
        Assault/...
      Anomaly-Videos-Part-2/
        Burglary/...
        Explosion/...
        Fighting/...
        RoadAccidents/...
      Anomaly-Videos-Part-3/
        Robbery/...
        Shooting/...
        Shoplifting/...
      Anomaly-Videos-Part-4/
        Stealing/...
        Vandalism/...
      Training-Normal-Videos-Part-1/*.mp4
      Training-Normal-Videos-Part-2/*.mp4
      Testing_Normal_Videos/*.mp4
      Normal_Videos_for_Event_Recognition/*.mp4
      UCF_Crimes-Train-Test-Split/
        Anomaly_Detection_splits/...
        Action_Regnition_splits/...
      Temporal_Anomaly_Annotation_for_Testing_Videos.txt
      ReadMe-Anomaly-Detection.txt

This script:
  1. Auto-unzips any .zip in `root` that hasn't been extracted yet.
  2. Recursively walks every folder under `root` looking for video files.
  3. Infers the class of each video from its parent folder name, matched
     (case-insensitively) against config.CRIME_CLASSES; any folder whose
     name contains "normal" is mapped to "Normal".
  4. Writes data/manifest.csv with columns: path,class
  5. Prints a per-class count so you can immediately see if anything was
     misclassified or missed.

Usage:
    python prepare_data.py --root data
"""

import argparse
import csv
import os
import zipfile
from collections import Counter

import config

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")


def unzip_all(root: str) -> None:
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".zip"):
                zip_path = os.path.join(dirpath, fname)
                target_dir = os.path.join(dirpath, os.path.splitext(fname)[0])
                if os.path.isdir(target_dir) and os.listdir(target_dir):
                    continue  # already extracted
                print(f"[prepare_data] Extracting {fname} ...")
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(dirpath)
                except zipfile.BadZipFile:
                    print(f"[prepare_data]   WARNING: {fname} is not a valid zip, skipping.")


def infer_class(video_path: str, root: str) -> str:
    """Look at every ancestor folder name (closest first) and match it against known classes."""
    rel_parts = os.path.relpath(video_path, root).split(os.sep)[:-1]  # exclude filename
    for part in reversed(rel_parts):  # closest-to-file folder first
        normalized = part.lower().replace("_", "").replace("-", "")
        if "normal" in normalized:
            return "Normal"
        for cls in config.CRIME_CLASSES:
            if cls.lower() == normalized or cls.lower() in normalized:
                return cls
    return None  # unmatched -- e.g. sits directly under a "Part-X" folder with no class subfolder


def build_manifest(root: str, out_csv: str) -> None:
    rows = []
    unmatched = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(VIDEO_EXTS):
                continue
            full_path = os.path.join(dirpath, fname)
            cls = infer_class(full_path, root)
            if cls is None:
                unmatched.append(full_path)
                continue
            rows.append((full_path, cls))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class"])
        writer.writerows(rows)

    counts = Counter(cls for _, cls in rows)
    print(f"\n[prepare_data] Wrote {len(rows)} videos to {out_csv}\n")
    print("Class distribution:")
    for cls in config.CRIME_CLASSES:
        print(f"  {cls:15s}: {counts.get(cls, 0)}")

    if unmatched:
        print(f"\n[prepare_data] WARNING: {len(unmatched)} videos could not be matched to a class "
              f"and were skipped. First few:")
        for p in unmatched[:10]:
            print(f"  {p}")
        print("  -> If these should count, either rename their parent folder to match a class "
              "in config.CRIME_CLASSES, or add a manual mapping in infer_class().")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=config.DATA_DIR,
                         help="folder containing the extracted/zipped UCF-Crime download")
    parser.add_argument("--out", default=os.path.join(config.DATA_DIR, "manifest.csv"))
    parser.add_argument("--skip_unzip", action="store_true", help="skip the auto-unzip step")
    args = parser.parse_args()

    if not args.skip_unzip:
        unzip_all(args.root)
    build_manifest(args.root, args.out)
