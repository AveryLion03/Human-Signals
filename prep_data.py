"""
prepare_data.py
───────────────
Reads COCO-format Roboflow datasets, runs MediaPipe FaceMesh on every image,
extracts 478 landmark (x, y, z) coordinates as features, and saves to
landmarks.csv ready for train.py.

EXPECTED FOLDER STRUCTURE (one per dataset):
    dataset1/
        train/
            img1.jpg
            img2.jpg
            _annotations.coco.json
        valid/
            img1.jpg
            _annotations.coco.json

USAGE
-----
1. Edit DATA_ROOTS below to point at your 3 dataset root folders.
2. Run:  python prepare_data.py
3. Output: landmarks.csv
"""

import os
import json
import csv
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# ── CONFIG — edit these to your actual dataset paths ─────────────────────────
DATA_ROOTS = [
    r"C:\Users\avery\Downloads\Facial\driver behaviour.v2i.coco",   # e.g. r"C:\Downloads\driver-behaviour"
    r"C:\Users\avery\Downloads\Facial\driver monitoring.v1i.coco",   # e.g. r"C:\Downloads\driver-monitoring"
]

OUTPUT_CSV   = "landmarks.csv"
MODEL_PATH   = "face_landmarker.task"
SPLITS       = ["train", "valid", "test"]   # subfolders to scan
ANNOTATIONS  = "_annotations.coco.json"     # Roboflow's default COCO filename

# ── Class name normalisation ──────────────────────────────────────────────────
# Maps any Roboflow class name → awake | drowsy | asleep
# Add more entries here if your dataset uses different names.
CLASS_MAP = {
    # awake
    "awake": "awake", "alert": "awake", "focus": "awake",
    "focused": "awake", "normal": "awake", "attentive": "awake",
    "no_drowsiness": "awake", "nodrowsiness": "awake", "0": "awake",
    # drowsy
    "drowsy": "drowsy", "drowsiness": "drowsy", "sleepy": "drowsy",
    "tired": "drowsy", "fatigue": "drowsy", "yawn": "drowsy",
    "microsleep": "drowsy", "1": "drowsy",
    # asleep
    "asleep": "asleep", "sleep": "asleep", "sleeping": "asleep",
    "closed": "asleep", "eyes_closed": "asleep", "2": "asleep",
}

def normalise_class(name: str):
    key = name.lower().strip().replace("-", "_").replace(" ", "_")
    return CLASS_MAP.get(key, None)

# ── Model download ────────────────────────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task",
            MODEL_PATH,
        )
        print("Model downloaded.")

# ── Detector ──────────────────────────────────────────────────────────────────
def make_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.FaceLandmarker.create_from_options(options)

def extract_landmarks(detector, img_path: str):
    """Returns flat list of 478*3=1434 floats, or None if no face detected."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results  = detector.detect(mp_image)
    if not results or not results.face_landmarks:
        return None
    lms = results.face_landmarks[0]
    return [coord for lm in lms for coord in (lm.x, lm.y, lm.z)]

# ── COCO parser ───────────────────────────────────────────────────────────────
def load_coco_labels(split_dir: str):
    """
    Reads _annotations.coco.json and returns a dict:
        { filename: normalised_class_label }

    In object detection COCO JSON, one image may have multiple annotations
    (bounding boxes). We take the most common class label across all
    annotations for that image as the image-level label.
    """
    json_path = os.path.join(split_dir, ANNOTATIONS)
    if not os.path.exists(json_path):
        print(f"  [SKIP] No {ANNOTATIONS} found in {split_dir}")
        return {}

    with open(json_path, "r") as f:
        coco = json.load(f)

    # Build id → category name map
    cat_map = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    # Build image id → filename map
    img_map = {img["id"]: img["file_name"] for img in coco.get("images", [])}

    # Tally class votes per image
    votes = {}   # image_id → list of normalised labels
    for ann in coco.get("annotations", []):
        img_id   = ann["image_id"]
        cat_name = cat_map.get(ann["category_id"], "")
        label    = normalise_class(cat_name)
        if label is None:
            continue
        votes.setdefault(img_id, []).append(label)

    # Majority vote → final label per image
    result = {}
    for img_id, labels in votes.items():
        fname = img_map.get(img_id, "")
        if not fname:
            continue
        majority = max(set(labels), key=labels.count)
        result[os.path.basename(fname)] = majority

    return result

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ensure_model()
    detector = make_detector()

    total = skipped = written = 0
    class_counts = {"awake": 0, "drowsy": 0, "asleep": 0}
    unmapped_classes = set()

    header = ["label"] + [f"{ax}{i}" for i in range(478) for ax in ("x", "y", "z")]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for root in DATA_ROOTS:
            if not os.path.exists(root):
                print(f"\n[WARN] Path not found, skipping: {root}")
                continue
            print(f"\n{'─'*55}")
            print(f"Dataset: {root}")

            for split in SPLITS:
                split_dir = os.path.join(root, split)
                if not os.path.exists(split_dir):
                    continue

                print(f"  Split: {split}")
                labels = load_coco_labels(split_dir)

                if not labels:
                    # Log unmapped categories for debugging
                    json_path = os.path.join(split_dir, ANNOTATIONS)
                    if os.path.exists(json_path):
                        with open(json_path) as jf:
                            coco = json.load(jf)
                        cats = [c["name"] for c in coco.get("categories", [])]
                        print(f"  [WARN] No labels mapped. Categories found: {cats}")
                        print(f"         Add these to CLASS_MAP in prepare_data.py")
                        unmapped_classes.update(cats)
                    continue

                print(f"  Found {len(labels)} labelled images")

                for fname, label in labels.items():
                    img_path = os.path.join(split_dir, fname)
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue
                    total += 1
                    feats = extract_landmarks(detector, img_path)
                    if feats is None:
                        skipped += 1
                        continue
                    writer.writerow([label] + feats)
                    written += 1
                    class_counts[label] = class_counts.get(label, 0) + 1
                    if written % 100 == 0:
                        print(f"    ... {written} images processed")

    print(f"\n{'═'*55}")
    print(f"Done!")
    print(f"  Total images attempted : {total}")
    print(f"  Successfully written   : {written}")
    print(f"  Skipped (no face/file) : {skipped}")
    print(f"  Class breakdown        : {class_counts}")
    if unmapped_classes:
        print(f"\n  [!] Unmapped category names found: {unmapped_classes}")
        print(f"      Add them to CLASS_MAP in prepare_data.py and re-run.")
    if written < 100:
        print(f"\n  [WARN] Very few samples — double-check your DATA_ROOTS paths.")
    else:
        print(f"\n  landmarks.csv saved → ready to run train.py")

if __name__ == "__main__":
    main()
