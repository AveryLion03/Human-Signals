"""
prepare_data.py
───────────────
Reads the Driver Drowsiness Dataset (DDD) folder structure:
    dataset/
        Drowsy/
            img1.jpg ...
        Non-Drowsy/
            img1.jpg ...

Runs MediaPipe FaceMesh, extracts 478 landmark (x,y,z) coords +
3 EAR (Eye Aspect Ratio) features → landmarks.csv

EAR drops sharply when eyes close, making it a strong drowsiness signal.
"""

import os
import csv
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT  = r"E:\Facial\Human-Signals-main\DDD"   # folder containing Drowsy/ and Non-Drowsy/
OUTPUT_CSV = "landmarks.csv"
MODEL_PATH = "face_landmarker.task"

CLASS_MAP = {
    "nondrowsy":  "awake",
    "non_drowsy": "awake",
    "alert":      "awake",
    "awake":      "awake",
    "drowsy":     "drowsy",
}

# ── MediaPipe landmark indices for EAR ───────────────────────────────────────
# Left eye:  outer=33, inner=133, top1=160, top2=158, bot1=144, bot2=153
# Right eye: outer=362, inner=263, top1=385, top2=387, bot1=373, bot2=380
LEFT_EYE  = {"outer": 33,  "inner": 133, "top1": 160, "top2": 158,
              "bot1":  144, "bot2":  153}
RIGHT_EYE = {"outer": 362, "inner": 263, "top1": 385, "top2": 387,
              "bot1":  373, "bot2":  380}

def compute_ear(landmarks, eye):
    """Eye Aspect Ratio = (v1 + v2) / (2 * horizontal)"""
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y])

    v1 = np.linalg.norm(pt(eye["top1"]) - pt(eye["bot1"]))
    v2 = np.linalg.norm(pt(eye["top2"]) - pt(eye["bot2"]))
    h  = np.linalg.norm(pt(eye["outer"]) - pt(eye["inner"]))
    return (v1 + v2) / (2.0 * h + 1e-6)

def compute_ears(landmarks):
    """Returns (left_ear, right_ear, avg_ear)."""
    left  = compute_ear(landmarks, LEFT_EYE)
    right = compute_ear(landmarks, RIGHT_EYE)
    return left, right, (left + right) / 2.0

# ── Model ─────────────────────────────────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task",
            MODEL_PATH,
        )
        print("Model downloaded.")

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

def extract_features(detector, img_path: str):
    """
    Returns flat list of (478*3 landmark coords) + (left_ear, right_ear, avg_ear),
    or None if no face detected. Deletes unusable images.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        os.remove(img_path)
        return None
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results  = detector.detect(mp_image)
    if not results or not results.face_landmarks:
        os.remove(img_path)
        return None

    lms        = results.face_landmarks[0]
    coords     = [c for lm in lms for c in (lm.x, lm.y, lm.z)]
    left_ear, right_ear, avg_ear = compute_ears(lms)
    return coords + [left_ear, right_ear, avg_ear]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ensure_model()
    detector = make_detector()

    landmark_cols = [f"{ax}{i}" for i in range(478) for ax in ("x", "y", "z")]
    header = ["label"] + landmark_cols + ["ear_left", "ear_right", "ear_avg"]

    total = written = deleted = 0
    class_counts = {"awake": 0, "drowsy": 0}

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for folder in os.listdir(DATA_ROOT):
            label = CLASS_MAP.get(folder.lower().strip().replace("-", "_"), None)
            if label is None:
                print(f"[SKIP] Unrecognised folder: {folder}")
                continue

            folder_path = os.path.join(DATA_ROOT, folder)
            if not os.path.isdir(folder_path):
                continue

            files = [fn for fn in os.listdir(folder_path)
                     if fn.lower().endswith((".jpg", ".jpeg", ".png"))]
            print(f"\nProcessing '{folder}' → '{label}' ({len(files)} images)")

            for fname in files:
                img_path = os.path.join(folder_path, fname)
                total += 1
                feats = extract_features(detector, img_path)
                if feats is None:
                    deleted += 1
                    continue
                writer.writerow([label] + feats)
                written += 1
                class_counts[label] += 1
                if written % 200 == 0:
                    print(f"  ... {written} written")

    print(f"\n{'═'*50}")
    print(f"Total attempted : {total}")
    print(f"Written         : {written}")
    print(f"Deleted/skipped : {deleted}")
    print(f"Class breakdown : {class_counts}")
    print(f"landmarks.csv ready → run train_model.py")

if __name__ == "__main__":
    main()