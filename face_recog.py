"""
face_mesh.py
────────────
Real-time driver drowsiness detection using MediaPipe FaceMesh + trained ML model.

Requirements:
    pip install mediapipe==0.10.9 opencv-python scikit-learn pandas

Run:
    python face_mesh.py

Controls:
    Q  →  Quit
    S  →  Save snapshot
    T  →  Toggle tesselation
    C  →  Toggle contours
"""

import os
import cv2
import time
import pickle
import collections
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH       = "face_landmarker.task"
CLASSIFIER_PATH  = "drowsy_model.pkl"
SMOOTHING_FRAMES = 10       # rolling average window for stable predictions
ALERT_THRESHOLD  = 0.60     # confidence above which DROWSY/ASLEEP triggers alert

# Alert colours per class
CLASS_COLORS = {
    "awake":  (80,  200, 80),   # green
    "drowsy": (0,   180, 255),  # orange
    "asleep": (50,  50,  220),  # red
}

# ── Model download ────────────────────────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model (~3 MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task",
            MODEL_PATH,
        )
        print("Model downloaded.")

# ── FaceTracker ───────────────────────────────────────────────────────────────
class FaceTracker:
    TESSELATION = mp.solutions.face_mesh.FACEMESH_TESSELATION
    CONTOURS    = mp.solutions.face_mesh.FACEMESH_CONTOURS

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector   = vision.FaceLandmarker.create_from_options(options)
        self.drawing    = mp.solutions.drawing_utils
        self.styles     = mp.solutions.drawing_styles
        self.show_tess  = True
        self.show_cont  = True

    def process(self, frame_bgr):
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect(mp_image)

    def get_feature_vector(self, results):
        """Return flat [x,y,z * 478] feature vector, or None if no face."""
        if not results or not results.face_landmarks:
            return None
        lms = results.face_landmarks[0]
        return np.array([c for lm in lms for c in (lm.x, lm.y, lm.z)],
                        dtype=np.float32).reshape(1, -1)

    def _to_proto(self, landmarks):
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in landmarks
        ])
        return proto

    def draw_mesh(self, frame_bgr, results):
        if not results or not results.face_landmarks:
            return frame_bgr
        for face_lms in results.face_landmarks:
            proto = self._to_proto(face_lms)
            if self.show_tess:
                self.drawing.draw_landmarks(
                    image=frame_bgr, landmark_list=proto,
                    connections=self.TESSELATION, landmark_drawing_spec=None,
                    connection_drawing_spec=self.styles.get_default_face_mesh_tesselation_style())
            if self.show_cont:
                self.drawing.draw_landmarks(
                    image=frame_bgr, landmark_list=proto,
                    connections=self.CONTOURS, landmark_drawing_spec=None,
                    connection_drawing_spec=self.styles.get_default_face_mesh_contours_style())
        return frame_bgr

# ── HUD ───────────────────────────────────────────────────────────────────────
def draw_hud(frame, fps, prediction, confidence, alert,
             show_tess, show_cont, model_loaded):
    h, w = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    small = 0.52

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 26), font, small, (200, 255, 200), 1, cv2.LINE_AA)

    if model_loaded and prediction:
        color = CLASS_COLORS.get(prediction, (255, 255, 255))
        label = f"{prediction.upper()}  {confidence*100:.0f}%"
        cv2.putText(frame, label, (w//2 - 80, 26), font, 0.65, color, 2, cv2.LINE_AA)

    tess_col = (200, 255, 200) if show_tess else (100, 100, 100)
    cont_col = (200, 255, 200) if show_cont else (100, 100, 100)
    cv2.putText(frame, "[T] Mesh",     (w - 200, 26), font, small, tess_col, 1, cv2.LINE_AA)
    cv2.putText(frame, "[C] Contours", (w - 120, 26), font, small, cont_col, 1, cv2.LINE_AA)

    # Drowsiness alert banner
    if alert and prediction in ("drowsy", "asleep"):
        alert_color = CLASS_COLORS[prediction]
        msg = "⚠  DROWSINESS DETECTED  ⚠" if prediction == "drowsy" else "⚠  DRIVER ASLEEP  ⚠"
        cv2.rectangle(frame, (0, h//2 - 40), (w, h//2 + 40), (0, 0, 0), -1)
        cv2.putText(frame, msg, (w//2 - 200, h//2 + 12),
                    font, 0.9, alert_color, 2, cv2.LINE_AA)

    # Bottom hint bar
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    hint = "S: snapshot    Q: quit"
    if not model_loaded:
        hint = "[ No model loaded — run train.py first ]    Q: quit"
    cv2.putText(frame, hint, (10, h - 10), font, small, (160, 160, 160), 1, cv2.LINE_AA)

    return frame

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ensure_model()

    # Load classifier if available
    model_loaded = False
    clf = le = None
    if os.path.exists(CLASSIFIER_PATH):
        with open(CLASSIFIER_PATH, "rb") as f:
            payload = pickle.load(f)
        clf          = payload["model"]
        le           = payload["label_encoder"]
        model_loaded = True
        print(f"Classifier loaded. Classes: {list(le.classes_)}")
    else:
        print(f"[WARN] {CLASSIFIER_PATH} not found — showing face mesh only.")
        print("       Run train.py to generate the model.")

    tracker = FaceTracker()
    cap     = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Webcam opened.  Q=quit  S=snapshot  T=mesh  C=contours")

    # Rolling prediction smoother
    pred_buffer = collections.deque(maxlen=SMOOTHING_FRAMES)
    prev_time   = time.time()
    snapshot_n  = 0
    prediction  = None
    confidence  = 0.0
    alert       = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results  = tracker.process(frame)
        annotated = tracker.draw_mesh(frame, results)

        # ── Classification ──
        if model_loaded:
            feats = tracker.get_feature_vector(results)
            if feats is not None:
                proba      = clf.predict_proba(feats)[0]
                pred_idx   = int(np.argmax(proba))
                pred_label = le.inverse_transform([pred_idx])[0]
                pred_conf  = float(proba[pred_idx])
                pred_buffer.append((pred_label, pred_conf))

            if pred_buffer:
                # Majority vote over smoothing window
                labels     = [p[0] for p in pred_buffer]
                prediction = max(set(labels), key=labels.count)
                confidence = np.mean([p[1] for p in pred_buffer
                                      if p[0] == prediction])
                alert      = (prediction in ("drowsy", "asleep")
                              and confidence >= ALERT_THRESHOLD)

        # ── FPS ──
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        annotated = draw_hud(annotated, fps, prediction, confidence,
                             alert, tracker.show_tess, tracker.show_cont,
                             model_loaded)

        cv2.imshow("Driver Drowsiness Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snapshot_n += 1
            fname = f"snapshot_{snapshot_n:03d}.png"
            cv2.imwrite(fname, annotated)
            print(f"Snapshot → {fname}")
        elif key == ord("t"):
            tracker.show_tess = not tracker.show_tess
        elif key == ord("c"):
            tracker.show_cont = not tracker.show_cont

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")

if __name__ == "__main__":
    main()
