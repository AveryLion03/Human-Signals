"""
face_mesh.py
────────────
Real-time driver alertness detection using MediaPipe FaceMesh + trained ML model.

Model input features (must match prepare_data.py — 1437 total):
    - 478 landmark (x,y,z) coords
    - EAR left, right, avg

Rule-based overrides (independent of model, computed live):
    - EAR  → eyes closed too long
    - MAR  → yawning detected
    - Head pose → head nodding / drooping

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
MODEL_PATH        = "face_landmarker.task"
CLASSIFIER_PATH   = "drowsy_model.pkl"
SMOOTHING_FRAMES  = 6
ALERT_THRESHOLD   = 0.65

NON_ALERT_THRESH  = 0.65    # probability threshold for Non-Alert
                             # keep in sync with evaluate_subjects.py

# EAR rule-based override
EAR_CLOSED_THRESH = 0.25
EAR_CLOSED_FRAMES = 6

# MAR rule-based override
MAR_YAWN_THRESH   = 0.65
MAR_YAWN_FRAMES   = 10

# Head pose rule-based override (degrees)
PITCH_THRESH      = 30.0    # chin-down nodding
ROLL_THRESH       = 30.0    # head tilting to shoulder

CLASS_COLORS = {
    "Alert":     (80,  200, 80),
    "Non-Alert": (0,   180, 255),
}

LABEL_DISPLAY = {
    "Alert":     "Alert",
    "Non-Alert": "Non-Alert",
    "awake":     "Alert",      # fallback for old model label names
    "drowsy":    "Non-Alert",
}

# ── EAR landmark indices ──────────────────────────────────────────────────────
LEFT_EYE  = {"outer": 33,  "inner": 133, "top1": 160, "top2": 158,
              "bot1":  144, "bot2":  153}
RIGHT_EYE = {"outer": 362, "inner": 263, "top1": 385, "top2": 387,
              "bot1":  373, "bot2":  380}

# ── MAR landmark indices ──────────────────────────────────────────────────────
MOUTH_TOP    = [13, 312, 311, 310]
MOUTH_BOTTOM = [14, 82,  81,  80]
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291

# ── Head pose 3D model points ─────────────────────────────────────────────────
HEAD_MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0   ],
    [0.0,   -330.0, -65.0  ],
    [-225.0, 170.0, -135.0 ],
    [225.0,  170.0, -135.0 ],
    [-150.0,-150.0, -125.0 ],
    [150.0, -150.0, -125.0 ],
], dtype=np.float64)
HEAD_MP_INDICES = [1, 152, 263, 33, 287, 57]

# ── EAR ───────────────────────────────────────────────────────────────────────
def compute_ear(landmarks, eye):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y])
    v1 = np.linalg.norm(pt(eye["top1"]) - pt(eye["bot1"]))
    v2 = np.linalg.norm(pt(eye["top2"]) - pt(eye["bot2"]))
    h  = np.linalg.norm(pt(eye["outer"]) - pt(eye["inner"]))
    return (v1 + v2) / (2.0 * h + 1e-6)

def compute_ears(landmarks):
    left  = compute_ear(landmarks, LEFT_EYE)
    right = compute_ear(landmarks, RIGHT_EYE)
    return left, right, (left + right) / 2.0

# ── MAR ───────────────────────────────────────────────────────────────────────
def compute_mar(landmarks):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y])
    top    = np.mean([pt(i) for i in MOUTH_TOP],    axis=0)
    bottom = np.mean([pt(i) for i in MOUTH_BOTTOM], axis=0)
    left   = pt(MOUTH_LEFT)
    right  = pt(MOUTH_RIGHT)
    return float(np.linalg.norm(top - bottom) /
                 (np.linalg.norm(left - right) + 1e-6))

# ── Head Pose ─────────────────────────────────────────────────────────────────
def compute_head_pose(landmarks, img_w, img_h):
    img_pts = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h]
         for i in HEAD_MP_INDICES],
        dtype=np.float64,
    )
    focal   = img_w
    cam_mat = np.array(
        [[focal, 0,     img_w / 2],
         [0,     focal, img_h / 2],
         [0,     0,     1        ]],
        dtype=np.float64,
    )
    success, rvec, _ = cv2.solvePnP(
        HEAD_MODEL_POINTS, img_pts, cam_mat,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    pitch  = float(np.degrees(np.arcsin(-rot[2, 0])))
    yaw    = float(np.degrees(np.arctan2(rot[2, 1], rot[2, 2])))
    roll   = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    return pitch, yaw, roll

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
        self.detector  = vision.FaceLandmarker.create_from_options(options)
        self.drawing   = mp.solutions.drawing_utils
        self.styles    = mp.solutions.drawing_styles
        self.show_tess = True
        self.show_cont = True

    def process(self, frame_bgr):
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect(mp_image)

    def get_feature_vector(self, results):
        """
        Returns (feature_vector, avg_ear, mar, pitch, yaw, roll, lms).
        Feature vector is 1437 features (landmarks + EAR only) to match
        prepare_data.py. MAR and head pose are returned separately for
        rule-based overrides but are NOT included in the model input.
        """
        if not results or not results.face_landmarks:
            return None, None, None, None, None, None, None

        lms    = results.face_landmarks[0]
        coords = np.array([c for lm in lms for c in (lm.x, lm.y, lm.z)],
                          dtype=np.float32)

        # Model input — EAR only
        left_ear, right_ear, avg_ear = compute_ears(lms)
        feature_vec = np.append(coords,
                                [left_ear, right_ear, avg_ear]).reshape(1, -1)

        # Rule-based signals — computed but not passed to model
        mar             = compute_mar(lms)
        pitch, yaw, roll = 0.0, 0.0, 0.0   # computed below after frame dims known

        return feature_vec, avg_ear, mar, pitch, yaw, roll, lms

    def get_head_pose(self, results, frame_bgr):
        """Separate head pose computation requiring frame dimensions."""
        if not results or not results.face_landmarks:
            return 0.0, 0.0, 0.0
        h, w = frame_bgr.shape[:2]
        return compute_head_pose(results.face_landmarks[0], w, h)

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
                    connection_drawing_spec=self.styles
                        .get_default_face_mesh_tesselation_style())
            if self.show_cont:
                self.drawing.draw_landmarks(
                    image=frame_bgr, landmark_list=proto,
                    connections=self.CONTOURS, landmark_drawing_spec=None,
                    connection_drawing_spec=self.styles
                        .get_default_face_mesh_contours_style())
        return frame_bgr

    def draw_indicators(self, frame, lms, avg_ear, mar, pitch, roll):
        """Draw EAR, MAR, and head pose readouts on frame."""
        if lms is None:
            return frame
        h, w = frame.shape[:2]

        eyes_open  = avg_ear >= EAR_CLOSED_THRESH
        yawning    = mar     >= MAR_YAWN_THRESH
        head_ok    = abs(pitch) < PITCH_THRESH and abs(roll) < ROLL_THRESH

        eye_color  = (80, 200, 80) if eyes_open else (50,  50, 220)
        mar_color  = (80, 200, 80) if not yawning else (0, 180, 255)
        pose_color = (80, 200, 80) if head_ok     else (0, 180, 255)

        # Eye corner dots
        for idx in [LEFT_EYE["outer"], LEFT_EYE["inner"],
                    RIGHT_EYE["outer"], RIGHT_EYE["inner"]]:
            lm = lms[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, eye_color, -1)

        lines = [
            (f"EAR:   {avg_ear:.3f}  {'[CLOSED]' if not eyes_open else ''}",
             eye_color),
            (f"MAR:   {mar:.3f}  {'[YAWN]' if yawning else ''}",
             mar_color),
            (f"Pitch: {pitch:+.1f}  Roll: {roll:+.1f}  "
             f"{'[HEAD DROP]' if not head_ok else ''}",
             pose_color),
        ]
        base_y = h - 70
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (10, base_y + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        return frame

# ── HUD ───────────────────────────────────────────────────────────────────────
def draw_hud(frame, fps, prediction, confidence, alert, override_reason,
             show_tess, show_cont, model_loaded):
    h, w  = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    small = 0.52

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 26), font, small,
                (200, 255, 200), 1, cv2.LINE_AA)

    if model_loaded and prediction:
        color = CLASS_COLORS.get(prediction, (255, 255, 255))
        label = f"{prediction.upper()}  {confidence*100:.0f}%"
        cv2.putText(frame, label, (w // 2 - 80, 26), font, 0.65,
                    color, 2, cv2.LINE_AA)

    tess_col = (200, 255, 200) if show_tess else (100, 100, 100)
    cont_col = (200, 255, 200) if show_cont else (100, 100, 100)
    cv2.putText(frame, "[T] Mesh",     (w - 200, 26), font, small,
                tess_col, 1, cv2.LINE_AA)
    cv2.putText(frame, "[C] Contours", (w - 120, 26), font, small,
                cont_col, 1, cv2.LINE_AA)

    if alert:
        alert_color = CLASS_COLORS["Non-Alert"]
        reason_map  = {
            "ear":   "⚠  EYES CLOSED  ⚠",
            "yawn":  "⚠  YAWNING DETECTED  ⚠",
            "pose":  "⚠  HEAD DROOPING  ⚠",
            "model": "⚠  NON-ALERTNESS DETECTED  ⚠",
        }
        msg = reason_map.get(override_reason, "⚠  NON-ALERTNESS DETECTED  ⚠")
        cv2.rectangle(frame, (0, h // 2 - 40), (w, h // 2 + 40), (0, 0, 0), -1)
        cv2.putText(frame, msg, (w // 2 - 220, h // 2 + 12),
                    font, 0.85, alert_color, 2, cv2.LINE_AA)

    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    hint = "S: snapshot    Q: quit    T: mesh    C: contours"
    if not model_loaded:
        hint = "[ No model loaded — run train.py first ]    Q: quit"
    cv2.putText(frame, hint, (10, h - 10), font, small,
                (160, 160, 160), 1, cv2.LINE_AA)

    return frame

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ensure_model()

    model_loaded  = False
    clf = le      = None
    non_alert_idx = None

    if os.path.exists(CLASSIFIER_PATH):
        with open(CLASSIFIER_PATH, "rb") as f:
            payload = pickle.load(f)
        clf          = payload["model"]
        le           = payload["label_encoder"]
        model_loaded = True
        print(f"Classifier loaded. Classes: {list(le.classes_)}")

        classes = list(le.classes_)
        if "Non-Alert" in classes:
            non_alert_idx = classes.index("Non-Alert")
        elif "drowsy" in classes:
            non_alert_idx = classes.index("drowsy")
        else:
            non_alert_idx = 1
    else:
        print(f"[WARN] {CLASSIFIER_PATH} not found — showing face mesh only.")

    tracker = FaceTracker()
    cap     = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Webcam opened.  Q=quit  S=snapshot  T=mesh  C=contours")

    pred_buffer        = collections.deque(maxlen=SMOOTHING_FRAMES)
    closed_frame_count = 0
    yawn_frame_count   = 0
    prev_time          = time.time()
    snapshot_n         = 0
    prediction         = None
    confidence         = 0.0
    alert              = False
    override_reason    = "model"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results   = tracker.process(frame)
        annotated = tracker.draw_mesh(frame, results)

        avg_ear = mar = pitch = yaw = roll = None
        lms_ref = None

        if model_loaded:
            feats, avg_ear, mar, _, _, _, lms_ref = \
                tracker.get_feature_vector(results)

            # Head pose needs frame dims — computed separately
            pitch, yaw, roll = tracker.get_head_pose(results, frame)

            if feats is not None:
                # ── ML prediction with threshold ──────────────────────────────
                proba          = clf.predict_proba(feats)[0]
                non_alert_prob = float(proba[non_alert_idx])

                if non_alert_prob >= NON_ALERT_THRESH:
                    pred_label = "Non-Alert"
                    pred_conf  = non_alert_prob
                else:
                    pred_idx   = int(np.argmax(proba))
                    raw_label  = le.inverse_transform([pred_idx])[0]
                    pred_label = LABEL_DISPLAY.get(raw_label, raw_label)
                    pred_conf  = float(proba[pred_idx])

                pred_buffer.append((pred_label, pred_conf))

                # ── Rule-based counters ───────────────────────────────────────
                closed_frame_count = (closed_frame_count + 1
                                      if avg_ear < EAR_CLOSED_THRESH else 0)
                yawn_frame_count   = (yawn_frame_count + 1
                                      if mar >= MAR_YAWN_THRESH else 0)

            if pred_buffer:
                labels     = [p[0] for p in pred_buffer]
                prediction = max(set(labels), key=labels.count)
                confidence = float(np.mean([p[1] for p in pred_buffer
                                            if p[0] == prediction]))

                # ── Rule-based overrides (priority order) ─────────────────────
                ear_override  = closed_frame_count >= EAR_CLOSED_FRAMES
                yawn_override = yawn_frame_count   >= MAR_YAWN_FRAMES
                pose_override = (pitch is not None and
                                 (abs(pitch) >= PITCH_THRESH or
                                  abs(roll)  >= ROLL_THRESH))

                if ear_override:
                    prediction, confidence, override_reason = "Non-Alert", 1.0, "ear"
                elif yawn_override:
                    prediction, confidence, override_reason = "Non-Alert", 1.0, "yawn"
                elif pose_override:
                    prediction, confidence, override_reason = "Non-Alert", 1.0, "pose"
                else:
                    override_reason = "model"

                alert = (prediction == "Non-Alert" and
                         (confidence >= ALERT_THRESHOLD or
                          ear_override or yawn_override or pose_override))

        if avg_ear is not None:
            annotated = tracker.draw_indicators(
                annotated, lms_ref, avg_ear, mar, pitch, roll)

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        annotated = draw_hud(annotated, fps, prediction, confidence,
                             alert, override_reason,
                             tracker.show_tess, tracker.show_cont,
                             model_loaded)

        cv2.imshow("Driver Alertness Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snapshot_n += 1
            os.makedirs("Images", exist_ok=True)
            fname = os.path.join("Images", f"snapshot_{snapshot_n:03d}.png")
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
