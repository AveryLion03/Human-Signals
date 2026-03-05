"""
Real-time Face Mesh Detection using MediaPipe + OpenCV
------------------------------------------------------
Requirements:
    pip install mediapipe opencv-python urllib3

Run:
    python face_mesh.py

Controls:
    Q  →  Quit
    S  →  Save a snapshot to 'snapshot.png'
    T  →  Toggle tesselation on/off
    C  →  Toggle contours on/off
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# ── Model download ────────────────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"

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

    def __init__(self, num_faces: int = 2):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector   = vision.FaceLandmarker.create_from_options(options)
        self.drawing    = mp.solutions.drawing_utils
        self.styles     = mp.solutions.drawing_styles
        self.show_tess  = True
        self.show_cont  = True

    def process(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self.detector.detect(mp_image)

    def _to_proto(self, landmarks):
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in landmarks
        ])
        return proto

    def draw(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        if not results or not results.face_landmarks:
            return frame_bgr
        for face_landmarks in results.face_landmarks:
            proto = self._to_proto(face_landmarks)
            if self.show_tess:
                self.drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=proto,
                    connections=self.TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.styles.get_default_face_mesh_tesselation_style(),
                )
            if self.show_cont:
                self.drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=proto,
                    connections=self.CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.styles.get_default_face_mesh_contours_style(),
                )
        return frame_bgr


# ── HUD overlay ───────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, fps: float, num_faces: int,
             show_tess: bool, show_cont: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    small  = 0.52
    color  = (200, 255, 200)
    dim    = (120, 120, 120)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 26),  font, small, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {num_faces}", (120, 26), font, small, color, 1, cv2.LINE_AA)

    tess_col = color if show_tess else dim
    cont_col = color if show_cont else dim
    cv2.putText(frame, "[T] Tesselation", (w - 310, 26), font, small, tess_col, 1, cv2.LINE_AA)
    cv2.putText(frame, "[C] Contours",    (w - 155, 26), font, small, cont_col, 1, cv2.LINE_AA)

    # Bottom hint bar
    hint = "S: snapshot    Q: quit"
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, hint, (10, h - 10), font, small, (160, 160, 160), 1, cv2.LINE_AA)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tracker = FaceTracker(num_faces=2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check that it's connected and not in use.")
        return

    # Try to set a decent resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam opened. Controls: Q=quit  S=snapshot  T=tesselation  C=contours")

    import time
    prev_time = time.time()
    snapshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame — exiting.")
            break

        # ── Detection ──
        results   = tracker.process(frame)
        annotated = tracker.draw(frame, results)

        # ── FPS ──
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        num_faces = len(results.face_landmarks) if results and results.face_landmarks else 0

        # ── HUD ──
        annotated = draw_hud(annotated, fps, num_faces,
                             tracker.show_tess, tracker.show_cont)

        cv2.imshow("Face Mesh  —  MediaPipe", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snapshot_count += 1
            fname = f"snapshot_{snapshot_count:03d}.png"
            cv2.imwrite(fname, annotated)
            print(f"Snapshot saved → {fname}")
        elif key == ord("t"):
            tracker.show_tess = not tracker.show_tess
        elif key == ord("c"):
            tracker.show_cont = not tracker.show_cont

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()
