"""
Generate AI detection overlay images using real models.
- MediaPipe Face Mesh → face wireframe
- YOLOv8-Pose → person bounding box + skeleton
Outputs: normal_overlay.png, smoking_overlay.png
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))

# ── MediaPipe Face Mesh ────────────────────────────────────────
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

FACE_MESH_SPEC = mp_draw.DrawingSpec(
    color=(0, 255, 210), thickness=1, circle_radius=1)
FACE_CONTOUR_SPEC = mp_draw.DrawingSpec(
    color=(0, 255, 190), thickness=1, circle_radius=0)


def draw_face_mesh(img):
    """Draw MediaPipe Face Mesh on image."""
    with mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as fm:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = fm.process(rgb)
        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                # Tesselation (dense triangulated mesh)
                mp_draw.draw_landmarks(
                    img, fl,
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles
                    .get_default_face_mesh_tesselation_style())
                # Contours
                mp_draw.draw_landmarks(
                    img, fl,
                    mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles
                    .get_default_face_mesh_contours_style())
                # Irises
                mp_draw.draw_landmarks(
                    img, fl,
                    mp_face.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles
                    .get_default_face_mesh_iris_connections_style())

                # Face bounding box with corner brackets
                h, w = img.shape[:2]
                xs = [lm.x * w for lm in fl.landmark]
                ys = [lm.y * h for lm in fl.landmark]
                x1, y1 = int(min(xs)) - 15, int(min(ys)) - 15
                x2, y2 = int(max(xs)) + 15, int(max(ys)) + 15
                _draw_bracket_box(img, x1, y1, x2, y2,
                                  (210, 255, 0), "Face Mesh", 0.98)
    return img


# ── YOLOv8-Pose ───────────────────────────────────────────────

# COCO pose skeleton: (start_idx, end_idx) pairs
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

LIMB_COLORS = [
    (255, 180, 0), (255, 180, 0), (255, 220, 50), (255, 220, 50),
    (255, 220, 50),
    (0, 180, 255), (0, 255, 120),
    (0, 180, 255), (0, 255, 120),
    (200, 100, 255), (200, 100, 255),
    (200, 100, 255),
    (255, 160, 80), (0, 200, 200),
    (255, 160, 80), (0, 200, 200),
]

KPT_COLOR = (255, 255, 255)


def draw_yolo_pose(img):
    """Draw YOLOv8-Pose detections: person bbox + skeleton."""
    model = YOLO("yolov8n-pose.pt")
    results = model(img, verbose=False)

    for r in results:
        if r.keypoints is None:
            continue
        boxes = r.boxes
        kpts_all = r.keypoints.data.cpu().numpy()

        for i, kpts in enumerate(kpts_all):
            # Person bounding box
            if boxes is not None and i < len(boxes):
                b = boxes[i]
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                _draw_bracket_box(img, x1 - 10, y1 - 10, x2 + 10, y2 + 10,
                                  (120, 255, 0), "Person", conf)

            # Skeleton limbs
            for idx, (s, e) in enumerate(SKELETON):
                if kpts[s][2] > 0.3 and kpts[e][2] > 0.3:
                    pt1 = (int(kpts[s][0]), int(kpts[s][1]))
                    pt2 = (int(kpts[e][0]), int(kpts[e][1]))
                    color = LIMB_COLORS[idx % len(LIMB_COLORS)]
                    cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)

            # Keypoint dots
            for kp in kpts:
                if kp[2] > 0.3:
                    cx, cy = int(kp[0]), int(kp[1])
                    cv2.circle(img, (cx, cy), 6, (255, 255, 255, 50), -1,
                               cv2.LINE_AA)
                    cv2.circle(img, (cx, cy), 4, KPT_COLOR, -1, cv2.LINE_AA)
    return img


# ── Cigarette fake detection box (for smoking image only) ─────

def draw_cigarette_box(img):
    """Draw a fake cigarette detection box on the tissue paper in hand."""
    h, w = img.shape[:2]
    # Tissue paper held in hand near mouth
    cx, cy = int(w * 0.42), int(h * 0.43)
    bw, bh = int(w * 0.13), int(h * 0.07)
    x1 = cx - bw // 2
    y1 = cy - bh // 2
    x2 = cx + bw // 2
    y2 = cy + bh // 2
    _draw_bracket_box(img, x1, y1, x2, y2,
                      (55, 55, 255), "Cigarette", 0.89, thickness=6)
    return img


# ── HUD overlays ──────────────────────────────────────────────

def draw_hud(img, is_warning=False):
    """Draw top bar + bottom data panel HUD."""
    h, w = img.shape[:2]

    # Slight darken + vignette
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # ── Top bar ──
    bar_h = 38
    cv2.rectangle(img, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(img, 1.0, img, 0.0, 0, img)
    # Re-draw top bar with alpha
    top_overlay = img.copy()
    cv2.rectangle(top_overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(top_overlay, 0.7, img, 0.3, 0, img)

    cv2.putText(img, "AI VISION SYSTEM v2.1", (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 210), 1,
                cv2.LINE_AA)
    # REC dot
    cv2.circle(img, (w - 125, 19), 5, (40, 40, 255), -1, cv2.LINE_AA)
    cv2.putText(img, "FPS: 28.3", (w - 110, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 220, 255), 1,
                cv2.LINE_AA)
    cv2.line(img, (0, bar_h), (w, bar_h), (50, 255, 210), 1)

    # ── Bottom panel ──
    ph = 105
    py = h - ph

    bottom = img.copy()
    cv2.rectangle(bottom, (0, py), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bottom, 0.78, img, 0.22, 0, img)
    cv2.line(img, (0, py), (w, py), (50, 255, 210), 1)

    if not is_warning:
        cv2.putText(img, "# FATIGUE MONITOR", (20, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 210), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "EAR: 0.32", (20, py + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1,
                    cv2.LINE_AA)

        # EAR bar
        bar_x, bar_y = 140, py + 37
        bar_w, bar_h = 120, 12
        cv2.rectangle(img, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (48, 42, 40), -1)
        cv2.rectangle(img, (bar_x, bar_y),
                      (bar_x + int(bar_w * 0.82), bar_y + bar_h),
                      (0, 255, 120), -1)
        cv2.putText(img, "SAFE", (bar_x + bar_w + 10, py + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 120), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "Drowsiness: NOT DETECTED", (20, py + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1,
                    cv2.LINE_AA)

        # Right half
        rx = w // 2 + 20
        cv2.line(img, (w // 2, py + 10), (w // 2, h - 10), (60, 55, 50), 1)
        cv2.putText(img, "# VIOLATION CHECK", (rx, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 210), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "No Violation Detected", (rx, py + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "All systems operating normally", (rx, py + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 85, 80), 1,
                    cv2.LINE_AA)
    else:
        # Warning bottom panel
        warn_bar = img.copy()
        cv2.rectangle(warn_bar, (0, py), (w, py + 32), (40, 40, 255), -1)
        cv2.addWeighted(warn_bar, 0.2, img, 0.8, 0, img)

        cv2.putText(img, "!! VIOLATION DETECTED - SMOKING", (20, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "Detection: Cigarette | Confidence: 89%",
                    (20, py + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "Action: SPEED LIMITED TO 30 km/h", (20, py + 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 2,
                    cv2.LINE_AA)

    return img


# ── Corner-bracket bounding box ───────────────────────────────

def _draw_bracket_box(img, x1, y1, x2, y2, color, label, conf, thickness=3):
    """Draw detection box with corner brackets + label tag."""
    bw = x2 - x1
    bh = y2 - y1
    corner = int(min(bw, bh) * 0.15)
    corner = max(corner, 12)

    # Dashed rectangle (simulate with dotted)
    for i in range(0, bw, 8):
        cv2.line(img, (x1 + i, y1), (x1 + min(i + 4, bw), y1), color, 1)
        cv2.line(img, (x1 + i, y2), (x1 + min(i + 4, bw), y2), color, 1)
    for i in range(0, bh, 8):
        cv2.line(img, (x1, y1 + i), (x1, y1 + min(i + 4, bh)), color, 1)
        cv2.line(img, (x2, y1 + i), (x2, y1 + min(i + 4, bh)), color, 1)

    # Corner brackets
    t = thickness
    # Top-left
    cv2.line(img, (x1, y1), (x1 + corner, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner), color, t, cv2.LINE_AA)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - corner, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner), color, t, cv2.LINE_AA)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + corner, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner), color, t, cv2.LINE_AA)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - corner, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner), color, t, cv2.LINE_AA)

    # Label tag
    label_text = f"{label}: {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(img, label_text, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────

def process_image(src_path, out_path, is_warning=False):
    img = cv2.imread(src_path)
    if img is None:
        print(f"  ERROR: cannot read {src_path}")
        return

    print(f"  Processing {os.path.basename(src_path)}...")
    print("    Running MediaPipe Face Mesh...")
    img = draw_face_mesh(img)
    print("    Running YOLOv8-Pose...")
    img = draw_yolo_pose(img)

    if is_warning:
        print("    Adding cigarette detection box...")
        img = draw_cigarette_box(img)

    print("    Drawing HUD...")
    img = draw_hud(img, is_warning)

    cv2.imwrite(out_path, img)
    print(f"    Saved: {out_path}")


if __name__ == "__main__":
    process_image(
        os.path.join(BASE, "normal.jpg"),
        os.path.join(BASE, "normal_overlay.png"),
        is_warning=False
    )
    process_image(
        os.path.join(BASE, "smoking.jpg"),
        os.path.join(BASE, "smoking_overlay.png"),
        is_warning=True
    )
    print("\nDone!")
