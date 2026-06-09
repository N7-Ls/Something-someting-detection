import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[警告] TensorFlow 未安裝，分類器以隨機模式運行")

# ─── 常數設定 ───────────────────────────────────────────
DATA_DIR = "dataset"
ROI_SIZE = (96, 96)

# MediaPipe landmark indices
TOP_LANDMARK  = 10    # 前額頂部
LEFT_EYE      = 33
RIGHT_EYE     = 263
CHIN_LANDMARK = 152   # 下巴底部

# ─── Arduino 預留函式 ──────────────────────────────────
def send_to_arduino(status: str):
    """
    預留 Arduino 通訊介面
    status: "UNLOCK" | "LOCK"
    """
    pass

# ─── 資料夾初始化 ──────────────────────────────────────
def init_dataset_dirs():
    dirs = [
        f"{DATA_DIR}/helmet/positive",
        f"{DATA_DIR}/helmet/negative",
        f"{DATA_DIR}/strap/positive",
        f"{DATA_DIR}/strap/negative",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# ─── ROI 計算 ──────────────────────────────────────────
def get_head_top_roi(landmarks, frame_w, frame_h):
    """根據前額頂部與眼角距離，動態計算頭頂 ROI（隨頭部轉動位移）"""
    top      = landmarks[TOP_LANDMARK]
    left_eye = landmarks[LEFT_EYE]
    right_eye= landmarks[RIGHT_EYE]

    top_x = int(top.x * frame_w)
    top_y = int(top.y * frame_h)

    eye_dist = abs(right_eye.x - left_eye.x) * frame_w
    roi_w = int(eye_dist * 1.4)
    roi_h = int(eye_dist * 1.0)

    x1 = max(0, top_x - roi_w // 2)
    y1 = max(0, top_y - roi_h)
    x2 = min(frame_w, x1 + roi_w)
    y2 = min(frame_h, top_y + int(roi_h * 0.2))

    return x1, y1, x2, y2

def get_chin_roi(landmarks, frame_w, frame_h):
    """根據下巴 landmark，動態計算下巴繩帶 ROI"""
    chin     = landmarks[CHIN_LANDMARK]
    left_eye = landmarks[LEFT_EYE]
    right_eye= landmarks[RIGHT_EYE]

    chin_x = int(chin.x * frame_w)
    chin_y = int(chin.y * frame_h)

    eye_dist = abs(right_eye.x - left_eye.x) * frame_w
    roi_w = int(eye_dist * 1.2)
    roi_h = int(eye_dist * 0.7)

    x1 = max(0, chin_x - roi_w // 2)
    y1 = max(0, chin_y - int(roi_h * 0.2))
    x2 = min(frame_w, x1 + roi_w)
    y2 = min(frame_h, chin_y + roi_h)

    return x1, y1, x2, y2

# ─── 分類器 ────────────────────────────────────────────
class DummyClassifier:
    """訓練完成前的佔位分類器，輸出固定 0.5"""
    def predict(self, roi_img):
        return 0.5, 0.5  # (positive_prob, negative_prob)

class MobileNetV2Classifier:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, roi_img):
        img = cv2.resize(roi_img, ROI_SIZE).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        probs = self.model.predict(img, verbose=0)[0]
        return float(probs[0]), float(probs[1])  # positive, negative

# ─── 畫面繪製 ──────────────────────────────────────────
def draw_roi_box(frame, x1, y1, x2, y2, color, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_status(frame, face_detected, helmet_prob, strap_prob, collect_mode, collect_target):
    h, w = frame.shape[:2]

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    mode_label = "[收資料]" if collect_mode else "[偵測]"
    cv2.putText(frame, f"MODE: {mode_label}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    if not face_detected:
        cv2.putText(frame, "FACE: 未偵測到", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "STATUS: 鎖定", (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return

    helmet_ok = helmet_prob > 0.5
    strap_ok  = strap_prob  > 0.5

    hc = (0, 255, 0) if helmet_ok else (0, 0, 255)
    sc = (0, 255, 0) if strap_ok  else (0, 0, 255)

    cv2.putText(frame, f"安全帽: {'是' if helmet_ok else '否'} ({helmet_prob:.2f})",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hc, 1)
    cv2.putText(frame, f"繩帶扣: {'是' if strap_ok  else '否'} ({strap_prob:.2f})",
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 1)

    if helmet_ok and strap_ok:
        cv2.putText(frame, "STATUS: 解鎖", (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        reason = "未戴安全帽" if not helmet_ok else "繩帶未扣"
        cv2.putText(frame, f"STATUS: 鎖定 ({reason})", (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 收資料模式提示
    if collect_mode:
        target_labels = {
            "helmet_pos": "收集: 頭頂安全帽 (正)",
            "helmet_neg": "收集: 頭頂無帽   (負)",
            "strap_pos":  "收集: 繩帶已扣   (正)",
            "strap_neg":  "收集: 繩帶未扣   (負)",
        }
        cv2.putText(frame, target_labels[collect_target], (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# ─── 主程式 ────────────────────────────────────────────
def main():
    init_dataset_dirs()

    # 載入分類器（模型存在才啟用）
    helmet_clf = DummyClassifier()
    strap_clf  = DummyClassifier()

    if TF_AVAILABLE:
        if os.path.exists("model_helmet.h5"):
            helmet_clf = MobileNetV2Classifier("model_helmet.h5")
            print("[INFO] 安全帽分類器已載入")
        else:
            print("[INFO] model_helmet.h5 不存在，使用 DummyClassifier")
        if os.path.exists("model_strap.h5"):
            strap_clf = MobileNetV2Classifier("model_strap.h5")
            print("[INFO] 繩帶分類器已載入")
        else:
            print("[INFO] model_strap.h5 不存在，使用 DummyClassifier")

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing   = mp.solutions.drawing_utils
    mp_styles    = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[錯誤] 無法開啟攝影機")
        return

    collect_mode   = False
    collect_target = "helmet_pos"
    last_save_time = 0.0
    save_interval  = 0.5  # 每 0.5 秒自動存一張

    print("\n=== 安全帽防偽偵測系統 ===")
    print("[C]   切換收資料 / 偵測模式")
    print("[1]   收集：頭頂安全帽 (正樣本)")
    print("[2]   收集：頭頂無安全帽 (負樣本)")
    print("[3]   收集：繩帶已扣 (正樣本)")
    print("[4]   收集：繩帶未扣 (負樣本)")
    print("[Q]   離開\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        face_detected = False
        helmet_prob   = 0.0
        strap_prob    = 0.0

        if results.multi_face_landmarks:
            face_detected = True
            lm = results.multi_face_landmarks[0].landmark

            # 畫面網格特徵點
            mp_drawing.draw_landmarks(
                frame,
                results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )

            # 動態 ROI
            hx1, hy1, hx2, hy2 = get_head_top_roi(lm, w, h)
            sx1, sy1, sx2, sy2 = get_chin_roi(lm, w, h)

            helmet_roi = frame[hy1:hy2, hx1:hx2]
            strap_roi  = frame[sy1:sy2, sx1:sx2]

            if helmet_roi.size > 0:
                helmet_prob, _ = helmet_clf.predict(helmet_roi)
            if strap_roi.size > 0:
                strap_prob, _ = strap_clf.predict(strap_roi)

            # 畫 ROI 框
            hc = (0, 255, 0) if helmet_prob > 0.5 else (0, 165, 255)
            sc = (0, 255, 0) if strap_prob  > 0.5 else (0, 165, 255)
            draw_roi_box(frame, hx1, hy1, hx2, hy2, hc, "Helmet ROI")
            draw_roi_box(frame, sx1, sy1, sx2, sy2, sc, "Strap ROI")

            # 自動存圖（收資料模式）
            if collect_mode:
                now = time.time()
                if now - last_save_time > save_interval:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path_map = {
                        "helmet_pos": (helmet_roi, f"{DATA_DIR}/helmet/positive/{ts}.jpg"),
                        "helmet_neg": (helmet_roi, f"{DATA_DIR}/helmet/negative/{ts}.jpg"),
                        "strap_pos":  (strap_roi,  f"{DATA_DIR}/strap/positive/{ts}.jpg"),
                        "strap_neg":  (strap_roi,  f"{DATA_DIR}/strap/negative/{ts}.jpg"),
                    }
                    roi_img, save_path = path_map[collect_target]
                    if roi_img.size > 0:
                        cv2.imwrite(save_path, roi_img)
                        last_save_time = now
                        print(f"[SAVED] {save_path}")

            # 控制邏輯
            if helmet_prob > 0.5 and strap_prob > 0.5:
                send_to_arduino("UNLOCK")
            else:
                send_to_arduino("LOCK")

        draw_status(frame, face_detected, helmet_prob, strap_prob,
                    collect_mode, collect_target)

        cv2.imshow("Helmet Detection System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            collect_mode = not collect_mode
            print(f"[MODE] {'收資料模式 ON' if collect_mode else '偵測模式 ON'}")
        elif key == ord('1'):
            collect_target, collect_mode = "helmet_pos", True
            print("[收集] 頭頂安全帽正樣本")
        elif key == ord('2'):
            collect_target, collect_mode = "helmet_neg", True
            print("[收集] 頭頂無安全帽負樣本")
        elif key == ord('3'):
            collect_target, collect_mode = "strap_pos", True
            print("[收集] 繩帶已扣正樣本")
        elif key == ord('4'):
            collect_target, collect_mode = "strap_neg", True
            print("[收集] 繩帶未扣負樣本")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
