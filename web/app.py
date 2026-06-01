"""
駕駛安全監控系統 – Flask Web 入口
流程：① 人臉辨識  →  ② 安全帽確認  →  ③ 行駛危險偵測
"""
import os
import sys
import json
import time
import base64
import logging
import tempfile
import threading

import cv2
import numpy as np
import requests as http
from PIL import Image
import pillow_heif
from flask import Flask, Response, jsonify, render_template, request

pillow_heif.register_heif_opener()

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)  # 讓 yolov8n-pose.pt 等相對路徑正確解析

sys.path.insert(0, os.path.join(BASE_DIR, "ResNet"))
sys.path.insert(0, os.path.join(BASE_DIR, "Helmet"))
sys.path.insert(0, os.path.join(BASE_DIR, "yolov8"))

# ── 匯入各模組 ────────────────────────────────────────────────────────────────
from uniface import RetinaFace, ArcFace
from preprocess_head import crop_head

import state as yolo_state
from display import annotate as yolo_annotate
from thread_capture import thread_capture
from thread_yolo import thread_yolo
from thread_mediapipe import thread_mediapipe
from thread_decision import thread_decision
from utils import put_nowait_safe

logging.basicConfig(level=logging.INFO, format="[%(threadName)s] %(message)s")
app = Flask(__name__)

# ── 常數 ──────────────────────────────────────────────────────────────────────
FEATURE_PATH      = os.path.join(BASE_DIR, "ResNet", "user_feature.npy")
SIMILARITY_THRESH = 0.4

HELMET_API_URL = "https://uncommutatively-unpersuadable-an.ngrok-free.dev/api/chat"
HELMET_API_KEY = "upiceollama"
HELMET_PROMPT  = (
    "Step 1: Is the person wearing a helmet? If no, output 'Result: No helmet'.\n\n"
    "Step 2: Is the chin strap buckle clipped shut under the chin?\n"
    "- FASTENED: the buckle is closed and the strap connects both sides under the chin.\n"
    "- UNFASTENED: the buckle is open, or the straps hang separately on the sides.\n\n"
    "Output exactly one of:\n"
    "- 'Result: No helmet'\n"
    "- 'Result: Helmet on, strap fastened'\n"
    "- 'Result: Helmet on, strap unfastened'"
)

# ── 應用程式狀態 ──────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "phase":         "face_verify",   # face_verify | helmet_check | driving
    "message":       "請對準鏡頭，點「開始驗證」確認身份",
    "analyzing":     False,
    "face_verified": False,
    "helmet_status": None,            # no_helmet | unfastened | fastened
    "alert_level":   0,
}

# ── 攝影機管理 ────────────────────────────────────────────────────────────────
_frame_lock    = threading.Lock()
_current_frame = None
_cam_stop      = threading.Event()
_cam_released  = threading.Event()
_cam_released.set()  # 初始狀態視為已釋放


def _camera_worker():
    global _current_frame
    _cam_released.clear()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("無法開啟攝影機")
        _cam_released.set()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logging.info("攝影機已啟動")

    while not _cam_stop.is_set():
        ret, frame = cap.read()
        if ret:
            with _frame_lock:
                _current_frame = frame.copy()
        time.sleep(0.033)

    cap.release()
    _cam_released.set()
    logging.info("攝影機已釋放")


def _start_camera():
    _cam_stop.clear()
    threading.Thread(target=_camera_worker, daemon=True, name="Camera").start()


def _stop_camera(timeout=2.0):
    _cam_released.clear()
    _cam_stop.set()
    _cam_released.wait(timeout=timeout)


_start_camera()

# ── 人臉辨識輔助 ──────────────────────────────────────────────────────────────

def _cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def _verify_frame(frame):
    """比對單幀與 user_feature.npy。回傳 (ok: bool, sim: float)。"""
    if not os.path.exists(FEATURE_PATH):
        return False, 0.0
    base       = np.load(FEATURE_PATH)
    detector   = None
    recognizer = None
    try:
        detector   = RetinaFace()
        recognizer = ArcFace()
        faces      = detector.detect(frame)
        if not faces:
            return False, 0.0
        feat = recognizer.get_normalized_embedding(frame, faces[0].landmarks)
        sim  = _cosine_sim(feat, base)
        return sim >= SIMILARITY_THRESH, sim
    except Exception as e:
        logging.error("face verify error: %s", e)
        return False, 0.0
    finally:
        del detector, recognizer

# ── 安全帽偵測輔助 ────────────────────────────────────────────────────────────

def _frame_to_b64(bgr):
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()


def _analyze_helmet(frame):
    """裁切頭部後呼叫 LLM API。回傳 'no_helmet' | 'unfastened' | 'fastened'。"""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, frame)
    cropped = None
    try:
        cropped_path = crop_head(tmp_path)
        cropped      = cv2.imread(cropped_path)
        if cropped_path != tmp_path and os.path.exists(cropped_path):
            os.remove(cropped_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    img = cropped if cropped is not None else frame
    payload = {
        "model":    "gemma3:27b",
        "messages": [{"role": "user", "content": HELMET_PROMPT,
                      "images": [_frame_to_b64(img)]}],
        "stream":     False,
        "keep_alive": 0,
        "options":  {"temperature": 0.0},
    }
    resp   = http.post(HELMET_API_URL,
                       headers={"Content-Type": "application/json",
                                "X-API-Key":    HELMET_API_KEY},
                       json=payload, timeout=300)
    resp.raise_for_status()
    answer = resp.json()["message"]["content"]
    lower  = answer.lower()
    if "no helmet" in lower:
        return "no_helmet"
    if "strap fastened" in lower:
        return "fastened"
    return "unfastened"

# ── MJPEG 串流 ────────────────────────────────────────────────────────────────

def _mjpeg_generator():
    fps_time, fps_count, fps_val = time.time(), 0, 0.0

    while True:
        with _lock:
            phase = _state["phase"]

        # Phase 3: 從 yolov8 queue_display 讀已標注幀
        if phase == "driving":
            try:
                packet = yolo_state.queue_display.get(timeout=0.5)
            except Exception:
                continue
            if packet is None:
                continue
            fid, frame = packet

            fps_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_val   = fps_count / (now - fps_time)
                fps_count = 0
                fps_time  = now

            with yolo_state.display_lock:
                snap = {k: (v.copy() if isinstance(v, list) else v)
                        for k, v in yolo_state.display_state.items()}

            with _lock:
                _state["alert_level"] = snap.get("alert_level", 0)

            annotated = yolo_annotate(frame, snap, fps_val, fid)

        # Phase 1 & 2: 從 Flask 攝影機讀幀
        else:
            with _frame_lock:
                frame = _current_frame.copy() if _current_frame is not None else None
            if frame is None:
                time.sleep(0.033)
                continue
            annotated = frame

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")
        time.sleep(0.033)

# ── 路由 ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    registered = os.path.exists(FEATURE_PATH)
    return render_template("index.html", registered=registered)


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    with _lock:
        return jsonify(_state.copy())


@app.route("/api/verify_face", methods=["POST"])
def api_verify_face():
    with _lock:
        if _state["analyzing"] or _state["phase"] != "face_verify":
            return jsonify({"error": "not ready"}), 400
        _state["analyzing"] = True
        _state["message"]   = "人臉辨識中，請保持正面..."

    def _run():
        with _frame_lock:
            frame = _current_frame.copy() if _current_frame is not None else None
        if frame is None:
            with _lock:
                _state["analyzing"] = False
                _state["message"]   = "無法取得畫面，請重試"
            return
        ok, sim = _verify_frame(frame)
        with _lock:
            _state["analyzing"] = False
            if ok:
                _state["face_verified"] = True
                _state["phase"]         = "helmet_check"
                _state["message"]       = "✓ 人臉辨識成功！請佩戴好安全帽後點「偵測安全帽」"
            else:
                _state["message"] = f"驗證失敗（相似度 {sim:.2f}），請重試"

    threading.Thread(target=_run, daemon=True, name="FaceVerify").start()
    return jsonify({"status": "started"})


@app.route("/api/check_helmet", methods=["POST"])
def api_check_helmet():
    with _lock:
        if _state["analyzing"] or _state["phase"] != "helmet_check":
            return jsonify({"error": "not ready"}), 400
        _state["analyzing"] = True
        _state["message"]   = "安全帽分析中，請稍候..."

    def _run():
        with _frame_lock:
            frame = _current_frame.copy() if _current_frame is not None else None
        if frame is None:
            with _lock:
                _state["analyzing"] = False
                _state["message"]   = "無法取得畫面，請重試"
            return
        try:
            label = _analyze_helmet(frame)
        except Exception as e:
            logging.error("helmet api error: %s", e)
            with _lock:
                _state["analyzing"] = False
                _state["message"]   = f"API 錯誤：{e}"
            return

        msg_map = {
            "fastened":   "✓ 安全帽已正確佩戴，下巴帶扣好！可點「開始行駛監控」",
            "unfastened": "安全帽已戴但下巴帶未扣，請扣好後再偵測",
            "no_helmet":  "未偵測到安全帽，請佩戴後再偵測",
        }
        with _lock:
            _state["analyzing"]     = False
            _state["helmet_status"] = label
            _state["message"]       = msg_map.get(label, label)

    threading.Thread(target=_run, daemon=True, name="HelmetCheck").start()
    return jsonify({"status": "started"})


@app.route("/api/start_driving", methods=["POST"])
def api_start_driving():
    with _lock:
        if _state["phase"] != "helmet_check":
            return jsonify({"error": "wrong phase"}), 400
        if _state.get("helmet_status") != "fastened":
            return jsonify({"error": "helmet not fastened"}), 400
        _state["phase"]   = "driving"
        _state["message"] = "行駛監控中"

    # 讓 Flask 釋放攝影機，yolov8 thread_capture 再接手
    _stop_camera()

    # 重置 yolov8 共享狀態
    yolo_state.stop_event.clear()
    for q in (yolo_state.queue_pose, yolo_state.queue_face,
              yolo_state.queue_decision, yolo_state.queue_display):
        while not q.empty():
            try:
                q.get_nowait()
            except Exception:
                break

    for target, name in [
        (thread_capture,   "Capture"),
        (thread_yolo,      "YOLOv8"),
        (thread_mediapipe, "MediaPipe"),
        (thread_decision,  "Decision"),
    ]:
        threading.Thread(target=target, name=name, daemon=True).start()

    return jsonify({"status": "driving"})


@app.route("/api/stop_driving", methods=["POST"])
def api_stop_driving():
    yolo_state.stop_event.set()
    put_nowait_safe(yolo_state.queue_display, None)

    # 重啟 Flask 攝影機
    _start_camera()

    with _lock:
        _state.update({
            "phase":         "face_verify",
            "message":       "請對準鏡頭，點「開始驗證」確認身份",
            "analyzing":     False,
            "face_verified": False,
            "helmet_status": None,
            "alert_level":   0,
        })
    return jsonify({"status": "stopped"})


@app.route("/api/register", methods=["POST"])
def api_register():
    """接收上傳的照片，提取人臉特徵並儲存至 user_feature.npy。支援 HEIC/HEIF。"""
    if "photo" not in request.files:
        return jsonify({"error": "no photo"}), 400
    try:
        pil_img = Image.open(request.files["photo"]).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return jsonify({"error": "invalid image"}), 400

    detector = recognizer = None
    try:
        detector   = RetinaFace()
        recognizer = ArcFace()
        faces      = detector.detect(img)
        if not faces:
            return jsonify({"error": "no face detected"}), 400
        if len(faces) > 1:
            return jsonify({"error": "multiple faces detected"}), 400
        feat = recognizer.get_normalized_embedding(img, faces[0].landmarks)
        np.save(FEATURE_PATH, feat)
        return jsonify({"status": "registered"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        del detector, recognizer


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
