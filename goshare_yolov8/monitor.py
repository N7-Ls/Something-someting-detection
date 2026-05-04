"""
4.4 多工視覺融合與駕駛行為監控模組
架構：5 條執行緒 + 4 條 Queue
"""

import csv
import datetime
import os
import cv2
import mediapipe as mp
import numpy as np
import queue
import threading
import time
import logging
from ultralytics import YOLO
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="[%(threadName)s] %(message)s")

# ─── 全域設定 ────────────────────────────────────────────────────────────────
CAMERA_INDEX       = 0
QUEUE_MAXSIZE      = 4
FACE_LANDMARKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CALIB_SECONDS  = 5.0    # 啟動後自動校準時間（秒）；期間請保持正常騎乘/直視前方姿勢
RECORD_CSV     = True   # 自動記錄感測數值 CSV（用於分析與校調門檻）
WRIST_MOUTH_RATIO  = 0.55   # 手腕─嘴部距離閾值（相對臉寬比例，與解析度無關）
DURATION_SMOKE_NOCIG = 3.0  # 無香菸模型時，僅靠手腕靠嘴判斷的持續秒數（較保守）
YAW_PITCH_LIMIT    = 45      # 視線偏轉安全閾值 (度)
PITCH_PHONE_LIMIT  = 32.0   # 低頭滑手機輔助判斷閾值 (度，正值=向下，以校正後 pitch 比較)
PITCH_CAM_OFFSET   = 25.0   # 攝影機仰角補償 (度)：龍頭安裝時鏡頭朝上約 25°，
                             # 正常騎乘頭部水平但 solvePnP 量測值偏正，故需扣除此偏移
                             # corr_pitch = raw_pitch - PITCH_CAM_OFFSET
                             # 眼平安裝（桌機/測試）設為 0.0
EAR_THRESHOLD      = 0.20   # 疲勞 EAR 閾值（CSV 顯示睜眼 median≈0.28，0.25 太靠近中心）
FUSE_TIME_WINDOW   = 0.1    # 跨模組時序融合容忍窗口 (秒)
YOLO_IMGSZ         = 320    # YOLO 推論輸入解析度（320 比 640 快約 4x）

DURATION_DISTRACT  = 2.0
DURATION_SMOKE     = 2.0
DURATION_PHONE     = 0.15  # 單幀 YOLO 偵測約 0.2s 可見，改短以確保能累積到閾值
DURATION_FATIGUE   = 1.5

# 條件消失後計時器的「保持秒數」，防止單幀中斷導致誤重置
HOLD_DISTRACT = 0.5
HOLD_PHONE    = 0.4   # phone 短暫消失仍維持警示
HOLD_FATIGUE  = 0.5
HOLD_SMOKE    = 1.0

YOLO_PHONE_CONF    = 0.15  # 手機偵測信心度門檻（低一點，handlebar 視角容易漏報）

# ── Arduino 設定 ──
ARDUINO_PORT = None   # Windows: "COM3"；Jetson: "/dev/ttyUSB0"；None = 純軟體模式
ARDUINO_BAUD = 9600

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_TOP_IDX    = 13
MOUTH_BOTTOM_IDX = 14

FACE_3D_POINTS = np.array([
    [0.0,    0.0,    0.0   ],
    [0.0,   -330.0, -65.0 ],
    [-225.0,  170.0, -135.0],
    [225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)
FACE_2D_IDX = [1, 152, 33, 263, 61, 291]

cig_model_available = False   # 由 thread_yolo 載入成功後設為 True

# ─── 動態攝影機仰角補償（啟動時自動校準，C 鍵可重置）────────────────────────
_cam_offset      = PITCH_CAM_OFFSET   # 執行期實際使用的補償值，由校準後更新
_cam_offset_lock = threading.Lock()
recalib_event    = threading.Event()  # 主執行緒按 C 時 set，決策緒收到後重新校準

def get_cam_offset():
    with _cam_offset_lock:
        return _cam_offset

def set_cam_offset(v):
    global _cam_offset
    with _cam_offset_lock:
        _cam_offset = v

# ─── Queue & 共享狀態 ────────────────────────────────────────────────────────
queue_pose     = queue.Queue(maxsize=QUEUE_MAXSIZE)
queue_face     = queue.Queue(maxsize=QUEUE_MAXSIZE)
queue_decision = queue.Queue(maxsize=QUEUE_MAXSIZE * 2)
queue_display  = queue.Queue(maxsize=2)           # 原始幀供顯示執行緒
stop_event     = threading.Event()

# 各執行緒更新此字典，顯示執行緒讀取（lock 保護）
display_state = {
    # YOLO
    "phone_boxes": [],   # [(x1,y1,x2,y2,conf), ...]
    "cig_boxes":   [],
    "wrist_xy":    [],   # [(x,y), ...]
    # MediaPipe
    "face_detected": False,
    "mp_frames":     0,   # MediaPipe 已處理幀數
    "ear_val":  None,
    "yaw":      None,
    "pitch":    None,
    "roll":     None,
    "mouth_xy": None,
    "face_pts": [],      # 眼角/嘴角關鍵點 [(x,y), ...]
    "mesh_landmarks": None,  # 全臉 468 點供繪製 face mesh
    "face_width": None,      # 臉部寬度（px），用於相對距離閾值計算
    "wrist_mouth_dist": None,  # 最近手腕距嘴部距離（debug 用）
    "pitch_corr": None,        # 校正後 pitch = raw_pitch - PITCH_CAM_OFFSET（決策緒更新）
    # Decision
    "alert_level": 0,    # 0=正常 1=提示 2=違規 3=危險
    "alert_msg":   "",
    "calib_status": f"校準中… {CALIB_SECONDS:.0f}s",  # 顯示於畫面頂部
}
display_lock = threading.Lock()

arduino_ctrl: "ArduinoController | None" = None   # 由 main() 初始化後賦值


# ─── 顏色常數 ────────────────────────────────────────────────────────────────
LEVEL_COLOR = {
    0: (0, 200, 0),      # 綠
    1: (0, 200, 255),    # 黃
    2: (0, 120, 255),    # 橘
    3: (0, 0, 255),      # 紅
}


# ─── 工具函式 ────────────────────────────────────────────────────────────────
def ear(landmarks, eye_idx, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_idx])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def head_pose(landmarks, w, h):
    pts_2d = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in FACE_2D_IDX],
        dtype=np.float64,
    )
    focal = w
    cam_matrix = np.array(
        [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64
    )
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(
        FACE_3D_POINTS, pts_2d, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    # cv2.RQDecomp3x3 直接回傳度（degrees），不需再 × 360
    # 常見網路教學有 × 360 的錯誤寫法，實測會使角度爆炸成萬級
    return angles[1], angles[0], angles[2]   # yaw, pitch, roll (degrees)


def pixel_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def put_nowait_safe(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def wrap_angle(a):
    """將角度正規化到 (-180, +180]，解決 solvePnP/RQDecomp 在 ±180° 跳變的問題。"""
    return ((a + 180.0) % 360.0) - 180.0


# ─── Arduino 控制器 ───────────────────────────────────────────────────────────
class ArduinoController:
    """
    透過 Serial 傳送等級指令給 Arduino。
    協議：'L0\\n'=全滅  'L1\\n'=黃燈+短促音  'L2\\n'/'L3\\n'=紅燈+連續音
    無硬體或 pyserial 未安裝時自動 fallback（僅 print）。
    """
    def __init__(self, port=None, baud=9600):
        self._ser  = None
        self._lock = threading.Lock()
        if port and SERIAL_AVAILABLE:
            try:
                self._ser = serial.Serial(port, baud, timeout=1)
                time.sleep(2.0)   # 等待 Arduino reset
                logging.info(f"Arduino 已連線：{port} @ {baud}baud")
            except Exception as e:
                logging.warning(f"Arduino 連線失敗（{e}），切換至純軟體模式")
        elif port and not SERIAL_AVAILABLE:
            logging.warning("pyserial 未安裝（pip install pyserial），Arduino 功能停用")

    @property
    def connected(self):
        return self._ser is not None and self._ser.is_open

    def send(self, level: int):
        cmd = f"L{level}\n".encode()
        if self.connected:
            try:
                with self._lock:
                    self._ser.write(cmd)
            except Exception as e:
                logging.warning(f"Arduino 傳送失敗：{e}")
        # 無論是否連線皆 log，方便無硬體時確認邏輯正確
        logging.debug(f"[Arduino] 指令：L{level}")

    def close(self):
        if self.connected:
            try:
                self._ser.write(b"L0\n")
                self._ser.close()
            except Exception:
                pass


# ─── 畫面標註 ─────────────────────────────────────────────────────────────────
def annotate(frame, state, fps):
    h, w = frame.shape[:2]
    out = frame.copy()
    level = state["alert_level"]
    color = LEVEL_COLOR[level]

    # ── 外框顏色代表警戒等級 ──
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 4)

    # ── 手機 BBox（紅色）──
    for (x1, y1, x2, y2, conf) in state["phone_boxes"]:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(out, f"Phone {conf:.2f}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # ── 香菸 BBox（橘色）──
    for (x1, y1, x2, y2, conf) in state["cig_boxes"]:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 120, 255), 2)
        cv2.putText(out, f"Cig {conf:.2f}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)

    # ── 手腕關鍵點（藍色圓點）──
    for (wx, wy) in state["wrist_xy"]:
        cv2.circle(out, (int(wx), int(wy)), 8, (255, 80, 0), -1)
        cv2.putText(out, "Wrist", (int(wx) + 10, int(wy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)

    # ── 嘴部中心（紫色圓點）──
    if state["mouth_xy"]:
        mx, my = state["mouth_xy"]
        cv2.circle(out, (int(mx), int(my)), 6, (200, 0, 200), -1)

    # ── 全臉 Face Mesh（偵測到時用青色小點覆蓋整個網格）──
    mesh = state.get("mesh_landmarks")
    if mesh:
        for (px, py) in mesh:
            cv2.circle(out, (int(px), int(py)), 1, (0, 220, 200), -1)

    # ── 眼角/嘴角等關鍵點（白色較大點）──
    for (px, py) in state["face_pts"]:
        cv2.circle(out, (int(px), int(py)), 3, (255, 255, 255), -1)

    # ── 左側資訊面板 ──
    ear_v    = state["ear_val"]
    yaw_v    = state["yaw"]
    pitch_v  = state["pitch"]          # raw pitch（量測值，供 debug 參考）
    pitchc_v = state.get("pitch_corr") # 校正後 pitch（= raw - CAM_OFFSET，決策依據）
    roll_v   = state["roll"]

    # 各行文字與顏色（EAR 過低用紅字，角度過大用黃字）
    def ear_color(v):
        if v is None: return (180, 180, 180)
        return (0, 0, 255) if v < EAR_THRESHOLD else (0, 255, 80)

    def yaw_color(v):
        if v is None: return (180, 180, 180)
        return (0, 200, 255) if abs(v) > YAW_PITCH_LIMIT else (0, 255, 80)

    def pitchc_color(v):
        # 校正後 pitch：向下超過 PITCH_PHONE_LIMIT 或向上超過 YAW_PITCH_LIMIT 時標黃
        if v is None: return (180, 180, 180)
        if v > PITCH_PHONE_LIMIT or v < -YAW_PITCH_LIMIT:
            return (0, 200, 255)
        return (0, 255, 80)

    face_ok  = state.get("face_detected", False)
    mp_cnt   = state.get("mp_frames", 0)
    face_txt = f"Face : OK  (#{mp_cnt})" if face_ok else f"Face : ND  (#{mp_cnt})"
    face_clr = (0, 255, 80) if face_ok else (0, 80, 255)

    fw   = state.get("face_width")
    wmd  = state.get("wrist_mouth_dist")
    thr  = (fw * WRIST_MOUTH_RATIO) if fw else None

    def wmd_color(d, t):
        if d is None or t is None: return (180, 180, 180)
        return (0, 0, 255) if d < t else (0, 255, 80)

    wmd_txt = (f"W-M  : {wmd:.0f}/{thr:.0f}px" if wmd is not None and thr is not None
               else "W-M  : --")

    # Ptch* = 攝影機補償後的 pitch，是決策實際使用的數值
    pitchc_txt = (f"Ptch*: {pitchc_v:+.1f}" if pitchc_v is not None else "Ptch*: --")

    panel = [
        (face_txt,                                                             face_clr),
        (f"EAR  : {ear_v:.3f}" if ear_v   is not None else "EAR  : --",    ear_color(ear_v)),
        (f"Yaw  : {yaw_v:+.1f}"  if yaw_v  is not None else "Yaw  : --",   yaw_color(yaw_v)),
        (pitchc_txt,                                                           pitchc_color(pitchc_v)),
        (f"Roll : {roll_v:+.1f}" if roll_v is not None else "Roll : --",    (180, 180, 180)),
        (wmd_txt,                                                              wmd_color(wmd, thr)),
        (f"FPS  : {fps:.1f}",                                                 (200, 200, 200)),
    ]

    LINE_H   = 28
    PANEL_W  = 235
    PANEL_H  = len(panel) * LINE_H + 10
    MARGIN   = 8

    # 半透明黑底
    overlay = out.copy()
    cv2.rectangle(overlay, (MARGIN, MARGIN), (MARGIN + PANEL_W, MARGIN + PANEL_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    for i, (txt, txt_color) in enumerate(panel):
        y_pos = MARGIN + LINE_H * (i + 1)
        cv2.putText(out, txt, (MARGIN + 6, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)          # 黑色描邊
        cv2.putText(out, txt, (MARGIN + 6, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, txt_color, 1)           # 彩色文字

    # ── 頂部校準狀態橫幅 ──
    calib_txt = state.get("calib_status", "")
    if calib_txt:
        is_calibrating = calib_txt.startswith("校準中")
        banner_color = (0, 140, 255) if is_calibrating else (0, 160, 60)
        overlay3 = out.copy()
        cv2.rectangle(overlay3, (0, 0), (w, 28), banner_color, -1)
        cv2.addWeighted(overlay3, 0.75, out, 0.25, 0, out)
        cv2.putText(out, calib_txt, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 3)
        cv2.putText(out, calib_txt, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)

    # ── 底部警報訊息 ──
    if state["alert_msg"]:
        bar_h = 42
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, h - bar_h), (w, h), color, -1)
        cv2.addWeighted(overlay2, 0.65, out, 0.35, 0, out)
        cv2.putText(out, state["alert_msg"], (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4)           # 黑描邊
        cv2.putText(out, state["alert_msg"], (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)     # 白字

    # ── Level 2：右上角速限徽章（30 km/h）──
    if level == 2:
        cx, cy, r = w - 72, 72, 58
        cv2.circle(out, (cx, cy), r, (0, 0, 200), -1)
        cv2.circle(out, (cx, cy), r, (255, 255, 255), 3)
        cv2.putText(out, "30",    (cx - 26, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(out, "km/h", (cx - 28, cy + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # ── Level 3：全螢幕 DANGER 覆蓋 + 時速歸零（等效 PyQt5 全螢幕儀表板）──
    if level == 3:
        danger_overlay = out.copy()
        cv2.rectangle(danger_overlay, (0, 0), (w, h), (0, 0, 160), -1)
        cv2.addWeighted(danger_overlay, 0.40, out, 0.60, 0, out)
        # DANGER 大字（黑描邊 + 紅字）
        txt_danger = "DANGER"
        (tw, th), _ = cv2.getTextSize(txt_danger, cv2.FONT_HERSHEY_SIMPLEX, 2.8, 7)
        tx = (w - tw) // 2;  ty = h // 2 - 10
        cv2.putText(out, txt_danger, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 0, 0), 10)
        cv2.putText(out, txt_danger, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 0, 255), 7)
        # 時速歸零
        txt_speed = "0  km/h"
        (sw, _), _ = cv2.getTextSize(txt_speed, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
        cv2.putText(out, txt_speed, ((w - sw) // 2, ty + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
        cv2.putText(out, txt_speed, ((w - sw) // 2, ty + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

    return out


# ─── 執行緒 1：影像擷取 (Producer) ──────────────────────────────────────────
def thread_capture():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logging.error("無法開啟攝影機")
        stop_event.set()
        return

    frame_id = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.warning("讀取影格失敗，跳過")
                time.sleep(0.01)
                continue

            ts = time.time()
            packet = (frame_id, ts, frame)
            put_nowait_safe(queue_pose,    packet)
            put_nowait_safe(queue_face,    packet)
            put_nowait_safe(queue_display, frame)   # 原始幀供顯示
            frame_id += 1
    except Exception as e:
        logging.error(f"擷取執行緒例外：{e}")
        stop_event.set()
    finally:
        cap.release()
        for _ in range(2):
            put_nowait_safe(queue_pose, None)
            put_nowait_safe(queue_face, None)
        put_nowait_safe(queue_display, None)
        logging.info("影像擷取執行緒結束")


# ─── 執行緒 2：YOLOv8 (Consumer 1) ──────────────────────────────────────────
def thread_yolo():
    try:
        # pose 模型：只用來取手腕關鍵點
        model_pose = YOLO("yolov8n-pose.pt")
        # general 模型：偵測手機（class 67）與一般物件
        model_det  = YOLO("yolov8n.pt")
    except Exception as e:
        logging.error(f"YOLOv8 模型載入失敗：{e}")
        return

    # 自訓練香菸模型（選用）
    global cig_model_available
    try:
        model_cig = YOLO("yolov8n_cigarette.pt")
        cig_model_available = True
        logging.info("香菸模型載入成功")
    except Exception:
        logging.warning("yolov8n_cigarette.pt 未找到，改用手腕靠嘴判斷（持續 3 秒）")
        model_cig = None

    while not stop_event.is_set():
        try:
            packet = queue_pose.get(timeout=1.0)
        except queue.Empty:
            continue
        if packet is None:
            break

        frame_id, ts, frame = packet
        result = {
            "source": "yolo", "frame_id": frame_id, "timestamp": ts,
            "phone_detected": False, "cigarette_detected": False, "wrist_xy": None,
        }

        try:
            phone_boxes, wrists, cig_boxes = [], [], []

            # ── 通用偵測：手機（class 67）──
            res_det = model_det(frame, verbose=False, conf=YOLO_PHONE_CONF, imgsz=YOLO_IMGSZ)[0]
            if res_det.boxes is not None:
                for box in res_det.boxes:
                    if int(box.cls[0]) == 67:
                        result["phone_detected"] = True
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        phone_boxes.append((x1, y1, x2, y2, float(box.conf[0])))

            # ── 姿態偵測：手腕關鍵點（保留 640 精度，不降幀以確保手腕準確）──
            res_pose = model_pose(frame, verbose=False, imgsz=640)[0]
            if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                kps_xy   = res_pose.keypoints[0].xy.cpu().numpy()    # shape (17,2)
                kps_conf = res_pose.keypoints[0].conf.cpu().numpy()  # shape (17,)
                for idx in [9, 10]:  # 9=左手腕, 10=右手腕
                    if idx < len(kps_xy) and kps_conf[idx] > 0.3:   # 過濾低信心度
                        x, y = kps_xy[idx]
                        if x > 0 and y > 0:
                            wrists.append((float(x), float(y)))
                if wrists:
                    result["wrist_xy"] = wrists

            # ── 香菸偵測（自訓練模型，無模型時略過）──
            if cig_model_available and model_cig is not None:
                res_cig = model_cig(frame, verbose=False)[0]
                if res_cig.boxes is not None and len(res_cig.boxes) > 0:
                    result["cigarette_detected"] = True
                    for box in res_cig.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cig_boxes.append((x1, y1, x2, y2, float(box.conf[0])))

            with display_lock:
                display_state["phone_boxes"] = phone_boxes
                display_state["cig_boxes"]   = cig_boxes
                display_state["wrist_xy"]    = wrists

        except Exception as e:
            logging.error(f"YOLOv8 推論例外：{e}")

        put_nowait_safe(queue_decision, result)

    logging.info("YOLOv8 執行緒結束")


# ─── 執行緒 3：MediaPipe FaceLandmarker Tasks API (Consumer 2) ───────────────
def thread_mediapipe():
    # MediaPipe 0.10+ 使用 Tasks API
    BaseOptions         = mp.tasks.BaseOptions
    FaceLandmarker      = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOpts  = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode   = mp.tasks.vision.RunningMode

    try:
        options = FaceLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        landmarker = FaceLandmarker.create_from_options(options)
    except Exception as e:
        logging.error(f"MediaPipe 初始化失敗：{e}")
        return

    DRAW_IDX = LEFT_EYE_IDX + RIGHT_EYE_IDX + [1, 61, 291, 199]

    while not stop_event.is_set():
        try:
            packet = queue_face.get(timeout=1.0)
        except queue.Empty:
            continue
        if packet is None:
            break

        frame_id, ts, frame = packet
        result = {
            "source": "face", "frame_id": frame_id, "timestamp": ts,
            "ear_val": None, "yaw": None, "pitch": None, "roll": None,
            "mouth_xy": None, "face_width": None,
        }

        try:
            h, w = frame.shape[:2]

            with display_lock:
                display_state["mp_frames"] += 1

            frame_copy = frame.copy()
            rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_result = landmarker.detect(mp_image)

            if mp_result.face_landmarks:
                lm = mp_result.face_landmarks[0]   # List[NormalizedLandmark]

                ear_l = ear(lm, LEFT_EYE_IDX,  w, h)
                ear_r = ear(lm, RIGHT_EYE_IDX, w, h)
                result["ear_val"] = (ear_l + ear_r) / 2.0

                yaw, pitch, roll = head_pose(lm, w, h)
                result["yaw"], result["pitch"], result["roll"] = yaw, pitch, roll

                mx = (lm[MOUTH_TOP_IDX].x + lm[MOUTH_BOTTOM_IDX].x) / 2 * w
                my = (lm[MOUTH_TOP_IDX].y + lm[MOUTH_BOTTOM_IDX].y) / 2 * h
                result["mouth_xy"] = (float(mx), float(my))

                # 臉寬：左眼角(33) 到 右眼角(263) 距離 * 3（近似全臉寬）
                ex_l = lm[33].x  * w;  ex_r = lm[263].x * w
                face_w = abs(ex_r - ex_l) * 3.0
                result["face_width"] = face_w

                face_pts = [(lm[i].x * w, lm[i].y * h) for i in DRAW_IDX]
                all_pts  = [(lm[i].x * w, lm[i].y * h) for i in range(len(lm))]

                with display_lock:
                    display_state["face_detected"]  = True
                    display_state["ear_val"]        = result["ear_val"]
                    display_state["yaw"]            = yaw
                    display_state["pitch"]          = pitch
                    display_state["roll"]           = roll
                    display_state["mouth_xy"]       = result["mouth_xy"]
                    display_state["face_pts"]       = face_pts
                    display_state["mesh_landmarks"] = all_pts
                    display_state["face_width"]     = face_w
            else:
                with display_lock:
                    display_state["face_detected"]  = False
                    display_state["ear_val"]        = None
                    display_state["yaw"]            = None
                    display_state["pitch"]          = None
                    display_state["roll"]           = None
                    display_state["mouth_xy"]       = None
                    display_state["face_pts"]       = []
                    display_state["mesh_landmarks"] = None

        except Exception as e:
            logging.error(f"MediaPipe 推論例外：{e}")

        put_nowait_safe(queue_decision, result)

    landmarker.close()
    logging.info("MediaPipe 執行緒結束")


# ─── 執行緒 4：決策中心 ─────────────────────────────────────────────────────
def thread_decision():
    cache_yolo = {}
    cache_face = {}
    timers       = {"distract": None, "smoke": None, "phone": None, "fatigue": None}
    hold_timers  = {"distract": None, "smoke": None, "phone": None, "fatigue": None}
    prev_level   = 0   # 追蹤上一個等級，等級變化時才發 Arduino 指令
    HOLD_MAP     = {
        "distract": HOLD_DISTRACT,
        "smoke":    HOLD_SMOKE,
        "phone":    HOLD_PHONE,
        "fatigue":  HOLD_FATIGUE,
    }

    def check_duration(key, condition, required_sec):
        """計時器加入 grace period：條件消失後仍保持 HOLD_MAP[key] 秒再重置。"""
        now = time.time()
        if condition:
            hold_timers[key] = None          # 取消待重置計時
            if timers[key] is None:
                timers[key] = now
            return (now - timers[key]) >= required_sec
        else:
            if timers[key] is None:
                return False                 # 從未開始，直接回傳 False
            # 條件剛變 False：啟動 hold 計時
            if hold_timers[key] is None:
                hold_timers[key] = now
            # Hold 期滿 → 真正重置主計時器
            if (now - hold_timers[key]) >= HOLD_MAP[key]:
                timers[key] = None
                hold_timers[key] = None
                return False
            # Hold 期間：視同條件仍成立
            return (now - timers[key]) >= required_sec

    # ── 自動校準（收集正常騎乘的 pitch 中位數作為 cam offset）──
    calib_pitches = []
    calib_start   = time.time()
    calibrated    = False

    def _finish_calibration():
        if len(calib_pitches) >= 10:
            new_off = float(np.median(calib_pitches))
            set_cam_offset(new_off)
            status = f"校準完成：offset={new_off:+.1f}° (C 鍵重新校準)"
            logging.info(f"攝影機補償校準完成：{new_off:.1f}°（樣本數 {len(calib_pitches)}）")
        else:
            new_off = get_cam_offset()
            status = f"校準樣本不足，沿用預設 {new_off:+.1f}° (C 鍵重新校準)"
            logging.warning("校準樣本不足，使用預設 PITCH_CAM_OFFSET")
        with display_lock:
            display_state["calib_status"] = status

    # ── CSV 記錄設定 ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"monitor_{ts_str}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp",
        "raw_pitch", "corr_pitch", "yaw", "roll", "ear",
        "cam_offset",
        "phone_bbox", "phone_gaze", "phone_wrist",
        "smoke", "fatigue", "distract",
        "alert_level", "alert_msg",
    ])
    logging.info(f"感測數值記錄至：{csv_path}")

    try:
        while not stop_event.is_set():
            # ── 重新校準請求（C 鍵觸發）──
            if recalib_event.is_set():
                recalib_event.clear()
                calib_pitches.clear()
                calib_start = time.time()
                calibrated  = False
                with display_lock:
                    display_state["calib_status"] = f"重新校準中… {CALIB_SECONDS:.0f}s"
                logging.info("重新校準開始")

            try:
                result = queue_decision.get(timeout=1.0)
            except queue.Empty:
                # 校準計時即使沒資料也要推進
                if not calibrated and (time.time() - calib_start) >= CALIB_SECONDS:
                    _finish_calibration()
                    calibrated = True
                continue
            if result is None:
                break

            fid = result["frame_id"]
            if result["source"] == "yolo":
                cache_yolo[fid] = result
            else:
                cache_face[fid] = result

            for c in (cache_yolo, cache_face):
                if len(c) > 30:
                    del c[min(c.keys())]

            # YOLO：優先取同 frame_id，否則用最新一筆（限 1s 內，與 face 策略一致）
            yolo = cache_yolo.get(fid)
            if yolo is None and cache_yolo:
                latest = cache_yolo[max(cache_yolo.keys())]
                if time.time() - latest["timestamp"] < 1.0:
                    yolo = latest
            # face 取最新一筆（兩模型速度不同，不強制同 frame_id）
            face = (cache_face.get(fid) or
                    (cache_face[max(cache_face.keys())] if cache_face else None))

            # ── 校準期間：收集 raw pitch；校準結束後計算中位數 ──
            raw_pitch = face["pitch"] if (face and face["pitch"] is not None) else None
            if not calibrated:
                remaining = CALIB_SECONDS - (time.time() - calib_start)
                if raw_pitch is not None:
                    calib_pitches.append(raw_pitch)
                if remaining <= 0:
                    _finish_calibration()
                    calibrated = True
                else:
                    with display_lock:
                        display_state["calib_status"] = f"校準中… {remaining:.0f}s（保持正常姿勢）"

            # ── 攝影機仰角補償（使用校準後的動態 offset）──
            cam_off    = get_cam_offset()
            # 校正步驟說明：
            # 1. raw - cam_off：去除攝影機仰角造成的系統性偏移
            # 2. wrap_angle()：正規化到 (-180, +180]，消除 ±180° 邊界跳變（-350° 等）
            # 3. 取負號：solvePnP 在此座標系中低頭使 raw 減少（差值為負），
            #            取負後 正值=低頭/滑手機、負值=抬頭/分心，符合判斷邏輯
            corr_pitch = (-wrap_angle(raw_pitch - cam_off)
                          if raw_pitch is not None else None)

            with display_lock:
                display_state["pitch_corr"] = corr_pitch

            # ── 視線分心：側轉（|Yaw|>閾值）或向上看（corr_pitch 負值超過閾值）──
            # 刻意排除向下低頭（corr_pitch>0），向下已由手機偵測涵蓋，不重複計為分心
            distract_cond = (
                face is not None and face["yaw"] is not None and corr_pitch is not None
                and (abs(face["yaw"]) > YAW_PITCH_LIMIT or corr_pitch < -YAW_PITCH_LIMIT)
            )
            phone_by_bbox = yolo is not None and yolo["phone_detected"]
            phone_by_gaze = corr_pitch is not None and corr_pitch > PITCH_PHONE_LIMIT
            phone_by_wrist = False
            if yolo and face and yolo["wrist_xy"] and face["mouth_xy"]:
                face_y = face["mouth_xy"][1]
                for wx, wy in yolo["wrist_xy"]:
                    if wy < face_y * 1.25:
                        phone_by_wrist = True
                        break
            phone_cond = phone_by_bbox or phone_by_gaze or phone_by_wrist
            fatigue_cond = (
                face is not None and face["ear_val"] is not None
                and face["ear_val"] < EAR_THRESHOLD
            )
            smoke_cond = False
            wrist_mouth_dist_min = None
            if yolo and face:
                if abs(yolo["timestamp"] - face["timestamp"]) <= FUSE_TIME_WINDOW:
                    if yolo["wrist_xy"] and face["mouth_xy"] and face.get("face_width"):
                        threshold_px   = face["face_width"] * WRIST_MOUTH_RATIO
                        # 垂直容忍：手腕 Y 必須在嘴部 Y ±0.35×臉寬範圍內
                        # 避免「撥髮／擦額頭」等手腕高於嘴部的動作誤觸
                        vert_thr = face["face_width"] * 0.35
                        mouth_y  = face["mouth_xy"][1]
                        all_dists = [pixel_dist(w, face["mouth_xy"]) for w in yolo["wrist_xy"]]
                        wrist_mouth_dist_min = min(all_dists)
                        # 同時滿足「水平+垂直」雙重距離才算靠嘴
                        wrist_close = any(
                            pixel_dist(w, face["mouth_xy"]) < threshold_px
                            and abs(w[1] - mouth_y) < vert_thr
                            for w in yolo["wrist_xy"]
                        )
                        if cig_model_available:
                            smoke_cond = yolo["cigarette_detected"] and wrist_close
                        else:
                            smoke_cond = wrist_close

            with display_lock:
                display_state["wrist_mouth_dist"] = wrist_mouth_dist_min

            smoke_duration = DURATION_SMOKE if cig_model_available else DURATION_SMOKE_NOCIG
            trig_fatigue  = check_duration("fatigue",  fatigue_cond,  DURATION_FATIGUE)
            trig_smoke    = check_duration("smoke",    smoke_cond,    smoke_duration)
            trig_phone    = check_duration("phone",    phone_cond,    DURATION_PHONE)
            trig_distract = check_duration("distract", distract_cond, DURATION_DISTRACT)

            if trig_fatigue:
                level, msg = 3, "[介入] Level 3: 疲勞駕駛，緊急喚醒，時速歸零"
            elif trig_smoke:
                level, msg = 2, "[介入] Level 2: 抽菸違規，鎖定時速 30km/h"
            elif trig_phone:
                level = 2
                if phone_by_bbox:
                    msg = "[介入] Level 2: 滑手機（畫面偵測），鎖定時速 30km/h"
                elif phone_by_wrist:
                    msg = "[介入] Level 2: 手腕舉至臉部（疑似持機），鎖定時速 30km/h"
                else:
                    msg = "[介入] Level 2: 疑似低頭滑手機，鎖定時速 30km/h"
            elif trig_distract:
                level, msg = 1, "[介入] Level 1: 視線分心，發出提示音與黃燈"
            else:
                level, msg = 0, ""

            if msg:
                print(msg)

            # ── Arduino：等級變化時才傳送指令（避免串列埠洗版）──
            if level != prev_level:
                prev_level = level
                if arduino_ctrl is not None:
                    arduino_ctrl.send(level)

            with display_lock:
                display_state["alert_level"] = level
                display_state["alert_msg"]   = msg

            # ── CSV 寫入（每個決策週期一行）──
            if RECORD_CSV:
                csv_writer.writerow([
                    f"{time.time():.3f}",
                    f"{raw_pitch:.2f}"   if raw_pitch   is not None else "",
                    f"{corr_pitch:.2f}"  if corr_pitch  is not None else "",
                    f"{face['yaw']:.2f}" if face and face["yaw"]     is not None else "",
                    f"{face['roll']:.2f}"if face and face["roll"]    is not None else "",
                    f"{face['ear_val']:.4f}" if face and face["ear_val"] is not None else "",
                    f"{cam_off:.1f}",
                    int(phone_by_bbox), int(phone_by_gaze), int(phone_by_wrist),
                    int(smoke_cond), int(fatigue_cond), int(distract_cond),
                    level, msg,
                ])

    finally:
        csv_file.close()
        logging.info(f"CSV 已儲存：{csv_path}")
        logging.info("決策中心執行緒結束")


# ─── 主程式（含顯示迴圈，必須在主執行緒）────────────────────────────────────
def main():
    global arduino_ctrl
    arduino_ctrl = ArduinoController(ARDUINO_PORT, ARDUINO_BAUD)

    threads = [
        threading.Thread(target=thread_capture,  name="Capture",   daemon=True),
        threading.Thread(target=thread_yolo,      name="YOLOv8",    daemon=True),
        threading.Thread(target=thread_mediapipe, name="MediaPipe", daemon=True),
        threading.Thread(target=thread_decision,  name="Decision",  daemon=True),
    ]
    for t in threads:
        t.start()

    fps_counter  = 0
    fps_time     = time.time()
    fps_display  = 0.0
    video_writer = None   # cv2.VideoWriter，R 鍵切換

    try:
        while not stop_event.is_set():
            try:
                frame = queue_display.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break

            # FPS 計算
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_display = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time    = now

            # 讀取最新狀態並繪製
            with display_lock:
                state_snap = {k: (v.copy() if isinstance(v, list) else v)
                              for k, v in display_state.items()}

            annotated = annotate(frame, state_snap, fps_display)

            # ── 錄影寫幀 ──
            if video_writer is not None:
                video_writer.write(annotated)
                # 錄影指示燈（右上角紅點 + REC）
                vh, vw = annotated.shape[:2]
                cv2.circle(annotated, (vw - 22, 18), 9, (0, 0, 220), -1)
                cv2.putText(annotated, "REC", (vw - 60, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 220), 2)

            cv2.imshow("GoShare Driver Monitor", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:       # Q / ESC：結束
                stop_event.set()
            elif key == ord("r") or key == ord("R"):  # R：切換錄影
                if video_writer is None:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    vts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    vpath = os.path.join(OUTPUT_DIR, f"monitor_{vts}.mp4")
                    vh, vw = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(vpath, fourcc, 15, (vw, vh))
                    logging.info(f"開始錄影：{vpath}")
                else:
                    video_writer.release()
                    video_writer = None
                    logging.info("停止錄影")
            elif key == ord("c") or key == ord("C"):  # C：重新校準
                recalib_event.set()

    except KeyboardInterrupt:
        logging.info("收到 KeyboardInterrupt，正在關閉...")
        stop_event.set()
    finally:
        if video_writer is not None:
            video_writer.release()
            logging.info("錄影檔案已儲存")

    # sentinel 清場
    for _ in range(2):
        put_nowait_safe(queue_pose,     None)
        put_nowait_safe(queue_face,     None)
        put_nowait_safe(queue_decision, None)

    for t in threads:
        t.join(timeout=5.0)

    if arduino_ctrl is not None:
        arduino_ctrl.close()

    cv2.destroyAllWindows()
    logging.info("程式結束")


if __name__ == "__main__":
    main()
