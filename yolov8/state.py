"""
跨執行緒共享狀態：Queue、display_state、lock、event。
所有執行緒只從此模組讀寫共享資料，不互相直接引用。
"""
import queue
import threading
from config import PITCH_CAM_OFFSET, QUEUE_MAXSIZE, CALIB_SECONDS

# ── Queue ──
queue_pose     = queue.Queue(maxsize=QUEUE_MAXSIZE)
queue_face     = queue.Queue(maxsize=QUEUE_MAXSIZE)
queue_decision = queue.Queue(maxsize=QUEUE_MAXSIZE * 2)
queue_display  = queue.Queue(maxsize=2)

# ── 控制事件 ──
stop_event    = threading.Event()
recalib_event = threading.Event()   # 主執行緒按 C 時 set，決策緒重新校準

# ── 攝影機仰角補償（執行期動態更新）──
_cam_offset      = PITCH_CAM_OFFSET
_cam_offset_lock = threading.Lock()

def get_cam_offset() -> float:
    with _cam_offset_lock:
        return _cam_offset

def set_cam_offset(v: float):
    global _cam_offset
    with _cam_offset_lock:
        _cam_offset = v

# ── 香菸模型可用旗標（由 thread_yolo 載入後設為 True）──
cig_model_available = False

# ── 顯示狀態（主執行緒讀取、各執行緒更新）──
display_state = {
    # YOLO
    "phone_boxes": [],
    "cig_boxes":   [],
    "wrist_xy":    [],
    # MediaPipe
    "face_detected":    False,
    "mp_frames":        0,
    "ear_val":          None,
    "yaw":              None,
    "pitch":            None,
    "roll":             None,
    "mouth_xy":         None,
    "face_pts":         [],
    "mesh_landmarks":   None,
    "face_width":       None,
    "wrist_mouth_dist": None,
    "pitch_corr":       None,
    "perclos":          0.0,
    # 決策
    "alert_level":  0,
    "alert_msg":    "",
    "calib_status": f"校準中… {CALIB_SECONDS:.0f}s",
    # 影格同步追蹤（建議 1）
    "yolo_frame_id": 0,
    "face_frame_id": 0,
    # 各違規項目觸發旗標（由 thread_decision 寫出，供 PyQt5 儀表板讀取）
    "alert_flags": {"phone": False, "smoke": False, "fatigue": False, "distract": False},
}
display_lock = threading.Lock()
