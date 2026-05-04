"""
全域設定與常數。修改參數只需動此檔。
"""
import os
import numpy as np

# ── 攝影機 ──
CAMERA_INDEX         = 0
FACE_LANDMARKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
OUTPUT_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ── Queue ──
QUEUE_MAXSIZE = 4

# ── 校準 ──
CALIB_SECONDS    = 5.0
PITCH_CAM_OFFSET = 25.0   # 初始估計值，校準後動態更新

# ── 感測閾值 ──
YOLO_IMGSZ        = 640
YOLO_POSE_IMGSZ   = 416    # pose 模型推論解析度（原 640，降低以減少延遲）
YOLO_PHONE_CONF   = 0.20   # 手機偵測門檻（專用模型 precision 較高，可適度提高）
YOLO_CIG_CONF     = 0.30   # 香菸偵測門檻（basant18 模型 conf 偏低，空間過濾補償誤報）
CIG_MOUTH_RATIO   = 1.2    # cig BBox 中心距嘴部距離 ≤ 臉寬 × 此倍數才計入
CIG_INTERVAL_SEC  = 0.3   # 香菸模型最短推論間隔（秒）；time-based 以避免丟幀造成間隔拉長
WRIST_MOUTH_RATIO    = 0.55
PHONE_MAX_AREA_RATIO = 0.10   # BBox 面積超過畫面 10% 視為誤判（如軀幹）
PHONE_SQUARE_MIN     = 0.7    # 長寬比介於此區間視為近正方形（軀幹誤判）
PHONE_SQUARE_MAX     = 1.4
YAW_PITCH_LIMIT   = 45.0
PITCH_PHONE_LIMIT = 32.0
EAR_THRESHOLD     = 0.20
FUSE_TIME_WINDOW  = 0.1

# ── 計時器（秒）──
DURATION_DISTRACT    = 2.0
DURATION_SMOKE       = 2.0
DURATION_SMOKE_NOCIG = 3.0
DURATION_PHONE       = 0.15
DURATION_FATIGUE     = 1.5

# ── Grace period（秒）──
HOLD_DISTRACT = 0.5
HOLD_PHONE    = 0.4
HOLD_FATIGUE  = 0.5
HOLD_SMOKE    = 1.0

# ── CSV ──
RECORD_CSV = True

# ── MediaPipe 臉部關鍵點索引 ──
LEFT_EYE_IDX     = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX    = [362, 385, 387, 263, 373, 380]
MOUTH_TOP_IDX    = 13
MOUTH_BOTTOM_IDX = 14
FACE_2D_IDX      = [1, 152, 33, 263, 61, 291]

# ── 近似 3D 臉型點（毫米單位，solvePnP 用）──
FACE_3D_POINTS = np.array([
    [  0.0,    0.0,    0.0],
    [  0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)
