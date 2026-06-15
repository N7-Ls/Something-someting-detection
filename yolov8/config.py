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
YOLO_POSE_IMGSZ   = 416    # pose 模型推論解析度（原 640，降低以減少延遲）
YOLO_PHONE_IMGSZ  = 320    # 手機偵測解析度（ROI 與全圖掃描皆用此值減少延遲）
YOLO_PHONE_CONF   = 0.30   # 手機偵測門檻（ROI 模式下空間已限制，可適度放低）
YOLO_CIG_CONF     = 0.38   # 香菸偵測門檻
YOLO_CIG_IMGSZ    = 320    # 香菸模型推論解析度（降低以減少延遲）
CIG_MOUTH_RATIO   = 1.2    # cig BBox 中心距嘴部距離 ≤ 臉寬 × 此倍數才計入
CIG_INTERVAL_SEC  = 0.3    # 香菸模型最短推論間隔（秒）
WRIST_MOUTH_RATIO = 0.55
PHONE_ROI_PAD     = 0.22   # 手腕 ROI 半徑（佔畫面最短邊比例）；手機通常在手腕周圍這個範圍
YAW_PITCH_LIMIT   = 45.0
PITCH_PHONE_LIMIT = 20.0
EAR_THRESHOLD     = 0.27   # 校準失敗時的 fallback 預設值
EAR_THRESHOLD_RATIO = 0.80 # 動態閾值 = 校準期間量到的睜眼基準 EAR 中位數 × 此比例
EAR_THRESHOLD_MIN   = 0.12 # 動態閾值下限，避免基準值異常過低
EAR_THRESHOLD_MAX   = 0.30 # 動態閾值上限，避免基準值異常過高
FUSE_TIME_WINDOW  = 0.1

# ── PERCLOS 疲勞偵測參數 ──
PERCLOS_WINDOW_SEC = 3.0   # 滾動窗口長度（秒）
PERCLOS_THRESHOLD  = 0.50  # 窗口內超過此比例的幀數 EAR < EAR_THRESHOLD 即判定疲勞

# ── 計時器（秒）──
DURATION_DISTRACT    = 2.0
DURATION_SMOKE       = 2.0
DURATION_SMOKE_NOCIG = 3.0
DURATION_PHONE       = 0.30
DURATION_FATIGUE     = 0.0   # PERCLOS 已提供時序過濾，不再需要額外持續時間

# ── Grace period（秒）──
HOLD_DISTRACT = 0.5
HOLD_PHONE    = 0.6
HOLD_FATIGUE  = 0.8   # 避免 PERCLOS 比率短暫回升就立即解除（秒）
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
