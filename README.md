# GoShare Driver Monitor

機車駕駛行為即時監控系統，部署於 NVIDIA Jetson Nano，整合 YOLOv8、MediaPipe 與 PyQt5，偵測疲勞、滑手機、抽菸、視線分心等危險駕駛行為並即時介入。

---

## 系統架構

```
攝影機
  ├─ queue_pose  → thread_yolo      (手機 / 手腕 / 香菸)
  └─ queue_face  → thread_mediapipe (EAR / 頭姿 / 嘴部)
                       ↓
                 queue_decision → thread_decision (融合 / 計時 / CSV)
                                      ↓
                                 display_state (Lock)
                                      ↓
                         主執行緒 UI (PyQt5 dashboard)

Helmet/
  攝影機 → preprocess_head (YOLOv8 pose 頭部裁切)
               → camera_helmet_detect (Gemma3:27B API 安全帽判斷)
```

---

## 警報等級

| Level | 顏色 | 條件 | 介入動作 |
|-------|------|------|----------|
| 0 | 綠 | 正常 | — |
| 1 | 黃 | 視線偏轉 (\|Yaw\| > 45°) 持續 2s | 提示音 |
| 2 | 橘 | 滑手機 / 抽菸達門檻 | 鎖速 30 km/h |
| 3 | 紅 | EAR < 0.20 持續 1.5s | 緊急喚醒、時速歸零 |

---

## 檔案結構

```
Goshare/
├── yolov8/                      # 核心監控模組（多執行緒）
│   ├── main.py                  # cv2 模式入口（開發用）
│   ├── config.py                # 所有可調參數
│   ├── state.py                 # 跨執行緒共享狀態
│   ├── utils.py                 # 純工具函式
│   ├── display.py               # 畫面標註渲染
│   ├── thread_capture.py        # 影像擷取執行緒
│   ├── thread_yolo.py           # YOLOv8 推論執行緒
│   ├── thread_mediapipe.py      # MediaPipe 推論執行緒
│   ├── thread_decision.py       # 決策 / 計時 / CSV 執行緒
│   ├── face_landmarker.task     # MediaPipe 人臉模型
│   ├── yolov8n-pose.pt          # YOLOv8n 姿態偵測模型
│   ├── yolov8s.pt               # YOLOv8s 手機偵測模型（COCO）
│   └── yolov8n_cigarette.pt     # 香菸自訓練模型（HF: basant18/Smoking-detection-YOLO26s）
├── PyQT5/
│   └── dashboard.py             # PyQt5 圖形儀表板（主要執行入口）
├── Helmet/
│   ├── camera_helmet_detect.py  # 即時安全帽偵測（Gemma3:27B API）
│   ├── preprocess_head.py       # YOLOv8 pose 頭部裁切前處理
│   └── yolov8n-pose.pt          # 姿態模型（Helmet 模組獨立使用）
├── ResNet/
│   ├── register_user.py         # 人臉特徵向量註冊
│   └── verify_user.py           # 人臉身分驗證
├── MediaPipe/
│   ├── helmet_detection.py      # 安全帽偵測（實驗性）
│   └── train.py                 # 安全帽偵測模型訓練腳本
├── pics/                        # UI 示意圖與 overlay 素材
├── requirements.txt
└── README.md
```

---

## 各模組說明

### `yolov8/` — 核心監控模組

| 檔案 | 功能 |
|------|------|
| `main.py` | 程式入口（cv2 模式），啟動 4 條執行緒，主執行緒以 `cv2.imshow` 顯示（開發 / 測試用） |
| `config.py` | 所有可調參數的集中定義（攝影機索引、閾值、計時器、grace period、YOLO 解析度等），**修改參數只需動此檔** |
| `state.py` | 跨執行緒共享狀態：4 條 Queue、`display_state` 字典、Lock、Event |
| `utils.py` | 純工具函式（無副作用）：EAR 計算、solvePnP 頭姿估計、`wrap_angle`、`put_nowait_safe` |
| `display.py` | 畫面標註渲染：偵測框、關鍵點、資訊面板、警報橫幅、Level 色塊 |
| `thread_capture.py` | 影像擷取（Producer）：推送至 `queue_pose`、`queue_face`、`queue_display`，Queue 滿時丟幀 |
| `thread_yolo.py` | YOLOv8 推論：pose 模型取手腕關鍵點 → 手機 ROI 偵測（yolov8s.pt class 67）；定時香菸偵測（yolov8n_cigarette.pt）|
| `thread_mediapipe.py` | MediaPipe Tasks API：雙眼 EAR（疲勞）、solvePnP 頭姿（Yaw/Pitch/Roll）、嘴部中心、臉寬估計 |
| `thread_decision.py` | 決策中心：攝影機仰角自動校準（啟動 5s）、時序融合（±0.1s）、各行為計時器 + grace period、4 級警報、CSV 記錄 |

### `PyQT5/` — 圖形儀表板（主要執行入口）

| 檔案 | 功能 |
|------|------|
| `dashboard.py` | 取代 `main.py` 的 `cv2.imshow`。匯入 `yolov8/` 四執行緒，QTimer(33ms) 刷新影像與右側狀態面板（LevelBar、4 個違規指示卡片、感測數值、校準按鈕） |

### `Helmet/` — 安全帽偵測模組

| 檔案 | 功能 |
|------|------|
| `preprocess_head.py` | 使用 YOLOv8n-pose 偵測人物，取肩膀關鍵點定位頭部範圍，裁切出頭頸區域（含安全帽），支援 HEIC/HEIF 輸入 |
| `camera_helmet_detect.py` | 開啟鏡頭，每 8 秒裁切一次頭部後傳送至 Gemma3:27B API 判斷：`Helmet on, strap fastened` / `Helmet on, strap unfastened` / `No helmet`，結果疊加至畫面左上角 |

### `ResNet/` — 人臉身分驗證

| 檔案 | 功能 |
|------|------|
| `register_user.py` | 從照片提取 ArcFace 特徵向量，確認唯一人臉後儲存為 `user_feature.npy` |
| `verify_user.py` | 擷取單幀與 `user_feature.npy` 做餘弦相似度比對（門檻 0.4），驗證駕駛身分 |

### `MediaPipe/` — 安全帽偵測（實驗性）

| 檔案 | 功能 |
|------|------|
| `helmet_detection.py` | MediaPipe Custom Model 安全帽偵測（實驗性，非主要流程） |
| `train.py` | 安全帽偵測模型訓練腳本（需 TensorFlow） |

---

## 使用套件

| 套件 | 用途 |
|------|------|
| `opencv-contrib-python-headless` | 影像讀取、繪圖、solvePnP 頭姿估計（headless，不含 Qt）|
| `numpy` | 數值計算（EAR、頭姿矩陣、特徵向量） |
| `mediapipe` | 人臉 468 點 Landmark（EAR、嘴部、頭姿）、Tasks API |
| `ultralytics` | YOLOv8 物件偵測與姿態偵測（手機、香菸、手腕、頭部裁切） |
| `PyQt5` | 圖形儀表板（VideoWidget、狀態面板、指示燈） |
| `pillow-heif` | Helmet 模組 HEIC/HEIF 圖片讀取支援 |
| `uniface` | ArcFace / RetinaFace 人臉特徵提取（ResNet 模組） |
| `requests` | Helmet 模組呼叫 Gemma3:27B API |
| `huggingface-hub` | 香菸模型自動下載（basant18/Smoking-detection-YOLO26s） |
| `pyserial` | Arduino 序列埠通訊（可選，警報硬體輸出） |

---

## 快速開始

### 環境安裝（Jetson Nano / linux-aarch64）

```bash
# PyQt5 需透過 conda 安裝
conda install "pyqt=5.15.11" -n goshare

# 其他套件
pip install -r requirements.txt
```

### 啟動 PyQt5 儀表板（主要入口）

```bash
cd PyQT5
QT_QPA_PLATFORM_PLUGIN_PATH=~/miniconda3/envs/goshare/plugins python dashboard.py
```

或將環境變數寫入 `~/.bashrc` 後直接執行：

```bash
echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=~/miniconda3/envs/goshare/plugins' >> ~/.bashrc
source ~/.bashrc

cd PyQT5
python dashboard.py
```

### 啟動 cv2 模式（開發 / 測試）

```bash
cd yolov8
python main.py
```

### 啟動安全帽偵測

```bash
cd Helmet
python camera_helmet_detect.py
```

### 鍵盤操作（dashboard / main.py）

| 按鍵 | 動作 |
|------|------|
| `C` | 重新校準攝影機仰角補償 |
| `Q` / `ESC` | 結束程式 |

---

## 系統需求

- Python 3.12（conda 環境 `goshare`）
- 部署目標：NVIDIA Jetson Nano（linux-aarch64）
- 攝影機：USB Webcam 或 CSI Camera（index 0）
- Helmet 模組：需連線至 Gemma3:27B API 伺服器

---

## 輸出

- **CSV**：每個決策週期一行，記錄 raw_pitch、corr_pitch、yaw、roll、EAR、各偵測旗標、alert_level，自動儲存於 `yolov8/output/monitor_YYYYMMDD_HHMMSS.csv`
