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
                         主執行緒 UI (cv2 / PyQt5)
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
goshare/
├── yolov8/                  # 核心監控模組（多執行緒）
│   ├── main.py
│   ├── config.py
│   ├── state.py
│   ├── utils.py
│   ├── display.py
│   ├── thread_capture.py
│   ├── thread_yolo.py
│   ├── thread_mediapipe.py
│   ├── thread_decision.py
│   ├── monitor.py           # 舊版單檔（不使用）
│   ├── face_landmarker.task # MediaPipe 人臉模型（LFS）
│   ├── yolov8n.pt           # YOLOv8n 通用偵測模型（LFS）
│   ├── yolov8n-pose.pt      # YOLOv8n 姿態偵測模型（LFS）
│   └── yolov8n_phone.pt     # 手機專用自訓練模型（LFS）
├── PyQT5/
│   └── dashboard.py         # PyQt5 圖形儀表板
├── ResNet/
│   ├── register_user.py     # 人臉特徵向量註冊
│   └── verify_user.py       # 人臉身分驗證
├── MediaPipe/
│   └── helmet_detection.py  # 安全帽偵測（實驗性）
├── llava_anlmage_27b_2/
│   └── llava_anlmage_27b_2.py  # LLaVA 多模態影像分析、用來快速辨識是否配戴安全帽
├── pics/                    # UI 示意圖與 overlay 素材
├── requirements.txt
└── README.md
```

---

## 各檔案功能

### `yolov8/` — 核心監控模組

| 檔案 | 功能 |
|------|------|
| `main.py` | 程式入口，啟動 4 條執行緒，主執行緒以 `cv2.imshow` 顯示並處理鍵盤輸入（R 錄影、C 校準、Q/ESC 結束） |
| `config.py` | 所有可調參數的集中定義（攝影機索引、閾值、計時器、grace period、YOLO 解析度等），**修改參數只需動此檔** |
| `state.py` | 跨執行緒共享狀態：4 條 Queue、`display_state` 字典、Lock、Event，以及攝影機仰角動態補償的 getter/setter |
| `utils.py` | 純工具函式（無副作用）：EAR 計算、solvePnP 頭姿估計、像素距離、`wrap_angle` 角度正規化、`put_nowait_safe` |
| `display.py` | 畫面標註渲染：根據 `display_state` 快照在影格上繪製偵測框、關鍵點、資訊面板、警報橫幅、Level 2 速限徽章、Level 3 DANGER 覆蓋 |
| `thread_capture.py` | 影像擷取執行緒（Producer）：從攝影機讀幀並推送至 `queue_pose`、`queue_face`、`queue_display`，Queue 滿時丟幀 |
| `thread_yolo.py` | YOLOv8 推論執行緒：通用模型偵測手機（class 67）、pose 模型取手腕關鍵點（idx 9/10）、自訓練模型偵測香菸 |
| `thread_mediapipe.py` | MediaPipe Tasks API 推論執行緒：計算雙眼 EAR（疲勞）、solvePnP 頭姿（Yaw/Pitch/Roll）、嘴部中心座標、臉寬估計 |
| `thread_decision.py` | 決策中心：自動攝影機仰角校準（啟動 5s 收集 pitch 中位數）、時序融合（±0.1s）、各危險行為計時器 + grace period、4 級警報判斷、Arduino 指令、CSV 記錄 |
| `monitor.py` | 舊版單檔實作（含 Arduino 完整邏輯），已被拆分為上述多執行緒架構取代，**不再使用** |

### `PyQT5/`

| 檔案 | 功能 |
|------|------|
| `dashboard.py` | PyQt5 圖形儀表板，取代 `main.py` 的 `cv2.imshow`。直接匯入 `yolov8/` 四執行緒，以 QTimer(33ms) 刷新影像與右側狀態面板（LevelBar、4 個違規指示燈、感測數值列表、校準/校準按鈕） |

### `ResNet/` — 人臉身分驗證（Jetson Nano JetPack 4.6 / Python 3.6 相容）

| 檔案 | 功能 |
|------|------|
| `register_user.py` | 從指定照片提取人臉特徵向量（ArcFace），確認唯一人臉後儲存為 `user_feature.npy` |
| `verify_user.py` | 開啟 Webcam 擷取單幀，與 `user_feature.npy` 做餘弦相似度比對（門檻 0.4），驗證後立即釋放模型記憶體 |

### `MediaPipe/`

| 檔案 | 功能 |
|------|------|
| `helmet_detection.py` | 安全帽偵測實驗模型（MediaPipe Custom Model） |
| `train.py` | 安全帽偵測模型訓練腳本 |

### `llava_anlmage_27b_2/`

| 檔案 | 功能 |
|------|------|
| `llava_anlmage_27b_2.py` | 使用 LLaVA 27B 多模態大語言模型對行車影像進行語意分析、用來快速辨識是否配戴安全帽，需要連接伺服器 |

---

## 使用模組

| 模組 | 用途 |
|------|------|
| `opencv-python` | 影像讀取、繪圖、solvePnP 頭姿估計、影片錄製 |
| `numpy` | 數值計算（EAR、頭姿矩陣、特徵向量） |
| `mediapipe` | 人臉 468 點 Landmark（EAR、嘴部、頭姿）、Tasks API |
| `ultralytics` | YOLOv8 物件偵測與姿態偵測（手機、香菸、手腕） |
| `tensorflow` | MediaPipe 後端依賴 |
| `PyQt5` | 圖形儀表板（VideoWidget、狀態面板、指示燈） |
| `uniface` | ArcFace / RetinaFace 人臉特徵提取與偵測（ResNet 模組） |
| `deepface` / `retinaface` | 人臉辨識備用依賴 |
| `requests` | HTTP 通訊（遠端上報預留） |
| `matplotlib` | 感測數值視覺化分析 |
| `pyserial` | Arduino 序列埠通訊（可選，警報硬體輸出） |

---

## 快速開始

### 環境安裝

```bash
pip install -r requirements.txt
```

### 啟動（cv2 模式）

```bash
cd yolov8
python main.py
```

### 啟動（PyQt5 儀表板）

```bash
cd PyQT5
python dashboard.py
```

### 鍵盤操作

| 按鍵 | 動作 |
|------|------|
| `R` | 開始 / 停止錄影（儲存至 `yolov8/output/`） |
| `C` | 重新校準攝影機仰角補償 |
| `Q` / `ESC` | 結束程式 |

---

## 系統需求

- Python 3.10+（ResNet 模組支援 3.6+）
- CUDA GPU 建議（CPU 亦可運行，FPS 較低）
- 部署目標：NVIDIA Jetson Nano（JetPack 4.6）
- 攝影機：USB Webcam 或 CSI Camera（index 0）

---

## 輸出

- **CSV**：每個決策週期一行，記錄 raw_pitch、corr_pitch、yaw、roll、EAR、各偵測旗標、alert_level，自動儲存於 `yolov8/output/monitor_YYYYMMDD_HHMMSS.csv`
- **影片**：R 鍵觸發，mp4 格式儲存於同目錄