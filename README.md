# GoShare Driver Monitor

機車駕駛行為即時監控系統，部署於 NVIDIA Jetson Nano，整合 ResNet 人臉辨識、YOLOv8、MediaPipe、LLM 安全帽判斷與 Flask Web 介面，提供「身份驗證 → 安全帽確認 → 行駛監控」三階段一條龍流程，偵測疲勞、滑手機、抽菸、視線分心等危險駕駛行為並即時介入。

---

## 系統架構

### 整合式 Web 流程（主要入口：`web/app.py`）

瀏覽器透過 MJPEG 串流即時顯示攝影機畫面，三階段共用同一支 Flask App 與同一顆鏡頭：

```
① 人臉辨識            ② 安全帽確認                 ③ 行駛監控
ResNet (RetinaFace +   Helmet/preprocess_head        yolov8 四執行緒管線
ArcFace) 比對          (YOLOv8 pose 頭部裁切)        （見下方）
user_feature.npy       → LLM API (Gemma3:27B)             ↓
       │               判斷：未戴 / 戴but未扣 / 已扣好    顯示已標註畫面
       └───────────────────────┴─────────────────────→ MJPEG 推送至瀏覽器
                          （Flask 釋放鏡頭 → thread_capture 接手）
```

### yolov8 即時監控管線（行駛監控階段核心，亦可獨立執行）

```
攝影機
  ├─ queue_pose  → thread_yolo      (手機 / 手腕 / 香菸)
  └─ queue_face  → thread_mediapipe (EAR / 頭姿 / 嘴部)
                       ↓
                 queue_decision → thread_decision (融合 / 計時 / CSV)
                                      ↓
                                 display_state (Lock)
                                      ↓
                queue_display → Web MJPEG 串流 / PyQt5 dashboard / cv2 視窗
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
├── web/                         # ★ 整合式 Web 介面（主要展示入口）
│   ├── app.py                   # Flask App：人臉辨識 → 安全帽確認 → 行駛監控（MJPEG 串流）
│   └── templates/
│       └── index.html           # 三步驟流程頁面（深色卡片式 UI）
├── yolov8/                      # 核心監控模組（多執行緒）
│   ├── main.py                  # cv2 模式入口（獨立開發 / 測試用）
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
│   └── dashboard.py             # PyQt5 本機儀表板（獨立執行 yolov8 管線用）
├── Helmet/
│   ├── camera_helmet_detect.py  # 獨立執行的即時安全帽偵測（LLM API）
│   ├── preprocess_head.py       # YOLOv8 pose 頭部裁切前處理（web/app.py 亦呼叫此函式）
│   └── yolov8n-pose.pt          # 姿態模型（Helmet 模組獨立使用）
├── ResNet/
│   ├── register_user.py         # 人臉特徵向量註冊（CLI 版，web/app.py 另有 /api/register）
│   ├── verify_user.py           # 人臉身分驗證（CLI 版）
│   └── user_feature.npy         # 已註冊使用者的人臉特徵向量
├── MediaPipe/
│   ├── helmet_detection.py      # 安全帽偵測（實驗性）
│   └── train.py                 # 安全帽偵測模型訓練腳本
├── pics/                        # UI 示意圖與 overlay 素材
├── requirements.txt
└── README.md
```

---

## 各模組說明

### `web/` — 整合式 Web 介面（主要展示入口）

| 檔案 | 功能 |
|------|------|
| `app.py` | Flask App，單一鏡頭、單一頁面完成完整流程：<br>① **人臉辨識**：呼叫 `RetinaFace` + `ArcFace`（uniface）擷取嵌入向量，與 `ResNet/user_feature.npy` 做餘弦相似度比對（門檻 0.4）<br>② **安全帽確認**：呼叫 `Helmet/preprocess_head.crop_head` 裁切頭部，送至 LLM API（Gemma3:27B）判斷「未戴安全帽 / 已戴但下巴帶未扣 / 已戴且扣好」，僅扣好才能進入下一階段<br>③ **行駛監控**：Flask 釋放鏡頭 → 啟動 `yolov8` 四執行緒（Capture / YOLOv8 / MediaPipe / Decision），透過 `queue_display` 取得已標註畫面<br>三階段全程以 **MJPEG** (`/video_feed`) 即時串流至瀏覽器，並提供 `/api/*` 路由驅動前端狀態機 |
| `templates/index.html` | 單頁式三步驟流程 UI（深色卡片風格）：步驟指示條、即時影像、操作按鈕（開始驗證 / 偵測安全帽 / 開始行駛監控）、人臉註冊上傳對話框 |

技術重點：
- 三階段共用同一支攝影機與同一個 Flask 行程，透過 `_camera_worker` 與 yolov8 `thread_capture` 互相讓出鏡頭控制權，避免裝置衝突
- `_state` + `threading.Lock` 管理跨請求的流程狀態機（`face_verify` → `helmet_check` → `driving`）
- 直接 `import` `yolov8/` 的 `state`、`display.annotate`、四條執行緒函式，不重寫推論邏輯，確保獨立執行與 Web 整合行為一致

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

### `PyQT5/` — 本機圖形儀表板

| 檔案 | 功能 |
|------|------|
| `dashboard.py` | 取代 `main.py` 的 `cv2.imshow`，獨立執行 yolov8 管線時的本機圖形介面。匯入 `yolov8/` 四執行緒，QTimer(33ms) 刷新影像與右側狀態面板（LevelBar、4 個違規指示卡片、感測數值、校準按鈕） |

### `Helmet/` — 安全帽偵測模組

| 檔案 | 功能 |
|------|------|
| `preprocess_head.py` | 使用 YOLOv8n-pose 偵測人物，取肩膀關鍵點定位頭部範圍，裁切出頭頸區域（含安全帽），支援 HEIC/HEIF 輸入 |
| `camera_helmet_detect.py` | 開啟鏡頭，每 8 秒裁切一次頭部後傳送至 Gemma3:27B API 判斷：`Helmet on, strap fastened` / `Helmet on, strap unfastened` / `No helmet`，結果疊加至畫面左上角 |

### `ResNet/` — 人臉身分驗證

| 檔案 | 功能 |
|------|------|
| `register_user.py` | CLI 版：從照片提取 ArcFace 特徵向量，確認唯一人臉後儲存為 `user_feature.npy`（`web/app.py` 的 `/api/register` 提供同等功能的網頁上傳介面） |
| `verify_user.py` | CLI 版：擷取單幀與 `user_feature.npy` 做餘弦相似度比對（門檻 0.4），驗證駕駛身分 |
| `user_feature.npy` | 已註冊使用者的人臉特徵向量，CLI 與 Web 流程共用同一份檔案 |

技術來源：https://medium.com/%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7-%E5%80%92%E5%BA%95%E6%9C%89%E5%A4%9A%E6%99%BA%E6%85%A7/%E8%AB%96%E6%96%87%E9%96%B1%E8%AE%80-cvpr-2020-retinaface-single-shot-multi-level-face-localisation-in-the-wild-234566d3a89b

### `MediaPipe/` — 安全帽偵測（實驗性）

| 檔案 | 功能 |
|------|------|
| `helmet_detection.py` | MediaPipe Custom Model 安全帽偵測（實驗性，非主要流程） |
| `train.py` | 安全帽偵測模型訓練腳本（需 TensorFlow） |

---

## 使用套件

| 套件 | 用途 |
|------|------|
| `opencv-contrib-python` / `opencv-python` | 影像讀取、繪圖、solvePnP 頭姿估計、MJPEG 編碼 |
| `numpy` | 數值計算（EAR、頭姿矩陣、特徵向量、餘弦相似度） |
| `mediapipe` | 人臉 468 點 Landmark（EAR、嘴部、頭姿）、Tasks API |
| `ultralytics` | YOLOv8 物件偵測與姿態偵測（手機、香菸、手腕、頭部裁切） |
| `flask` | `web/app.py` 整合式介面：路由、MJPEG 串流、流程狀態 API |
| `PyQt5` | 本機儀表板（VideoWidget、狀態面板、指示燈） |
| `pillow` / `pillow-heif` | 影像格式轉換、Helmet / 人臉註冊模組 HEIC/HEIF 圖片讀取支援 |
| `uniface` | RetinaFace（人臉偵測）+ ArcFace（特徵提取），用於人臉註冊與驗證 |
| `requests` | 呼叫 LLM 安全帽判斷 API（Gemma3:27B，經 ngrok 對外）|
| `huggingface-hub` | 香菸模型自動下載（basant18/Smoking-detection-YOLO26s） |
| `tensorflow` / `torch` / `onnxruntime` | 各模型（YOLO / MediaPipe / uniface / 自訓練模型）推論後端 |

---

## 快速開始

### 環境安裝（Jetson Nano / linux-aarch64）

```bash
# PyQt5 需透過 conda 安裝
conda install "pyqt=5.15.11" -n goshare

# 其他套件
pip install -r requirements.txt
```

### 啟動整合式 Web 介面（主要展示入口）

```bash
python web/app.py
```

於瀏覽器開啟 `http://<裝置IP>:5000`，依序完成「① 人臉辨識 → ② 安全帽確認 → ③ 行駛監控」。
首次使用需先點右上角「註冊人臉」上傳照片，建立 `ResNet/user_feature.npy`。

> 安全帽判斷需連線至 LLM API 伺服器（Gemma3:27B，預設經 ngrok 對外），若伺服器離線會於步驟 ② 顯示錯誤訊息。

### 啟動 PyQt5 本機儀表板（獨立執行 yolov8 管線）

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

### 獨立啟動安全帽偵測 / 人臉註冊驗證（CLI 工具）

```bash
cd Helmet && python camera_helmet_detect.py     # 即時安全帽偵測
cd ResNet && python register_user.py            # 註冊人臉特徵
cd ResNet && python verify_user.py              # 驗證人臉身分
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
- 攝影機：USB Webcam 或 CSI Camera（index 0），Web 流程與 yolov8 管線共用同一顆鏡頭
- 安全帽確認需連線至 LLM API 伺服器（Gemma3:27B，預設經 ngrok 對外網址存取）
- Web 介面預設監聽 `0.0.0.0:5000`，需與裝置在同一網路下用瀏覽器存取

---

## 輸出

- **CSV**：每個決策週期一行，記錄 raw_pitch、corr_pitch、yaw、roll、EAR、各偵測旗標、alert_level，自動儲存於 `yolov8/output/monitor_YYYYMMDD_HHMMSS.csv`
