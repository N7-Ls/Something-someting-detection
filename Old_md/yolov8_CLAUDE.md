# GoShare Driver Monitor — YOLOv8 模組技術文件

## 專案概述

共享機車駕駛行為監控系統（4.4 多工視覺融合模組）。
運行環境：NVIDIA Jetson Nano（部署）/ Windows（開發）。
技術棧：Python 3.x + OpenCV + YOLOv8 + MediaPipe + Threading + CSV。

---

## 整體架構

### 多執行緒資料流

```
攝影機
  ├─ queue_pose (maxsize=4) ──► [Thread 2] YOLOv8
  │                                手機偵測 / 手腕關鍵點 / 香菸偵測
  └─ queue_face (maxsize=4) ──► [Thread 3] MediaPipe
                                   EAR / Yaw-Pitch-Roll / 嘴部中心 / 臉寬
                                         │              │
                              queue_decision (maxsize=8)
                                              │
                              [Thread 4] 決策中心（融合）
                              自動校準 / 時序融合 / 計時器 / Grace Period / CSV
                                              │
                              display_state (Lock 保護)
                                              │
                              [主執行緒] UI 迴圈
                              annotate / cv2.imshow / 錄影 / 鍵盤
                              （或 PyQT5/dashboard.py 取代 cv2.imshow）
```

### 模組職責

| 檔案 | 職責 |
|------|------|
| `config.py` | 所有常數集中管理，修改參數只改此檔 |
| `state.py` | 跨執行緒共享狀態（Queue / Lock / Event） |
| `utils.py` | 純工具函式（EAR、solvePnP、wrap_angle、pixel_dist） |
| `thread_capture.py` | 影像擷取，同時投遞三條 Queue，Queue 滿時丟幀維持實時性 |
| `thread_yolo.py` | YOLOv8 推論：手機偵測（面積/長寬比過濾）、手腕關鍵點、香菸偵測 |
| `thread_mediapipe.py` | 臉部分析：EAR、solvePnP 頭部姿態、嘴部中心、臉寬 |
| `thread_decision.py` | 決策核心：自動校準、時序融合（±0.1s）、計時器 + Grace Period、CSV 寫入 |
| `display.py` | 畫面標註：邊框顏色（Level 0-3）、資訊面板、警報訊息 |
| `main.py` | 入口（cv2）：啟動 4 執行緒，主執行緒跑 cv2.imshow 迴圈 |
| `monitor.py` | 舊版單檔版本，保留參考用，**不使用** |
| `../PyQT5/dashboard.py` | 入口（Qt）：取代 main.py，提供 PyQt5 圖形儀表板 |

---

## 關鍵設計決策

### 1. corr_pitch 補償公式
```python
corr_pitch = -wrap_angle(raw_pitch - cam_offset)
```
- **正值 = 低頭**（觸發 PHONE 或 DISTRACT）
- **負值 = 抬頭**（觸發 DISTRACT）
- `wrap_angle` 必不可少：解決 ±180° 邊界跳變
- 負號必不可少：solvePnP 在龍頭幾何下，低頭使 raw_pitch 減少

### 2. 計時器 + Grace Period
```python
def check_duration(key, condition, required_sec):
    # 條件成立：啟動主計時器
    # 條件消失：啟動 hold 計時器（HOLD_MAP[key] 秒）
    # Hold 期滿：真正重置主計時器
```
防止單幀 YOLO 結果缺失導致計時器誤重置。

### 3. YOLO fallback cache
YOLO 速度慢於 MediaPipe；若當前幀無結果，取 1s 內最新快取。
`DURATION_PHONE=0.15s`（短於幀間隔），搭配 fallback 確保手機計時器能累積。

### 4. 自動校準（Pitch 補償）
啟動後 5s 收集 raw_pitch 樣本，圓形平均計算補償值。
按 C 鍵可重新校準。

---

## 警報等級定義

| Level | 條件 | 視覺效果 |
|-------|------|---------|
| 0 | 正常 | 綠色邊框 |
| 1 | 視線分心（\|Yaw\|>45° 或 corr_pitch<-45°）持續 2s | 黃色邊框 + 底部提示 |
| 2 | 手機（偵測/低頭/手腕靠臉）或抽菸，持續門檻達到 | 橘色邊框 + 30km/h 標示 |
| 3 | 疲勞（EAR<0.20）持續 1.5s | 紅色邊框 + DANGER 全幅 + 時速歸零 |

優先級：Level 3 > Level 2 > Level 1 > Level 0。

---

## 關鍵參數（config.py）

```python
YOLO_IMGSZ = 640            # 通用偵測解析度
YOLO_POSE_IMGSZ = 416       # 姿態模型（降低以減延遲）
YOLO_PHONE_CONF = 0.20      # 手機偵測門檻
YOLO_CIG_CONF = 0.30        # 香菸偵測門檻
WRIST_MOUTH_RATIO = 0.55    # 手腕-嘴部靠近閾值（相對臉寬）
YAW_PITCH_LIMIT = 45.0      # 視線偏轉閾值（度）
PITCH_PHONE_LIMIT = 32.0    # 低頭滑手機閾值（度）
EAR_THRESHOLD = 0.20        # 疲勞閾值
FUSE_TIME_WINDOW = 0.1      # 時序融合容忍窗口（秒）
PITCH_CAM_OFFSET = 25.0     # 攝影機仰角初始估計
CALIB_SECONDS = 5.0         # 自動校準秒數
CIG_INTERVAL_SEC = 0.3      # 香菸推論最短間隔（time-based，取代已廢棄的 CIG_SKIP_FRAMES）
DURATION_PHONE = 0.15       # 手機計時門檻（短於幀間隔）
DURATION_FATIGUE = 1.5      # 疲勞計時門檻
DURATION_DISTRACT = 2.0     # 分心計時門檻
DURATION_SMOKE = 2.0        # 抽菸計時門檻（有模型）
DURATION_SMOKE_NOCIG = 3.0  # 抽菸計時門檻（無模型，更保守）
HOLD_PHONE = 0.4            # 手機 Grace Period
HOLD_SMOKE = 1.0            # 抽菸 Grace Period
```

---

## 已完成的 suggestions.md 優化（不可回退）

| 建議 | 修改檔案 | 說明 |
|------|---------|------|
| 建議 2：丟幀策略 | `config.py` | 移除 `CIG_SKIP_FRAMES`，僅保留 `CIG_INTERVAL_SEC` |
| 建議 3：計時器函數 | `thread_capture.py` | `ts = time.perf_counter()`（原 `time.time()`） |
| 建議 5：cache 過期 | `thread_decision.py` | fallback cache 比對改用 `time.perf_counter()`，與 ts 同源 |
| 建議 1：影格同步 | `state.py`, `thread_yolo.py`, `thread_mediapipe.py`, `display.py`, `main.py` | `queue_display` 改傳 `(frame_id, frame)`；display 面板新增 `Sync: Y-N F-N` 落差指示（>8 幀變黃） |
| 建議 6：BBox 過濾 | `thread_yolo.py` | 面積 >10% 或近正方形（0.7~1.4）視為誤判丟棄 |
| 建議 7：動態 ROI | `thread_yolo.py` | 香菸模型改用臉部中心裁切 ROI，100% 排除軀幹誤判 |

**建議 4（GIL/Multiprocessing）**：Jetson Nano 效能不足時可評估，目前無需修改。

---

## 已修正的歷史 Bug（不可重蹈）

| Bug | 根因 | 修正方式 |
|-----|------|---------|
| head_pose 角度爆炸成萬級 | `cv2.RQDecomp3x3` 已返回度，不需再 ×360 | 移除 `* 360` |
| corr_pitch ±180° 邊界跳變 | raw - offset 未 wrap | 加入 `wrap_angle()` |
| 低頭觸發 DISTRACT、抬頭觸發 PHONE | solvePnP 方向與預期相反 | `corr_pitch = -wrap_angle(...)` |
| 手機計時器永遠無法累積 | YOLO 單幀條件即刻 False | fallback cache + Grace Period + DURATION_PHONE=0.15s |
| 香菸推論間隔因丟幀拉長 | 原用幀計數 skip | 改用 `time.perf_counter()` 時間間隔 |

---

## 目前程式狀態

### 已完成
- 四執行緒並行架構（擷取、YOLOv8、MediaPipe、決策）
- 攝影機自動校準（5s 圓形平均）
- 時序融合（±0.1s 容忍）
- Grace Period 計時器
- 手機偵測（面積 + 長寬比過濾）+ fallback COCO class 67
- 手腕關鍵點（yolov8n-pose.pt）
- 香菸偵測 fallback（手腕靠嘴判定，無模型時啟用）
- 疲勞偵測（EAR）
- 視線分心偵測（Yaw / corr_pitch）
- CSV 感測記錄（逐決策週期）
- 影片錄製（R 鍵，MP4）
- 畫面標註（Level 顏色、資訊面板、警報訊息）
- 優雅退出（sentinel + join timeout）
- 手機模型（`yolov8n_phone.pt`）與香菸模型（`yolov8n_cigarette.pt`）均已存在
- 影格同步追蹤（`queue_display` 傳 `(frame_id, frame)`，`display_state` 記錄 `yolo_frame_id`/`face_frame_id`，UI 面板顯示 `Sync: Y-N F-N`，落後 >8 幀變黃）
- 所有 frame packet 時間戳改用 `time.perf_counter()`（防止 NTP 校時影響計時器與融合邏輯）
- YOLO fallback cache 過期保護（`time.perf_counter()` 比對，超過 0.5s 不採用）
- `display_state["alert_flags"]` — 各違規觸發旗標（phone/smoke/fatigue/distract），由 `thread_decision.py` 寫出，供 `PyQT5/dashboard.py` 讀取
- PyQt5 儀表板（`PyQT5/dashboard.py`）— 鏡頭回饋、違規指示燈（三態）、警告等級橫條、模型狀態面板

### 未完成（依優先級）

| 優先級 | 項目 | 說明 |
|--------|------|------|
| 🔴 P1 | Arduino 移植 | `monitor.py` 有完整 `ArduinoController` 實作，**尚未移植至 `thread_decision.py`** |
| 🟡 P2 | 脫帽偵測 | 4.5 規格要求的 Level 2 違規項目，目前完全無對應程式碼 |
| 🟡 P3 | 機車視角模型微調 | 兩個模型已存在，可針對龍頭俯角場景蒐集資料微調 |
| 🟡 P4 | PyQt5 儀表板錄影按鈕 | `dashboard.py` 尚未整合錄影功能（對應 main.py R 鍵） |

---

## 需要改進的部分

### 立即可做

1. **Arduino 串列埠整合**（在 `thread_decision.py` 等級變化點觸發）
   - Level 1 → 串列埠送 `b'1'`，蜂鳴器短音 + 黃 LED
   - Level 2 → 串列埠送 `b'2'`，連續警示 + 紅 LED
   - Level 3 → 串列埠送 `b'3'`，長鳴 + LED 爆閃

### 模型微調建議（兩個模型已存在，非緊急）

**手機模型（`yolov8n_phone.pt` 已存在）**
- 若誤報率仍高，收集 150+ 張機車龍頭俯角標註資料微調
- 重點：大量負樣本（無手機的正常駕駛畫面）

**香菸模型（`yolov8n_cigarette.pt` 已存在）**
- 若誤報率仍高，收集 100+ 張真實場景資料微調
- ROI 裁切已實作，訓練時加入「手靠嘴但無香菸」的負樣本

### 效能優化（邊緣設備）

- 若 Jetson Nano 效能瓶頸，考慮將 YOLOv8 改用 TensorRT 量化模型
- MediaPipe 可降低 `min_face_detection_confidence` 至 0.25 換取速度
- 考慮將 GIL 限制嚴重的部分改用 `multiprocessing`

---

## 鍵盤操作

| 鍵 | 功能 |
|----|------|
| R / r | 切換錄影（`output/monitor_*.mp4`） |
| C / c | 重新校準攝影機 pitch 補償 |
| Q / ESC | 結束程式 |

## 啟動方式

**cv2 模式（終端機）**
```bash
python main.py
```

**PyQt5 儀表板模式**
```bash
cd ../PyQT5
python dashboard.py
```

## CSV 輸出欄位

```
timestamp, raw_pitch, corr_pitch, yaw, roll, ear, cam_offset,
phone_bbox, phone_gaze, phone_wrist,
smoke, fatigue, distract,
alert_level, alert_msg
```
