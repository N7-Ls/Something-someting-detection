# YOLOv8 駕駛監控模組

部署環境：NVIDIA Jetson Nano / Windows 開發。
技術棧：Python + OpenCV + YOLOv8 + MediaPipe + Threading。

被 `web/app.py` 直接 import（`state`/`display.annotate`/五條 thread 函式）；
`main.py` 是獨立 cv2 測試入口，`PyQT5/dashboard.py` 是另一種獨立 GUI 入口。

## 執行緒架構

```
攝影機 → queue_pose → thread_yolo (手機/手腕)
       → queue_face → thread_mediapipe (EAR/頭姿/嘴部)
       → queue_cig  → thread_cigarette (香菸，獨立執行緒避免阻塞手機/pose)
                         ↓
                   queue_decision → thread_decision (融合/計時/PERCLOS/CSV)
                                        ↓
                                   display_state (Lock)
                                        ↓
                    web/app.py MJPEG ／ PyQT5/dashboard.py ／ main.py cv2
```

## 模組職責

| 檔案 | 職責 |
|------|------|
| `config.py` | 所有常數，修改參數只改此檔 |
| `state.py` | 跨執行緒共享狀態（Queue/Lock/Event） |
| `utils.py` | 純工具函式（EAR、solvePnP、wrap_angle） |
| `thread_capture.py` | 影像擷取，Queue 滿時丟幀 |
| `thread_yolo.py` | 手機偵測（ROI 模式）+ 手腕/手肘關鍵點 |
| `thread_cigarette.py` | 香菸偵測，獨立執行緒，time-based 間隔推論 + 臉部 ROI 裁切 |
| `thread_mediapipe.py` | EAR + solvePnP 頭姿 + 嘴部中心 + 臉寬 |
| `thread_decision.py` | 自動校準 + 時序融合(±0.1s) + PERCLOS 疲勞 + Grace Period + CSV |
| `display.py` | 畫面標註（Level 顏色、資訊面板、`annotate(minimal=True)` 給 PyQt5 用）|
| `main.py` | 入口(cv2)：啟動 5 執行緒 + cv2.imshow |

## 手機偵測架構（ROI 模式）

攝影機拍的是「手持手機」，手機幾乎必定被手遮擋、輪廓不完整，全圖掃描誤報率極高。
**正確流程：先 pose 找手腕/手肘 → 對 ROI 跑 phone 模型。**

手機模型：`yolov8s.pt`（COCO class 67 = cell phone，固定 `_PHONE_CLS=67`）
- 放棄 IndUSV nano 專用模型——在手持遮擋、局部可見、低光場景表現差
- COCO yolov8s 訓練資料多樣，涵蓋各種角度與遮擋，配合 ROI 控制誤報

```
thread_yolo 每幀：
  1. yolov8n-pose → 手腕 (idx 9,10, conf>0.10) + 手肘 (idx 7,8, conf>0.13)
  2. 有手腕 → 以手腕為中心的 ROI（pad×1.3~1.8，偏向手腕上方）
  3. 無手腕但有手肘（且在畫面上 3/4）→ 以手肘為中心的 ROI 向下延伸（pad×1.1~2.5）
  4. 都沒有 → 全圖 fallback（YOLO_PHONE_IMGSZ=320）
```

手機觸發條件（任一，邏輯在 `thread_decision.py`）：
- `phone_by_bbox`：ROI 內偵測到手機 bbox（conf ≥ `YOLO_PHONE_CONF`=0.30）
- `phone_by_gaze`：`corr_pitch > PITCH_PHONE_LIMIT`（8°，低頭看手機）
- `phone_by_wrist`：手腕 y < 嘴部 y + 臉寬×2.0（手舉到臉旁的持機範圍）

## 關鍵設計

- `utils.head_pose()`：`cv2.RQDecomp3x3` 對同一旋轉矩陣有兩種等價解
  `(pitch, yaw, roll)` 與 `(pitch+180, 180-yaw, roll+180)`，會在兩者間跳動造成
  `corr_pitch` 假性跳變 ±100°以上。已修正為固定取 `|roll|<=90°` 的分支。
- `corr_pitch = -wrap_angle(raw_pitch - cam_offset)`：正值=低頭，負號因 solvePnP 方向與預期相反
- 自動校準分兩段：進入行駛階段後先延遲 `CALIB_DELAY_SEC`(3s) 不收樣，
  讓使用者回到正常騎乘姿勢，再花 `CALIB_SECONDS`(5s) 收集 `cam_offset`/EAR 基準。
  避免剛點「開始行駛」時頭部姿勢（如低頭看手機按按鈕）被當成校準基準，
  導致 `corr_pitch` 整段偏移、低頭判定範圍被吃掉
- **疲勞偵測用 PERCLOS**（非單純 EAR 持續時間）：3 秒滾動窗口
  (`PERCLOS_WINDOW_SEC`) 內 `ear < EAR_THRESHOLD`(0.27) 的幀數比例
  ≥ `PERCLOS_THRESHOLD`(0.50) 即觸發；`DURATION_FATIGUE=0`（PERCLOS 已含時序過濾）
- Grace Period：條件消失後再等 `HOLD_MAP[key]` 秒才重置，防單幀 YOLO 缺失誤重置
- 香菸模型：獨立執行緒 `thread_cigarette`，time-based 間隔推論（`CIG_INTERVAL_SEC`，
  `YOLO_CIG_IMGSZ`=320），以臉部為中心裁 ROI 送入，結果寫入 `display_state["cig_boxes"]`；
  `thread_decision` 直接讀取 `cig_boxes` 判斷 `cig_near_mouth`，
  模型不可用時 fallback 為「手腕靠嘴」判定（`DURATION_SMOKE_NOCIG`=3.0s）
- 時間戳全用 `time.perf_counter()`，防 NTP 校時影響

## 警報等級（優先序 3 > 2 > 1）

| Level | 條件 | 顏色 |
|-------|------|------|
| 1 | distract：\|Yaw\|>45° 或 corr_pitch<-45° 持續 2s | 黃 |
| 2 | phone（任一條件）達 0.3s，或 smoke 達 2~3s | 橘 |
| 3 | fatigue：PERCLOS ≥ 50%（3s 窗口） | 紅 |

## 未完成

| 優先 | 項目 |
|------|------|
| P1 | Arduino 移植（`_trash/yolov8/monitor.py` 有完整實作，需移植至 thread_decision.py） |
| P2 | 脫帽偵測 |
| P3 | 機車視角模型微調 |
| P4 | PyQt5 錄影按鈕 |

## 啟動

```bash
python main.py                       # cv2 模式（獨立測試）
cd ../PyQT5 && python dashboard.py   # Qt 模式（獨立測試）
cd ../web  && python app.py          # 整合 Web 流程（主要入口）
```

鍵盤（main.py / dashboard.py）：R=錄影切換、C=重新校準、Q/ESC=結束
