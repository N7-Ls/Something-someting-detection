# YOLOv8 駕駛監控模組

部署環境：NVIDIA Jetson Nano / Windows 開發。
技術棧：Python + OpenCV + YOLOv8 + MediaPipe + Threading。

## 執行緒架構

```
攝影機 → queue_pose → thread_yolo (手機/手腕/香菸)
       → queue_face → thread_mediapipe (EAR/頭姿/嘴部)
                         ↓
                   queue_decision → thread_decision (融合/計時/CSV)
                                        ↓
                                   display_state (Lock)
                                        ↓
                                   主執行緒 UI / PyQT5/dashboard.py
```

## 模組職責

| 檔案 | 職責 |
|------|------|
| `config.py` | 所有常數，修改參數只改此檔 |
| `state.py` | 跨執行緒共享狀態（Queue/Lock/Event） |
| `utils.py` | 純工具函式（EAR、solvePnP、wrap_angle） |
| `thread_capture.py` | 影像擷取，Queue 滿時丟幀 |
| `thread_yolo.py` | 手機偵測（ROI 模式）+ 手腕關鍵點 + 香菸偵測 |
| `thread_mediapipe.py` | EAR + solvePnP 頭姿 + 嘴部中心 |
| `thread_decision.py` | 自動校準 + 時序融合(±0.1s) + Grace Period + CSV |
| `display.py` | 畫面標註（Level 顏色、資訊面板） |
| `main.py` | 入口(cv2)：啟動 4 執行緒 + cv2.imshow |

## 手機偵測架構（ROI 模式）

攝影機拍的是「手持手機」，手機幾乎必定被手遮擋、輪廓不完整，全圖掃描誤報率極高。
**正確流程：先 pose 找手腕/手肘 → 對 ROI 跑 phone 模型。**

手機模型：`yolov8s.pt`（COCO class 67 = cell phone）
- 放棄 IndUSV nano 專用模型——在手持遮擋、局部可見、低光場景表現差
- COCO yolov8s 訓練資料多樣，涵蓋各種角度與遮擋，配合 ROI 控制誤報

```
thread_yolo 每幀：
  1. yolov8n-pose → 手腕 (idx 9,10, conf>0.15) + 手肘 (idx 7,8, conf>0.20)
  2. 有手腕 → ROI 以手腕為中心，往上延伸 1.4×pad（手機在手腕上方）
  3. 無手腕但有手肘（且在畫面上 3/4）→ ROI 往下延伸 2.0×pad（手機在手肘下方）
  4. 無任何關節點 → 跳過 bbox 偵測
```

`phone_cls = 67`（固定，COCO cell phone）

手機觸發條件（任一）：
- `phone_by_bbox`：ROI 內偵測到手機 bbox（conf ≥ 0.35）
- `phone_by_gaze`：corr_pitch > 32°（低頭）
- `phone_by_wrist`：手腕 y 座標 < mouth_y × 1.25

## 關鍵設計

- `corr_pitch = -wrap_angle(raw_pitch - cam_offset)`：正值=低頭，負號因 solvePnP 方向與預期相反
- Grace Period：條件消失後再等 HOLD_MAP[key] 秒才重置，防單幀 YOLO 缺失誤重置
- 香菸模型：time-based 間隔推論（CIG_INTERVAL_SEC），以臉部為中心裁 ROI 送入
- 時間戳全用 `time.perf_counter()`，防 NTP 校時影響

## 警報等級

| Level | 條件 | 顏色 |
|-------|------|------|
| 1 | \|Yaw\|>45° 或 corr_pitch<-45° 持續 2s | 黃 |
| 2 | 手機/抽菸達門檻 | 橘 |
| 3 | EAR<0.20 持續 1.5s | 紅 |

## 主要 config 參數

| 參數 | 值 | 說明 |
|------|----|------|
| `YOLO_PHONE_CONF` | 0.35 | ROI 模式下誤報由空間限制，可低於全圖模式 |
| `PHONE_ROI_PAD` | 0.22 | 手腕 ROI 半徑（佔 min(h,w) 比例） |
| `DURATION_PHONE` | 0.15s | 手機條件持續觸發門檻 |
| `HOLD_PHONE` | 0.4s | Grace period |

## 未完成

| 優先 | 項目 |
|------|------|
| P1 | Arduino 移植（monitor.py 有完整實作，需移植至 thread_decision.py） |
| P2 | 脫帽偵測 |
| P3 | 機車視角模型微調 |
| P4 | PyQt5 錄影按鈕 |

## 啟動

```bash
python main.py          # cv2 模式
cd ../PyQT5 && python dashboard.py  # Qt 模式
```

鍵盤：R=錄影切換、C=重新校準、Q/ESC=結束
