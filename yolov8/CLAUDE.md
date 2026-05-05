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
| `thread_yolo.py` | 手機偵測 + 手腕關鍵點 + 香菸偵測 |
| `thread_mediapipe.py` | EAR + solvePnP 頭姿 + 嘴部中心 |
| `thread_decision.py` | 自動校準 + 時序融合(±0.1s) + Grace Period + CSV |
| `display.py` | 畫面標註（Level 顏色、資訊面板） |
| `main.py` | 入口(cv2)：啟動 4 執行緒 + cv2.imshow |
| `monitor.py` | 舊版單檔，**不使用** |

## 關鍵設計

- `corr_pitch = -wrap_angle(raw_pitch - cam_offset)`：正值=低頭，負號因 solvePnP 方向與預期相反
- Grace Period：條件消失後再等 HOLD_MAP[key] 秒才重置，防單幀 YOLO 缺失誤重置
- YOLO fallback cache：結果快取 0.5s，搭配 DURATION_PHONE=0.15s 確保計時器能累積
- 時間戳全用 `time.perf_counter()`，防 NTP 校時影響

## 警報等級

| Level | 條件 | 顏色 |
|-------|------|------|
| 1 | \|Yaw\|>45° 或 corr_pitch<-45° 持續 2s | 黃 |
| 2 | 手機/抽菸達門檻 | 橘 |
| 3 | EAR<0.20 持續 1.5s | 紅 |

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
