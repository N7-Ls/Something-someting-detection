# GoShare PyQt5 儀表板 — 實作計劃與進度

## 目的

取代 `yolov8/main.py` 的 `cv2.imshow` 迴圈，提供完整圖形化駕駛監控儀表板。
直接在同一 process 內匯入並啟動 yolov8 四執行緒，讀取 `state.py` 共享狀態渲染 UI。

---

## 架構

```
PyQT5/dashboard.py
  │
  ├─ sys.path.insert → ../yolov8/
  │
  ├─ 啟動 4 條 daemon threads（同 yolov8/main.py）
  │   thread_capture / thread_yolo / thread_mediapipe / thread_decision
  │
  ├─ QTimer(33ms) → _refresh()
  │   ├─ 從 queue_display 拉最新 (frame_id, frame)
  │   ├─ 呼叫 display.annotate() 標註影格
  │   └─ 讀 display_state 快照更新各 Widget
  │
  └─ QMainWindow
      ├─ VideoWidget（左側，可伸縮）
      └─ 右側控制面板
          ├─ LevelBar（警告等級 + 措施文字）
          ├─ AlertIcon × 4（手機/抽菸/疲勞/分心）
          ├─ StatusPanel（模型運行數值）
          └─ 按鈕（重新校準 / 結束）
```

---

## UI 佈局

```
┌──────────────────────────────┬────────────────┐
│                              │  警告等級        │
│   鏡頭回饋                    │  LEVEL 2  ████  │
│   (annotated video)          │  鎖定 30 km/h   │
│                              ├────────────────┤
│                              │  違規指示        │
│                              │  📱 手機操作  ●  │
│                              │  🚬 吸菸行為  ●  │
│                              │  😴 疲勞駕駛  ●  │
│                              │  👁 視線分心  ●  │
│                              ├────────────────┤
│                              │  模型狀態        │
│                              │  FPS  : 24.3   │
│                              │  臉部 : OK#1234 │
│                              │  EAR  : 0.283  │
│                              │  Yaw  : +12.3° │
│                              │  Ptch*: +8.1°  │
│                              │  Sync : Y2 F1  │
│                              │  校準 : 完成    │
│                              │  Cig  : 已載入  │
│                              ├────────────────┤
│                              │ [校準] [結束]   │
└──────────────────────────────┴────────────────┘
```

---

## 指示燈顏色邏輯

| 狀態 | 顏色 | 說明 |
|------|------|------|
| 灰色 | `#3a3a3a` | 正常，無任何條件 |
| 橘色 | `#e67e22` | 原始感測條件成立（瞬間，未計時） |
| 紅色 | `#e74c3c` | 計時器觸發（持續達門檻，真正警報） |

觸發狀態來自 `display_state["alert_flags"]`（由 `thread_decision.py` 寫出）。
原始條件由儀表板直接從感測值推斷（phone_boxes、ear_val、yaw、pitch_corr 等）。

---

## 依賴的 yolov8 修改

| 檔案 | 修改內容 |
|------|---------|
| `yolov8/state.py` | `display_state` 加入 `"alert_flags": {...}` |
| `yolov8/thread_decision.py` | 計算 `trig_*` 後寫入 `display_state["alert_flags"]` |

---

## 實作狀態

| 項目 | 狀態 |
|------|------|
| CLAUDE.md 計劃文件 | ✅ |
| yolov8/state.py 加 alert_flags | ✅ |
| yolov8/thread_decision.py 寫 alert_flags | ✅ |
| dashboard.py — VideoWidget | ✅ |
| dashboard.py — AlertIcon | ✅ |
| dashboard.py — LevelBar | ✅ |
| dashboard.py — StatusPanel | ✅ |
| dashboard.py — DashboardWindow | ✅ |
| dashboard.py — main() 啟動執行緒 + Qt | ✅ |

---

## 如何執行

```bash
cd PyQT5
python dashboard.py
```

### 依賴套件
```bash
pip install PyQt5 opencv-python ultralytics mediapipe
```

### 鍵盤操作
| 鍵 | 功能 |
|----|------|
| C | 重新校準攝影機 pitch 補償 |
| Q / ESC | 結束程式 |

---

## 未完成 / 後續優化

| 優先級 | 項目 |
|--------|------|
| 🟡 | 錄影按鈕整合（對應 yolov8/main.py 的 R 鍵） |
| 🟡 | Arduino 連線狀態顯示於 StatusPanel |
| 🟡 | 歷史警報 log 區塊（最近 N 筆觸發記錄） |
| 🟡 | 深色/淺色主題切換 |
