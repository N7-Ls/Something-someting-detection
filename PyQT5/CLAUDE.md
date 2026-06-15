# PyQt5 儀表板

> ❌ **與 `web/app.py` 主流程無關**，是獨立執行 yolov8 管線時的本機 GUI（取代
> `yolov8/main.py` 的 cv2.imshow）。修改行駛監控邏輯只需動 `yolov8/`，
> 這裡只是另一種顯示方式。

直接匯入 yolov8 四執行緒，讀取 `state.py` 共享狀態渲染 UI。

## 架構

```
dashboard.py
  ├─ sys.path.insert → ../yolov8/
  ├─ 啟動 4 daemon threads（同 yolov8/main.py）
  ├─ QTimer(33ms) → _refresh()
  │   ├─ 從 queue_display 拉 (frame_id, frame)
  │   ├─ 呼叫 display.annotate(minimal=True)
  │   ├─ PhoneStatusBar.tick()（更新時鐘）
  │   └─ 讀 display_state 更新 Widget
  └─ QMainWindow
      ├─ VideoWidget（左側，自動縮放保持長寬比）
      └─ 右側：手機通知介面（PhoneFrame）
          ├─ PhoneStatusBar（時間 + "GoShare"）
          ├─ LevelBar（警告等級色條）
          ├─ NotificationCard × 4（違規指示）
          ├─ StatusPanel（模型數值格狀顯示）
          └─ 按鈕列（重新校準 / 結束）
```

## 右側 UI 設計（手機通知風格）

`_PANEL_W = 300`，外框 `border-radius: 20px` 模擬手機外殼。

### NotificationCard（取代舊 AlertIcon）

無 emoji，改用純文字 badge：

| key | badge | 說明 |
|-----|-------|------|
| phone | TEL | 手機操作 |
| smoke | SMK | 吸菸行為 |
| fatigue | DRW | 疲勞駕駛 |
| distract | EYE | 視線分心 |

三種狀態：
- `normal`：灰色，副標「待機中」
- `condition`：橘色，副標「條件成立」（瞬間條件，未計時）
- `triggered`：紅色，副標「警報觸發」（計時器到達門檻）

觸發來源：`display_state["alert_flags"]`（由 thread_decision.py 寫出）

## 指示燈狀態

| 顏色 | 意義 |
|------|------|
| 灰 `#3a3a3a` | 正常 |
| 橘 `#e67e22` | 瞬間條件成立（未計時） |
| 紅 `#e74c3c` | 計時器觸發（真正警報） |

## 未完成

- 錄影按鈕整合（對應 yolov8/main.py R 鍵）
- Arduino 連線狀態顯示
- 歷史警報 log 區塊

## 啟動

```bash
cd PyQT5
python dashboard.py
```

鍵盤：C=重新校準、Q/ESC=結束
