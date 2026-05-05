# PyQt5 儀表板

取代 `yolov8/main.py` 的 cv2.imshow，提供圖形化監控介面。
直接匯入 yolov8 四執行緒，讀取 `state.py` 共享狀態渲染 UI。

## 架構

```
dashboard.py
  ├─ sys.path.insert → ../yolov8/
  ├─ 啟動 4 daemon threads（同 yolov8/main.py）
  ├─ QTimer(33ms) → _refresh()
  │   ├─ 從 queue_display 拉 (frame_id, frame)
  │   ├─ 呼叫 display.annotate()
  │   └─ 讀 display_state 更新 Widget
  └─ QMainWindow
      ├─ VideoWidget（左側）
      └─ 右側：LevelBar / AlertIcon×4 / StatusPanel / 按鈕
```

## 指示燈狀態

| 顏色 | 意義 |
|------|------|
| 灰 `#3a3a3a` | 正常 |
| 橘 `#e67e22` | 瞬間條件成立（未計時） |
| 紅 `#e74c3c` | 計時器觸發（真正警報） |

觸發來源：`display_state["alert_flags"]`（由 thread_decision.py 寫出）

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
