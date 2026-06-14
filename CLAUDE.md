# GoShare 駕駛安全監控系統

機車駕駛行為即時監控，目標部署 NVIDIA Jetson Nano。完整架構/套件/啟動方式見 [README.md](README.md)。

## 主要入口

`web/app.py`（Flask，預設 port 5001——5000 在 Windows 開發機上常被
PerformanceMonitor.exe 佔用）。三階段流程：人臉辨識 → 安全帽確認 → 行駛監控，
全程透過 `/video_feed` MJPEG 串流同一支攝影機畫面。

## 模組地圖（哪些東西被主流程用到）

| 資料夾 | 角色 | 與 web/app.py 的關係 |
|--------|------|----------------------|
| `web/` | Flask 整合介面（**主流程**） | 入口本身 |
| `ResNet/` | 人臉辨識 (RetinaFace+ArcFace) | app.py 內**重寫了一份**同樣邏輯；`register_user.py`/`verify_user.py` 是獨立 CLI 版本，未被 import |
| `Helmet/` | 安全帽偵測 | app.py 直接 `import preprocess_head.crop_head`；`camera_helmet_detect.py` 是獨立 demo（同樣邏輯但跑 cv2 視窗） |
| `yolov8/` | 行駛監控四執行緒管線 | app.py 直接 import `state`/`display`/四條 thread；`main.py` 是獨立 cv2 測試入口 |
| `PyQT5/` | 本機 GUI 儀表板 | ❌ 與 web 流程無關，獨立執行 yolov8 管線時使用 |
| `MediaPipe/` | 安全帽偵測（實驗性，MediaPipe+MobileNetV2） | ❌ 未整合，被 `Helmet/` 的 LLM 方案取代 |
| `_trash/` | 已淘汰的舊版程式碼（含 `monitor.py` 的 Arduino 實作） | ❌ 不參與執行 |

## 重複實作注意事項

人臉驗證與安全帽裁切邏輯目前在 `web/app.py` 與
`ResNet/`、`Helmet/` 各有一份。**改動門檻值/演算法時兩邊都要看**，
或考慮之後改成共用 import。

各資料夾細節見其下的 `CLAUDE.md`。
