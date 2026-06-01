# 整合方向建議

目標：將所有功能模組整合，以手機作為最終畫面呈現介面。

---

## 現有模組清單

| 模組 | 狀態 | 功能 |
|------|------|------|
| `yolov8/` + `PyQT5/dashboard.py` | 可運行 | 駕駛行為監控（疲勞 / 手機 / 香菸 / 視線） |
| `Helmet/camera_helmet_detect.py` | 可運行 | 安全帽 + 扣環偵測（Gemma3:27B API） |
| `Helmet/preprocess_head.py` | 可運行 | YOLOv8 頭部裁切前處理 |
| `ResNet/verify_user.py` | 可運行 | 騎乘前駕駛身分驗證 |
| `ResNet/register_user.py` | 可運行 | 駕駛人臉特徵註冊 |

---

## 三種整合方向

### 方案 A：Flask + MJPEG 串流

**架構**
```
Jetson Nano
  └─ Flask server
      ├─ /video_feed  → MJPEG 影像串流
      └─ /state       → JSON polling 狀態

手機瀏覽器 (同 WiFi / 熱點)
  └─ 開啟 http://<jetson-ip>:5000
```

**優點**
- 實作最快（1–2 天可 demo）
- 不需裝 app，任何手機瀏覽器可用
- 程式碼改動最小

**缺點**
- MJPEG 耗頻寬，延遲約 100–300ms
- 狀態需 polling，不夠即時
- 擴展性較差

---

### 方案 B：FastAPI + WebSocket + 前端 SPA（建議）

**架構**
```
Jetson Nano
  └─ FastAPI server
      ├─ WebSocket /ws/video   → base64 JPEG 影像幀
      ├─ WebSocket /ws/state   → 即時偵測狀態 JSON
      └─ REST /api/helmet      → 安全帽偵測結果
         REST /api/verify      → 駕駛身分驗證

手機瀏覽器 / PWA
  └─ 儀表板頁面（仿 dashboard.py 手機版）
      ├─ 左側：即時影像
      └─ 右側：警報等級 + 4 張違規指示卡
```

**優點**
- 前後端完全分離，手機 UI 不受 PyQt5 限制
- WebSocket 雙向即時，狀態更新無延遲
- 可做成 PWA 加到手機主畫面，外觀像 app
- 安全帽偵測、身分驗證都可整進同一個 API
- 架構與現有 Helmet 模組呼叫遠端 API 的模式一致

**缺點**
- 需要寫前端（HTML/CSS/JS 或 React）
- 開發時間約 3–5 天

**前端參考元件**（對應 dashboard.py）

| PyQt5 元件 | 網頁對應 |
|-----------|---------|
| `VideoWidget` | `<canvas>` 或 `<img>` + WebSocket |
| `LevelBar` | CSS 色條 + 動態 class |
| `NotificationCard` | Bootstrap Card 或純 CSS |
| `StatusPanel` | CSS Grid 數值表格 |

---

### 方案 C：ngrok + 現有 dashboard（快速 demo）

**架構**
```
Jetson Nano
  └─ 改 dashboard.py 輸出為網頁 → ngrok 穿透

外網手機瀏覽器
  └─ 開啟 ngrok URL
```

**優點**
- 不需改太多程式碼
- 外網也能連（不限同 WiFi）

**缺點**
- ngrok 免費版有流量限制
- 非長期部署方案
- PyQt5 架構本身不適合直接轉網頁

---

## 建議：採用方案 B

```
騎乘前                    騎乘中
─────                    ──────
1. verify_user           1. dashboard 監控執行緒全開
   → 身分確認              (yolov8 + mediapipe + decision)
2. camera_helmet_detect  2. WebSocket 推送影像 + 狀態
   → 安全帽 + 扣環確認    3. 手機瀏覽器顯示即時儀表板
3. 通過後才允許啟動監控
```

---

## 整合優先順序

| 優先 | 項目 | 說明 |
|------|------|------|
| P1 | FastAPI server 骨架 | 先讓 WebSocket 能推影像和狀態 |
| P2 | 手機前端基本頁面 | 影像 + 警報等級 + 4 張指示卡 |
| P3 | 安全帽偵測整入 API | 騎乘前驗證流程 |
| P4 | 身分驗證整入 API | 綁定駕駛人臉 |
| P5 | PWA 設定 | 加到主畫面、全螢幕模式 |

---

## 技術選型備注

- **後端**：`fastapi` + `uvicorn`，Jetson Nano 上比 Flask 更省資源
- **影像傳輸**：WebSocket 傳 base64 JPEG（約 30–50KB/frame），每秒 15 幀足夠
- **前端**：純 HTML/CSS/JS 即可，不一定要 React；手機全螢幕用 `viewport` meta 設定
- **網路**：Jetson 開熱點，手機連入，固定 IP `192.168.x.1`
