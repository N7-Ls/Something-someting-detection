# Web — Flask 整合介面（主流程）

`app.py` 是整個系統的入口。單一 Flask process + 單一鏡頭，跑完整三階段流程。
啟動：`python web/app.py`（port 5001）。

## 狀態機

```
face_verify → helmet_check → driving → (stop_driving 回到 face_verify)
```

`_state`（dict + `_lock`）記錄 `phase` / `message` / `analyzing` /
`face_verified` / `helmet_status` / `alert_level`，前端每 700ms 輪詢
`/api/state`。所有耗時操作（人臉比對、呼叫 LLM）都丟到背景 thread，
用 `analyzing` flag 防止重複觸發。

## 攝影機交接

Phase 1/2 用 Flask 自己的 `_camera_worker`（寫入 `_current_frame`）。
進入 `driving` 前呼叫 `_stop_camera()` 釋放鏡頭，改由
`yolov8/thread_capture` 接手；`stop_driving` 再呼叫 `_start_camera()` 還原。
**兩者不能同時開鏡頭**，這是最容易出錯的地方。

## 與其他模組的整合（重複實作警告）

- **人臉辨識**：`_verify_frame()` 直接呼叫 `uniface` 的 `RetinaFace`/`ArcFace`，
  比對 `ResNet/user_feature.npy`（門檻 `SIMILARITY_THRESH=0.4`）。
  邏輯與 `ResNet/verify_user.py`、`register_user.py` **重複但不共用**。
- **安全帽**：`_analyze_helmet()` 呼叫 `Helmet/preprocess_head.crop_head` 裁頭，
  再送 `HELMET_API_URL`（Gemma3:27B，經 ngrok）。Prompt/解析邏輯與
  `Helmet/camera_helmet_detect.py` **重複但不共用**。
- **行駛監控**：直接 `import` `yolov8/` 的 `state`、`display.annotate`、
  四條 thread 函式（`thread_capture`/`thread_yolo`/`thread_mediapipe`/`thread_decision`），
  不重寫推論邏輯。

## 警報音效

`templates/index.html` 用 Web Audio API 在**瀏覽該頁面的裝置**上播放提示音
（不依賴 Jetson 的音訊輸出，因為 Orin Nano 預設只有 HDMI 音源）。
依 `/api/state` 的 `alert_level` 變化在前端播放：等級越高頻率越高、
重複提醒間隔越短（`BEEP_INTERVAL`/`BEEP_PATTERN`）。
`AudioContext` 需在使用者手勢中建立/resume，故在 `post()`（按鈕點擊）時呼叫
`_ensureAudio()` 解鎖，避免進入 driving 階段後無法自動發聲。

## MJPEG 串流 (`/video_feed`)

- Phase 1/2：直接吐 `_current_frame`
- Phase 3：從 `yolov8.state.queue_display` 取已標註幀，並把
  `yolo_state.display_state["alert_level"]` 同步回 `_state["alert_level"]`

## 路由速查

| 路由 | 用途 |
|------|------|
| `/api/verify_face` | Phase 1 → 觸發人臉比對 |
| `/api/check_helmet` | Phase 2 → 觸發 LLM 安全帽判斷 |
| `/api/start_driving` | Phase 2 → 3，需 `helmet_status == "fastened"` |
| `/api/stop_driving` | Phase 3 → 1，停止 yolov8 四執行緒 |
| `/api/register` | 上傳照片寫入 `ResNet/user_feature.npy`（支援 HEIC） |
