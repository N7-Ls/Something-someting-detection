# Helmet — 安全帽偵測模組

## 檔案

| 檔案 | 角色 | 用途 |
|------|------|------|
| `preprocess_head.py` | **被 `web/app.py` 直接 import** | `crop_head(image_path)`：用根目錄 `yolov8n-pose.pt` 偵測人物 + 肩膀關鍵點，裁出頭頸區域（含安全帽/扣帶），支援 HEIC/HEIF。偵測失敗時回傳原圖路徑。 |
| `camera_helmet_detect.py` | 獨立 demo（cv2 視窗） | 每 `CHECK_INTERVAL=8` 秒自動裁頭並呼叫 LLM API，畫面左上角顯示結果。**Prompt/API 設定與 `web/app.py` 重複維護**，調整判斷邏輯時兩處都要改。 |

## 裁切策略 (`crop_head`)

- 選人：取畫面中 bbox 面積最大的人（最靠近鏡頭=駕駛本人），避免旁人入鏡時誤判
- 上邊界：YOLO person bbox 頂部
- 下邊界：肩膀關鍵點 y 座標 + 15%（讓下巴扣帶入鏡）；無肩膀時取 person bbox 上半部
- 左右：bbox 左右邊界各內縮 8%
- 輸出：`head_<原檔名>.jpg`，與原圖同目錄

## LLM 安全帽判斷

呼叫 Gemma3:27B（經 ngrok，`HELMET_API_URL`/`HELMET_API_KEY`），
Prompt 要求輸出三選一：
- `Result: No helmet`
- `Result: Helmet on, strap fastened`
- `Result: Helmet on, strap unfastened`

`web/app.py._analyze_helmet()` 依此字串對應到
`no_helmet` / `unfastened` / `fastened`，只有 `fastened` 才能進入行駛監控。

## 依賴

`ultralytics`（YOLOv8 pose）、`pillow` + `pillow-heif`（HEIC 支援）、
`requests`（呼叫 LLM API）。模型檔 `yolov8n-pose.pt` 在專案根目錄。
