# 專案：基於多工視覺融合架構之共享機車主動式安全防護系統
# 任務：實作 4.4 多工視覺融合與駕駛行為監控模組

---

## 一、系統現況與架構限制

* 運行環境：NVIDIA Jetson Nano 邊緣運算裝置（開發測試於 Windows + conda `goshare` 環境）。
* 前置作業 (4.2 & 4.3) 已完成並釋放資源。4.4 模組專責行駛中的即時監控。
* 目標：延遲低於 500ms、維持 15 FPS 以上。

---

## 二、實作進度

### ✅ 已完成

| 項目 | 說明 |
|---|---|
| 多執行緒架構 | 4 執行緒 + 4 Queue（capture / yolo / mediapipe / decision） |
| 影像擷取執行緒 | `cv2.VideoCapture`，frame drop 機制，sentinel 退出 |
| YOLOv8 執行緒 | `yolov8n.pt`（手機 conf=0.15）+ `yolov8n-pose.pt`（手腕），det imgsz=320 |
| MediaPipe 執行緒 | Tasks API（0.10+），計算 EAR / Yaw / Pitch / Roll / 嘴部中心 / 臉寬 |
| 決策中心 | Level 1~3 時序計時，grace period 防止單幀中斷重置，高 Level 優先 |
| 攝影機校準 | 啟動 5s 自動收集 pitch 中位數為補償值（C 鍵重置）|
| 畫面顯示 | debug 面板顯示校正後 Ptch*、Face Mesh、BBox、底部警報列 |
| CSV 感測記錄 | 每決策週期寫一行至 `output/monitor_YYYYMMDD_HHMMSS.csv` |
| 影片錄製 | R 鍵開始 / 停止，存為 `output/monitor_YYYYMMDD_HHMMSS.mp4` |
| 優雅退出 | Q / ESC / KeyboardInterrupt，sentinel 清場，CSV 自動 flush |

### ⬜ 尚未實作（rules.txt 4.5 節）

| 項目 | 說明 |
|---|---|
| Arduino 整合 | Level 1 → 蜂鳴器短促音 + 黃色 LED；Level 2/3 → 連續警示音 + 紅燈 |
| PyQt5 儀表板 | Level 2 → 強制顯示 30km/h 速限；Level 3 → 全螢幕 DANGER + 時速歸零 |
| 脫帽偵測 | Level 2 違規之一，目前完全未涵蓋 |

---

## 三、模型說明

| 模型 | 用途 | 狀態 |
|---|---|---|
| `yolov8n.pt` | 手機偵測（COCO class 67），imgsz=320，conf=0.15 | ✅ 自動下載 |
| `yolov8n-pose.pt` | 手腕關鍵點（COCO index 9, 10），imgsz=640 | ✅ 自動下載 |
| `yolov8n_cigarette.pt` | 香菸偵測（自訓練） | ❌ 不存在，fallback 啟用 |
| `face_landmarker.task` | MediaPipe FaceLandmarker（3.7MB） | ✅ 已下載至專案資料夾 |

---

## 四、已修正的 Bug（重要，勿重複引入）

### B1：`head_pose()` 多乘 × 360（致命）
`cv2.RQDecomp3x3` 回傳已是度（degrees），原本程式碼乘以 360 使角度爆炸成萬級。
**修正**：`return angles[1], angles[0], angles[2]`（移除 `* 360`）

### B2：corr_pitch 未 wrap → 雙峰問題
solvePnP 在龍頭攝影機幾何下，raw_pitch 正常騎乘落在 ±180° 邊界附近（約 +173°），
+173° 和 -173° 是同一物理方向，但 `raw - offset` 不 wrap 時得到 -350°，誤觸 distract。
**修正**：加入 `wrap_angle()` 正規化到 (-180°, +180°]：
```python
def wrap_angle(a):
    return ((a + 180.0) % 360.0) - 180.0
```

### B3：corr_pitch 方向相反
solvePnP 在此幾何下，低頭使 raw_pitch 減少（差值為負），
不取負則低頭觸發 DISTRACT、抬頭觸發 PHONE，完全相反。
**修正**：`corr_pitch = -wrap_angle(raw_pitch - cam_off)` → 正值=低頭，負值=抬頭

### B4：phone_bbox 永遠單幀，timer 無法累積
YOLO 手機偵測每次只有 1 幀（在 handlebar 角度下模型不穩定），
而 `check_duration` 條件一 False 立即重置 timer，0.5s 永遠達不到。
**三重修正**：
1. YOLO fallback cache：MediaPipe 幀也使用最近 1s 內的 YOLO 結果
2. `check_duration` 加 grace period（條件消失後等 HOLD 秒才真正重置）
3. `DURATION_PHONE = 0.15s`（短於 YOLO 幀間隔，單幀可觸發）

---

## 五、關鍵參數速查

```python
# ── 感測閾值 ──
YOLO_IMGSZ         = 320    # det 模型推論解析度
YOLO_PHONE_CONF    = 0.15   # 手機 YOLO 信心度門檻（低一點補償 handlebar 視角漏報）
WRIST_MOUTH_RATIO  = 0.55   # 手腕-嘴部距離閾值（相對臉寬）
YAW_PITCH_LIMIT    = 45.0   # Yaw 分心 / corr_pitch 上仰分心閾值（度）
PITCH_PHONE_LIMIT  = 32.0   # 低頭滑手機閾值（corr_pitch 正值，約低頭 29° 觸發）
EAR_THRESHOLD      = 0.20   # 疲勞 EAR 閾值（sleep median≈0.28，留 0.08 緩衝）
FUSE_TIME_WINDOW   = 0.1    # 跨模組時序融合容忍窗口（秒）

# ── 攝影機補償（執行期自動校準，初始估計值）──
PITCH_CAM_OFFSET   = 25.0   # 初始估計，校準後由 _cam_offset 動態更新
CALIB_SECONDS      = 5.0    # 啟動後自動校準秒數

# ── 計時器 ──
DURATION_DISTRACT  = 2.0    # 視線分心持續觸發秒數
DURATION_SMOKE     = 2.0    # 抽菸違規持續觸發秒數（有香菸模型）
DURATION_SMOKE_NOCIG = 3.0  # 抽菸違規持續觸發秒數（無香菸模型）
DURATION_PHONE     = 0.15   # 手機違規持續觸發秒數（短：單幀 YOLO 偵測可觸發）
DURATION_FATIGUE   = 1.5    # 疲勞危險持續觸發秒數
HOLD_DISTRACT      = 0.5    # distract grace period（秒）
HOLD_PHONE         = 0.4    # phone grace period（秒）
HOLD_FATIGUE       = 0.5    # fatigue grace period（秒）
HOLD_SMOKE         = 1.0    # smoke grace period（秒）
```

---

## 六、corr_pitch 校正公式（關鍵）

```
corr_pitch = -wrap_angle(raw_pitch - cam_offset)
```

`cam_offset` 由自動校準決定（正常值約 +170° ~ +180°，因龍頭攝影機朝上）。

| 情境 | raw_pitch | corr_pitch | 判斷 |
|---|---|---|---|
| 正常騎乘 | ≈ +173° | ≈ 0° | OK |
| 低頭 33°（手機） | ≈ +140° | ≈ +37° | PHONE (>32°) |
| 低頭 77°（嚴重） | ≈ +100° | ≈ +77° | PHONE |
| 抬頭 38° | ≈ -145° | ≈ -38° | OK |
| 抬頭 63°（分心） | ≈ -120° | ≈ -63° | DISTRACT (<-45°) |

> **不可移除 wrap_angle 或負號**，否則 distract/phone 方向對調、±180° 邊界跳變。

---

## 七、資料流架構

```
Camera
  │
  ├─ queue_pose (maxsize=4) ──► [執行緒 2] YOLOv8
  │                                  手機 BBox(conf≥0.15) + 手腕關鍵點
  │                                       │
  └─ queue_face (maxsize=4) ──► [執行緒 3] MediaPipe
                                     EAR / 頭部姿態 / 嘴部中心 / 臉寬
                                           │
                               queue_decision (maxsize=8)
                                           │
                                  [執行緒 4] 決策中心
                                  ├─ 5s 自動校準（收集 pitch 中位數）
                                  ├─ YOLO fallback cache（1s 內）
                                  ├─ corr_pitch = -wrap_angle(raw - offset)
                                  ├─ check_duration + grace period
                                  ├─ CSV 逐行寫入
                                  └─ Level 1 / 2 / 3 介入
                                           │
                               display_state (lock 保護)
                                           │
                               [主執行緒] 畫面標註 + imshow + 影片錄製
```

---

## 八、介入邏輯

| Level | 事件 | 觸發條件 | 持續時間 |
|---|---|---|---|
| 1 | 視線分心 | \|Yaw\| > 45° 或 corr\_pitch < -45°（側轉或抬頭）| 2.0s |
| 2 | 抽菸違規 | 香菸偵測 + 手腕靠嘴 / 手腕靠嘴（無模型） | 2.0 / 3.0s |
| 2 | 手機違規 | phone\_bbox OR corr\_pitch > 32° OR 手腕舉至臉部 | 0.15s |
| 3 | 疲勞危險 | EAR < 0.20 持續 1.5s | 1.5s |

高 Level 優先：fatigue > smoke > phone > distract

---

## 九、鍵盤操作 & 輸出

| 鍵 | 功能 |
|---|---|
| `R` | 切換錄影（`output/monitor_*.mp4`） |
| `C` | 重新校準攝影機 pitch 補償值 |
| `Q` / `ESC` | 結束程式 |

CSV 欄位：`timestamp, raw_pitch, corr_pitch, yaw, roll, ear, cam_offset, phone_bbox, phone_gaze, phone_wrist, smoke, fatigue, distract, alert_level, alert_msg`

---

## 十、目前已知問題

### 🔴 高優先

1. **香菸模型不存在**：`yolov8n_cigarette.pt` 未訓練，fallback 為「手腕靠近嘴部持續 3 秒」，精確度低。

2. **Arduino / PyQt5 完全未整合**：所有介入僅為 `print()`，無實際聲光。

### 🟡 中優先

3. **手機偵測仍不穩定**：`yolov8n.pt` 在 handlebar 俯角下容易漏報，已降 conf 至 0.15 + fallback cache + grace period 緩解，但根本解法需針對性微調模型。

4. **solvePnP 精度**：使用近似 3D 臉型點，Pitch 有系統性偏移（由自動校準補償），Roll 值雜訊較大（目前未使用 Roll 做判斷）。

5. **抽菸手腕 fallback 容易誤觸**：騎士自然舉手（如撥髮、擦臉）可能觸發手腕靠嘴條件。
