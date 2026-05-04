# 共享機車主動式安全防護系統：身分驗證模組開發規格書

## 1. 系統架構與流程定位
本模組為防護系統的前置啟動階段，針對邊緣運算裝置 (NVIDIA Jetson Nano) 設計。
執行流程為嚴格的**序列式 (Sequential)**，以避免記憶體溢出 (OOM)：
1. **[階段一：本模組執行]** 系統啟動 -> 擷取攝影機單張畫面 -> 執行單樣本身分驗證 (One-Shot Face Verification)。
2. **[階段二：釋放與交接]** 驗證通過後 -> 徹底釋放本模組的偵測器與辨識器記憶體 -> 回傳成功訊號，觸發系統後續的 YOLOv8 + MediaPipe 雙模型並行監控與 PyQt5 儀表板 (後續部分目前不用實作，做到釋放模組記憶體即可)。

## 2. 開發環境與依賴套件 (重要限制)
程式碼必須確保絕對的向後相容性，以配合最終部署的硬體環境：
* **開發與測試環境：** 個人電腦 (Python 3.9+)
* **最終部署環境：** NVIDIA Jetson Nano (JetPack 4.6，系統綁定 **Python 3.6**)
* **語法限制：** 絕對禁止使用 Python 3.7 (含) 以上才支援的新語法（例如 Walrus 運算子 `:=`、內建 `dataclasses` 等），確保程式碼能在 Python 3.6 順利執行。
* **已安裝套件：** 僅限使用 `uniface`, `opencv-python`, `numpy`。請勿自行引入未列出的第三方深度學習框架 (如 `torch`, `tensorflow`, `deepface`)。

## 3. 演算法與 API 呼叫規範 (強制要求)
為確保在 Jetson Nano 上的推論效能，本模組直接採用原生支援 ONNX Runtime 的開源套件 `uniface`。
請完全依照以下 `uniface` 的 API 邏輯進行開發，無需手動編寫權重下載或模型建構代碼：
* **初始化：**
  `from uniface.models import RetinaFace, ArcFace`
  `detector = RetinaFace()`
  `recognizer = ArcFace()`
* **推論流程：**
  1. 取得 Bounding Box 與特徵點：`boxes, landmarks = detector.detect(img)`
  2. 提取特徵向量：`features = recognizer.get_normalized_embedding(img, landmarks[0])`
* **相似度計算：** 餘弦相似度 (Cosine Similarity)

## 4. 核心功能需求 (請產出以下兩個 Python 腳本)

請協助撰寫模組化的 Python 腳本，需具備清晰的 Function 封裝，方便主程式後續呼叫。

### 任務 A：`register_user.py` (用戶基準特徵註冊)
* **輸入：** 讀取指定的本地單張註冊照片 (例如 `base_image.jpg`)。
* **處理流程：**
  1. 呼叫 `RetinaFace` 偵測人臉。
  2. 確認畫面中「有且僅有一張人臉」，若無人臉或多張人臉需拋出明確錯誤提示或回傳例外。
  3. 傳入 `landmarks` 給 `ArcFace` 提取高維度特徵向量。
* **輸出：** 將特徵向量使用 `numpy.save` 儲存為本地檔案 (`user_feature.npy`)。

### 任務 B：`verify_user.py` (單次即時身分驗證與記憶體釋放)
* **輸入：** 透過 OpenCV 啟動 Webcam 擷取當下一幀畫面 (Frame)，並載入 `user_feature.npy`。
* **處理流程：**
  1. 對當下畫面執行 `RetinaFace` 偵測與 `ArcFace` 特徵提取。
  2. 計算「當下特徵向量」與「基準特徵向量」的餘弦相似度。
  3. 判定閾值 (Threshold) 暫設為 `0.4` (提供變數供後續微調)。
* **記憶體管理 (關鍵)：** 無論驗證成功或失敗，流程結束前必須使用 `del detector, recognizer` 釋放模型物件，確保後續階段有充足記憶體。
* **輸出：** 驗證成功回傳 `True`，失敗回傳 `False`。