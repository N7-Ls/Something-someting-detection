# 身分驗證模組

邊緣運算（Jetson Nano）單次臉部驗證，驗證後**立即釋放模型記憶體**再交棒給監控模組。

## 限制

- 目標部署：Jetson Nano JetPack 4.6（Python 3.6），禁用 3.7+ 語法（walrus `:=`、dataclasses 等）
- 只能用：`uniface`、`opencv-python`、`numpy`，禁止引入 torch/tensorflow/deepface

## uniface API

```python
from uniface import RetinaFace, ArcFace
detector, recognizer = RetinaFace(), ArcFace()
boxes, landmarks = detector.detect(img)
feat = recognizer.get_normalized_embedding(img, landmarks[0])
# 結束後：del detector, recognizer
```

相似度用餘弦相似度，門檻 0.4（可調）。

## 檔案

- `register_user.py`：讀單張照片 → 確認唯一人臉 → 儲存特徵向量至 `user_feature.npy`
- `verify_user.py`：Webcam 單幀 → 比對 `user_feature.npy` → 回傳 True/False → 釋放模型
