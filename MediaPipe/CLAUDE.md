# 安全帽偵測模組

臉部幾何定位（MediaPipe 468 點）+ 材質分類（MobileNetV2）判斷頭頂是否戴安全帽。

## 流程

1. MediaPipe 鎖定雙眼與前額，動態計算頭頂 ROI（隨頭部轉動位移）
2. ROI 輸入 MobileNetV2 二元分類：「安全帽硬殼」vs「頭髮/皮膚」
3. 偵測到人臉且 ROI=安全帽 → 解鎖；否則維持閉鎖

## 控制邏輯

- 預設狀態：動力鎖定
- 預留 `send_to_arduino()` 函式輸出解鎖/閉鎖訊號

## 框架

OpenCV + MediaPipe + TensorFlow/Keras（`train.py` 訓練 MobileNetV2 二元分類器）
