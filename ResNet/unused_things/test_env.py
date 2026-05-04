from deepface import DeepFace
import cv2

print("=== 環境測試開始 ===")

# 1. 測試模型載入 (ArcFace 骨幹即為 ResNet34)
try:
    # 第一次執行時，Deepface 會自動從網路下載 ArcFace 的預訓練權重檔 (約幾百MB)
    model = DeepFace.build_model("ArcFace")
    print("✅ ArcFace (ResNet34) 模型載入成功！")
except Exception as e:
    print(f"❌ ArcFace 模型載入失敗: {e}")

# 2. 測試偵測器載入 (RetinaFace)
try:
    from retinaface import RetinaFace
    print("✅ RetinaFace 偵測器模組載入成功！")
except Exception as e:
    print(f"❌ RetinaFace 模組載入失敗: {e}")

print("=== 環境測試完畢 ===")