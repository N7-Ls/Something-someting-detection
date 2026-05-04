import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import time

# ==========================================
# 準備工作：請替換成你實際的圖片路徑
# ==========================================
# 實驗一測試圖：找一張多人合照（重現論文 Fig 7 的 RetinaFace 大規模偵測）
CROWD_IMG_PATH = "crowd.jpg" 

# 實驗二測試圖：單樣本驗證（重現論文 Fig 3 的 1對1 比對）
# img1 和 img2 放同一個人的不同照片（測試 True），img1 和 img3 放不同人（測試 False）
IMG1_PATH = "personA_1.jpg" 
IMG2_PATH = "personA_2.jpg" 
IMG3_PATH = "personB_1.jpg" 

print("載入套件完成，開始重現論文實驗...\n")

# ==========================================
# 實驗一：大規模人臉偵測 (對應論文 Fig 7)
# 目標：驗證 RetinaFace 偵測器在複雜環境下提取多張人臉的能力
# ==========================================
print("=== 實驗一：RetinaFace 高精度人臉偵測 ===")
try:
    start_time = time.time()
    # 呼叫 DeepFace 進行特徵提取，強制指定 detector_backend 為 retinaface
    faces = DeepFace.extract_faces(img_path=CROWD_IMG_PATH, detector_backend='retinaface')
    end_time = time.time()
    
    print(f"✅ 成功！RetinaFace 共偵測到 {len(faces)} 張人臉。")
    print(f"⏱️ 偵測耗時: {end_time - start_time:.2f} 秒\n")
    
    # 視覺化：將原始圖片畫上 Bounding Box 來重現論文的視覺效果
    img_cv = cv2.imread(CROWD_IMG_PATH)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    for face_obj in faces:
        # 取得臉部座標區域
        facial_area = face_obj['facial_area']
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']
        # 畫上紅色框線 (對應論文 Fig 6, Fig 7 的標示方式)
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    plt.figure(figsize=(10, 6))
    plt.imshow(img_cv)
    plt.title(f"RetinaFace Detection: {len(faces)} faces")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"❌ 實驗一發生錯誤，請檢查 crowd.jpg 是否存在。詳細錯誤: {e}\n")


# ==========================================
# 實驗二：1對1 身分驗證 (對應論文 Fig 3 與 Table 1 最佳化方案)
# 參數完全對應文獻：
# - 偵測器 (Face Detector) = RetinaFace
# - 模型 (Face Recognition Model) = ResNet34 + ArcFace (在 Deepface 中統稱為 'ArcFace')
# - 相似度度量 (Similarity Metric) = Cosine
# ==========================================
print("=== 實驗二：ResNet34 + ArcFace 身分驗證 ===")
try:
    # 測試 A：同一人的比對 (預期 Verified: True)
    print(">>> 測試 A：同一人不同照片比對")
    result_same = DeepFace.verify(
        img1_path=IMG1_PATH,
        img2_path=IMG2_PATH,
        model_name="ArcFace",          # 使用基於 ResNet34 與 ArcFace loss 的模型
        detector_backend="retinaface", # 使用 RetinaFace 作為前處理偵測器
        distance_metric="cosine"       # 使用餘弦相似度
    )
    
    print(f"比對結果 (Verified): {result_same['verified']}")
    print(f"特徵距離 (Distance): {result_same['distance']:.4f}")
    print(f"判定閾值 (Threshold): {result_same['threshold']}")
    print("-" * 30)

    # 測試 B：不同人的比對 (預期 Verified: False)
    print(">>> 測試 B：不同人照片比對")
    result_diff = DeepFace.verify(
        img1_path=IMG1_PATH,
        img2_path=IMG3_PATH,
        model_name="ArcFace",          
        detector_backend="retinaface", 
        distance_metric="cosine"       
    )
    
    print(f"比對結果 (Verified): {result_diff['verified']}")
    print(f"特徵距離 (Distance): {result_diff['distance']:.4f}")
    print(f"判定閾值 (Threshold): {result_diff['threshold']}")
    print("\n✅ 實驗二重現完成！")

except Exception as e:
    print(f"❌ 實驗二發生錯誤，請檢查測試圖片是否存在。詳細錯誤: {e}")