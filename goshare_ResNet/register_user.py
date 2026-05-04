"""
任務 A：用戶基準特徵註冊
讀取本地單張註冊照片，提取人臉特徵向量後儲存為 user_feature.npy。
相容 Python 3.6+
"""

import numpy as np
import cv2
from uniface import RetinaFace, ArcFace


def register_user(image_path, output_path="user_feature.npy"):
    """
    從指定照片中提取人臉特徵向量並儲存。

    Args:
        image_path (str): 註冊照片路徑，例如 'base_image.jpg'
        output_path (str): 輸出特徵向量檔案路徑，預設 'user_feature.npy'

    Returns:
        str: 成功時回傳儲存路徑

    Raises:
        FileNotFoundError: 找不到指定圖片
        ValueError: 畫面中無人臉或超過一張人臉
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("找不到圖片：{}".format(image_path))

    detector = RetinaFace()
    recognizer = ArcFace()

    faces = detector.detect(img)

    if not faces:
        raise ValueError("未偵測到人臉，請確認照片中有清晰的正面人臉。")

    if len(faces) > 1:
        raise ValueError("偵測到 {} 張人臉，註冊照片必須只有一張人臉。".format(len(faces)))

    features = recognizer.get_normalized_embedding(img, faces[0].landmarks)
    np.save(output_path, features)

    print("[register_user] 特徵向量已儲存至：{}".format(output_path))
    return output_path


if __name__ == "__main__":
    register_user("personA_2.jpg")
