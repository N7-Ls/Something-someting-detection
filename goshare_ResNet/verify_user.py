"""
任務 B：單次即時身分驗證與記憶體釋放
透過 Webcam 擷取單幀畫面，與 user_feature.npy 比對後釋放模型記憶體。
相容 Python 3.6+
"""

import numpy as np
import cv2
from uniface import RetinaFace, ArcFace

SIMILARITY_THRESHOLD = 0.4  # 可依需求調整


def cosine_similarity(vec_a, vec_b):
    """計算兩向量的餘弦相似度。"""
    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def capture_frame(camera_index=0):
    """從 Webcam 擷取單幀畫面。"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("無法開啟攝影機 (index={})".format(camera_index))
    # 關閉自動曝光並手動設定，避免過曝（值可依實際環境調整）
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = 手動模式（部分驅動為 0.25）
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)        # 負值表示較短曝光時間
    # 暖機：丟棄前幾幀，避免攝影機初始化時的黑畫面
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise IOError("無法從攝影機讀取畫面。")
    return frame


def verify_user(feature_path="user_feature.npy", camera_index=0,
                threshold=SIMILARITY_THRESHOLD):
    """
    擷取當下畫面並與基準特徵比對，完成後釋放模型記憶體。

    Args:
        feature_path (str): 基準特徵向量檔案路徑，預設 'user_feature.npy'
        camera_index (int): 攝影機編號，預設 0
        threshold (float): 餘弦相似度閾值，預設 0.4

    Returns:
        bool: 驗證通過回傳 True，否則回傳 False
    """
    detector = None
    recognizer = None
    result = False

    try:
        base_features = np.load(feature_path)

        frame = capture_frame(camera_index)

        detector = RetinaFace()
        recognizer = ArcFace()

        faces = detector.detect(frame)

        if not faces:
            print("[verify_user] 未偵測到人臉，驗證失敗。")
            result = False
        else:
            current_features = recognizer.get_normalized_embedding(frame, faces[0].landmarks)
            similarity = cosine_similarity(current_features, base_features)
            print("[verify_user] 相似度：{:.4f}，閾值：{}".format(similarity, threshold))

            if similarity >= threshold:
                print("[verify_user] 驗證通過。")
                result = True
            else:
                print("[verify_user] 相似度不足，驗證失敗。")
                result = False

    except Exception as e:
        print("[verify_user] 發生錯誤：{}".format(e))
        result = False

    finally:
        # 無論成功或失敗，皆釋放模型記憶體
        if detector is not None:
            del detector
        if recognizer is not None:
            del recognizer
        print("[verify_user] 模型記憶體已釋放。")

    return result


if __name__ == "__main__":
    success = verify_user()
    print("[verify_user] 驗證結果：{}".format(success))
