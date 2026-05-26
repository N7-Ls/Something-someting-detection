import cv2
import os
import numpy as np
from PIL import Image
import pillow_heif
from ultralytics import YOLO

pillow_heif.register_heif_opener()

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n-pose.pt")
    return _model

# COCO keypoint 索引
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6


def _read_image(image_path: str):
    """讀圖片為 BGR numpy array，支援 HEIC/HEIF。"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in {".heic", ".heif"}:
        pil_img = Image.open(image_path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img = cv2.imread(image_path)
    return img


def crop_head(image_path: str) -> str:
    """
    裁切策略：
    - 上方：person bounding box 頂部（YOLO 會緊貼安全帽頂）
    - 下方：肩膀關鍵點位置（讓下巴＋吊環都可見）
    - 左右：person bounding box 左右邊界

    Returns:
        裁切後的圖片路徑（JPEG）。若偵測失敗，回傳原始路徑。
    """
    img = _read_image(image_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片：{image_path}")

    img_h, img_w = img.shape[:2]
    model = _get_model()
    results = model(img, verbose=False)

    best_box = None
    best_kps = None
    best_conf = 0.0

    for r in results:
        if r.keypoints is None:
            continue
        for i, kps in enumerate(r.keypoints.xy):
            conf = float(r.boxes.conf[i]) if r.boxes is not None else 0.0
            if conf > best_conf:
                best_conf = conf
                best_box = r.boxes.xyxy[i].tolist()
                best_kps = kps

    if best_box is None:
        print(f"[WARN] 未偵測到人物，使用原圖：{image_path}")
        return image_path

    # person bounding box
    box_x1, box_y1, box_x2, box_y2 = map(int, best_box)

    # 取肩膀 y 座標
    def get_kp(idx):
        x, y = float(best_kps[idx][0]), float(best_kps[idx][1])
        return (x, y) if x > 0 and y > 0 else None

    left_sh  = get_kp(KP_LEFT_SHOULDER)
    right_sh = get_kp(KP_RIGHT_SHOULDER)

    shoulders = [s for s in [left_sh, right_sh] if s is not None]

    if shoulders:
        shoulder_y = sum(s[1] for s in shoulders) / len(shoulders)
        head_height = shoulder_y - box_y1
        # 下方：肩膀位置再往下加 15%，讓吊環有空間
        bottom = int(shoulder_y + head_height * 0.15)
    else:
        # 沒有肩膀資訊，取 person box 上半部
        head_height = box_y2 - box_y1
        bottom = int(box_y1 + head_height * 0.50)

    # 左右稍微往內縮 8%
    box_w = box_x2 - box_x1
    inset = int(box_w * 0.08)

    # 裁切範圍
    x1 = max(0, box_x1 + inset)
    y1 = max(0, box_y1)
    x2 = min(img_w, box_x2 - inset)
    y2 = min(img_h, bottom)

    head_img = img[y1:y2, x1:x2]

    base_dir    = os.path.dirname(image_path)
    base_stem   = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(base_dir, "head_" + base_stem + ".jpg")
    cv2.imwrite(output_path, head_img)

    print(f"[OK] {output_path}  (conf: {best_conf:.2f})")
    return output_path


if __name__ == "__main__":
    import glob

    image_dir = "pic"
    patterns  = ["*.jpg", "*.jpeg", "*.png"]
    images    = []
    for p in patterns:
        images.extend(glob.glob(os.path.join(image_dir, p)))

    images = [f for f in images if not os.path.basename(f).startswith("head_")]

    print(f"找到 {len(images)} 張圖片，開始批次裁切頭部...\n")
    for img_path in images:
        try:
            crop_head(img_path)
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
