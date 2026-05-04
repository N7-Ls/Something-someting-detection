"""
執行緒 2：YOLOv8 推論。
偵測手機、手腕關鍵點、香菸。

手機模型：yolov8n_phone.pt（IndUSV/yolov8n-mobile-phone，MIT）
  - class 0 = mobile_phone，Precision=88.4%，mAP50=81.6%
  - 專門訓練、多角度，優於 COCO class 67

香菸模型：yolov8n_cigarette.pt（basant18/Smoking-detection-YOLO26s，Apache 2.0）
  - class 0 = smoke，Precision=92.61%
"""
import logging
import os
import queue
import time
import state
from config import (
    YOLO_PHONE_CONF, YOLO_IMGSZ, YOLO_POSE_IMGSZ, YOLO_CIG_CONF,
    CIG_INTERVAL_SEC, PHONE_MAX_AREA_RATIO, PHONE_SQUARE_MIN, PHONE_SQUARE_MAX,
)
from state  import queue_pose, queue_decision, stop_event, display_state, display_lock
from utils  import put_nowait_safe
from ultralytics import YOLO

_PHONE_LOCAL = "yolov8n_phone.pt"
_PHONE_HF    = "IndUSV/yolov8n-mobile-phone"
_PHONE_HF_FILE = "yolov8n-mobile-phone.pt"

_CIG_LOCAL = "yolov8n_cigarette.pt"
_CIG_HF    = "basant18/Smoking-detection-YOLO26s"
_CIG_CLASS = 0


def _hf_download(repo_id, filename, local_path):
    """從 Hugging Face 下載模型並快取至本地。"""
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        pt = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(pt, local_path)
    except ImportError:
        raise RuntimeError("需要 huggingface_hub：pip install huggingface_hub")


def _load_phone_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, _PHONE_LOCAL)
    if os.path.exists(local_path):
        model = YOLO(local_path)
        logging.info(f"手機模型載入：{local_path}，classes={model.names}")
        return model
    logging.info(f"從 Hugging Face 下載手機模型：{_PHONE_HF}")
    _hf_download(_PHONE_HF, _PHONE_HF_FILE, local_path)
    model = YOLO(local_path)
    logging.info(f"手機模型已快取至：{local_path}")
    return model


def _load_cig_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, _CIG_LOCAL)
    if os.path.exists(local_path):
        model = YOLO(local_path)
        logging.info(f"香菸模型載入：{local_path}，classes={model.names}")
        return model
    logging.info(f"從 Hugging Face 下載香菸模型：{_CIG_HF}")
    _hf_download(_CIG_HF, "weights/best.pt", local_path)
    model = YOLO(local_path)
    logging.info(f"香菸模型已快取至：{local_path}")
    return model


def thread_yolo():
    try:
        model_pose = YOLO("yolov8n-pose.pt")
    except Exception as e:
        logging.error(f"pose 模型載入失敗：{e}")
        return

    try:
        model_phone = _load_phone_model()
        logging.info("手機模型就緒")
    except Exception as e:
        logging.warning(f"手機專用模型載入失敗（{e}），fallback 至 COCO class 67")
        model_phone = YOLO("yolov8n.pt")

    try:
        model_cig = _load_cig_model()
        state.cig_model_available = True
        logging.info("香菸模型就緒")
    except Exception as e:
        logging.warning(f"香菸模型載入失敗（{e}），改用手腕靠嘴 fallback（持續 3 秒）")
        model_cig = None

    # 判斷手機模型是否為專用模型（class 0）或 COCO fallback（class 67）
    phone_cls = 0 if (model_phone.names.get(0) == "mobile_phone") else 67

    _last_cig_time = 0.0

    while not stop_event.is_set():
        try:
            packet = queue_pose.get(timeout=1.0)
        except queue.Empty:
            continue
        if packet is None:
            break

        frame_id, ts, frame = packet
        result = {
            "source": "yolo", "frame_id": frame_id, "timestamp": ts,
            "phone_detected": False, "cigarette_detected": False, "wrist_xy": None,
        }

        try:
            phone_boxes, wrists, cig_boxes = [], [], []
            fh, fw_img = frame.shape[:2]
            frame_area = fh * fw_img

            res_phone = model_phone(frame, verbose=False, conf=YOLO_PHONE_CONF, imgsz=YOLO_IMGSZ)[0]
            if res_phone.boxes is not None:
                for box in res_phone.boxes:
                    if int(box.cls[0]) != phone_cls:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bw, bh = x2 - x1, y2 - y1
                    # 面積過大（軀幹）或近正方形（非手機）→ 跳過
                    if bw * bh > frame_area * PHONE_MAX_AREA_RATIO:
                        continue
                    aspect = bw / bh if bh > 0 else 0
                    if PHONE_SQUARE_MIN < aspect < PHONE_SQUARE_MAX:
                        continue
                    result["phone_detected"] = True
                    phone_boxes.append((x1, y1, x2, y2, float(box.conf[0])))

            res_pose = model_pose(frame, verbose=False, imgsz=YOLO_POSE_IMGSZ)[0]
            if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                kps_xy   = res_pose.keypoints[0].xy.cpu().numpy()
                kps_conf = res_pose.keypoints[0].conf.cpu().numpy()
                for idx in [9, 10]:
                    if idx < len(kps_xy) and kps_conf[idx] > 0.3:
                        x, y = kps_xy[idx]
                        if x > 0 and y > 0:
                            wrists.append((float(x), float(y)))
                if wrists:
                    result["wrist_xy"] = wrists

            # 香菸模型依時間間隔推論，避免丟幀導致間隔不穩定
            _now = time.perf_counter()
            if state.cig_model_available and model_cig is not None:
                if _now - _last_cig_time >= CIG_INTERVAL_SEC:
                    _last_cig_time = _now

                    # 動態 ROI：以臉部中心裁出「頭頂至胸部」區域再送入香菸模型
                    with display_lock:
                        _mouth_xy  = display_state.get("mouth_xy")
                        _face_w    = display_state.get("face_width")

                    roi_frame  = frame
                    roi_offset = (0, 0)
                    if _mouth_xy and _face_w and _face_w > 20:
                        mx, my = _mouth_xy
                        fw_half = _face_w
                        rx1 = max(0, int(mx - fw_half))
                        rx2 = min(fw_img, int(mx + fw_half))
                        ry1 = max(0, int(my - fw_half * 1.5))
                        ry2 = min(fh, int(my + fw_half))
                        if (rx2 - rx1) > 20 and (ry2 - ry1) > 20:
                            roi_frame  = frame[ry1:ry2, rx1:rx2]
                            roi_offset = (rx1, ry1)

                    res_cig = model_cig(roi_frame, verbose=False, conf=YOLO_CIG_CONF)[0]
                    cig_boxes = []
                    if res_cig.boxes is not None:
                        for box in res_cig.boxes:
                            if int(box.cls[0]) != _CIG_CLASS:
                                continue
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            # 座標還原至全幅影格
                            x1 += roi_offset[0]; y1 += roi_offset[1]
                            x2 += roi_offset[0]; y2 += roi_offset[1]
                            cig_boxes.append((x1, y1, x2, y2, float(box.conf[0])))
                    with display_lock:
                        display_state["cig_boxes"] = cig_boxes
                else:
                    with display_lock:
                        cig_boxes = list(display_state["cig_boxes"])
                if cig_boxes:
                    result["cigarette_detected"] = True

            with display_lock:
                display_state["phone_boxes"]   = phone_boxes
                display_state["cig_boxes"]     = cig_boxes
                display_state["wrist_xy"]      = wrists
                display_state["yolo_frame_id"] = frame_id

        except Exception as e:
            logging.error(f"YOLOv8 推論例外：{e}")

        put_nowait_safe(queue_decision, result)

    logging.info("YOLOv8 執行緒結束")
