"""
執行緒：香菸偵測。
與 pose/手機流程解耦，避免推論耗時阻塞 thread_yolo（手機/pose）。

香菸模型：yolov8n_cigarette.pt（basant18/Smoking-detection-YOLO26s，Apache 2.0）
  - class 0 = smoke，Precision=92.61%
"""
import logging
import os
import queue
import time
import state
from config import YOLO_CIG_CONF, YOLO_CIG_IMGSZ, CIG_INTERVAL_SEC
from state  import queue_cig, stop_event, display_state, display_lock
from ultralytics import YOLO

_CIG_LOCAL = "yolov8n_cigarette.pt"
_CIG_HF    = "basant18/Smoking-detection-YOLO26s"
_CIG_CLASS = 0


def _hf_download(repo_id, filename, local_path):
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        pt = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(pt, local_path)
    except ImportError:
        raise RuntimeError("需要 huggingface_hub：pip install huggingface_hub")


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


def thread_cigarette():
    try:
        model_cig = _load_cig_model()
        state.cig_model_available = True
        logging.info("香菸模型就緒")
    except Exception as e:
        logging.warning(f"香菸模型載入失敗（{e}），抽菸偵測改用手腕靠嘴 fallback")
        return

    _last_cig_time = 0.0

    while not stop_event.is_set():
        try:
            packet = queue_cig.get(timeout=1.0)
        except queue.Empty:
            continue
        if packet is None:
            break

        _, _, frame = packet

        now = time.perf_counter()
        if now - _last_cig_time < CIG_INTERVAL_SEC:
            continue
        _last_cig_time = now

        try:
            fh, fw_img = frame.shape[:2]

            # 動態 ROI：以臉部中心裁出「頭頂至胸部」區域再送入香菸模型
            with display_lock:
                mouth_xy = display_state.get("mouth_xy")
                face_w   = display_state.get("face_width")

            roi_frame  = frame
            roi_offset = (0, 0)
            if mouth_xy and face_w and face_w > 20:
                mx, my  = mouth_xy
                fw_half = face_w
                rx1 = max(0, int(mx - fw_half))
                rx2 = min(fw_img, int(mx + fw_half))
                ry1 = max(0, int(my - fw_half * 1.5))
                ry2 = min(fh, int(my + fw_half))
                if (rx2 - rx1) > 20 and (ry2 - ry1) > 20:
                    roi_frame  = frame[ry1:ry2, rx1:rx2]
                    roi_offset = (rx1, ry1)

            res_cig = model_cig(roi_frame, verbose=False, conf=YOLO_CIG_CONF,
                                imgsz=YOLO_CIG_IMGSZ)[0]
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

        except Exception as e:
            logging.error(f"香菸推論例外：{e}")

    logging.info("香菸偵測執行緒結束")
