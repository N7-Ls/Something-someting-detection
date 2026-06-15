"""
執行緒 2：YOLOv8 推論。
偵測手機（全圖掃描）。

手機模型：yolov8s.pt（COCO class 67 = cell phone）
  - COCO 訓練資料多樣，涵蓋各種角度、遮擋、手持場景

pose/手腕 ROI 偵測已移除：此攝影機角度下 pose 永遠偵測不到手腕/手肘
（wrist_xy 恆為 None），ROI 流程從未被使用，移除可省下每幀約 265ms 的
pose 推論時間。手機偵測改為全程全圖掃描，並提高 imgsz/降低 conf
以補償全圖小目標召回率。

香菸偵測已移至獨立執行緒（thread_cigarette），避免阻塞此處的手機推論。
"""
import logging
import os
import queue
from config import YOLO_PHONE_CONF, YOLO_PHONE_IMGSZ
from state  import queue_pose, queue_decision, stop_event, display_state, display_lock
from utils  import put_nowait_safe
from ultralytics import YOLO

_PHONE_CLS = 67   # COCO cell phone


def thread_yolo():
    _phone_pt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8s.pt")
    try:
        model_phone = YOLO(_phone_pt)
        logging.info(f"手機模型就緒：yolov8s.pt  class={_PHONE_CLS} ({model_phone.names[_PHONE_CLS]})")
    except Exception as e:
        logging.error(f"手機模型載入失敗：{e}")
        return

    phone_cls = _PHONE_CLS

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
            "phone_detected": False, "wrist_xy": None,
        }

        try:
            phone_boxes = []
            res_ph = model_phone(frame, verbose=False,
                                  conf=YOLO_PHONE_CONF, imgsz=YOLO_PHONE_IMGSZ)[0]
            if res_ph.boxes is not None:
                for box in res_ph.boxes:
                    if int(box.cls[0]) != phone_cls:
                        continue
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    conf_v = float(box.conf[0])
                    logging.debug(f"[phone] hit: conf={conf_v:.2f}")
                    result["phone_detected"] = True
                    phone_boxes.append((bx1, by1, bx2, by2, conf_v))

            with display_lock:
                display_state["phone_boxes"]   = phone_boxes
                display_state["yolo_frame_id"] = frame_id

        except Exception as e:
            logging.error(f"YOLOv8 推論例外：{e}")

        put_nowait_safe(queue_decision, result)

    logging.info("YOLOv8 執行緒結束")
