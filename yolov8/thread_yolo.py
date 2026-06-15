"""
執行緒 2：YOLOv8 推論。
偵測手機、手腕關鍵點。

手機模型：yolov8s.pt（COCO class 67 = cell phone）
  - COCO 訓練資料多樣，涵蓋各種角度、遮擋、手持場景
  - 配合 ROI（手腕/手肘周圍）降低誤報

香菸偵測已移至獨立執行緒（thread_cigarette），避免阻塞此處的手機/pose推論。
"""
import logging
import os
import queue
from config import (
    YOLO_PHONE_CONF, YOLO_POSE_IMGSZ, YOLO_PHONE_IMGSZ, PHONE_ROI_PAD,
)
from state  import queue_pose, queue_decision, stop_event, display_state, display_lock
from utils  import put_nowait_safe
from ultralytics import YOLO

_PHONE_CLS = 67   # COCO cell phone


def thread_yolo():
    try:
        model_pose = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-pose.pt"))
    except Exception as e:
        logging.error(f"pose 模型載入失敗：{e}")
        return

    # COCO yolov8s：手持遮擋場景比 IndUSV nano 模型更可靠
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
            phone_boxes, wrists = [], []
            fh, fw_img = frame.shape[:2]

            # ── Step 1: Pose → 手腕 + 手肘 ──────────────────────────────────
            # 注意：若攝影機視角讓 pose 永遠偵測不到關節（wrists=0, elbows=0），
            # 可將下方 pose 推論整段註解掉以省去每幀 50-100ms，
            # 手機偵測會直接走全圖 fallback。
            elbows = []
            res_pose = model_pose(frame, verbose=False, imgsz=YOLO_POSE_IMGSZ)[0]
            if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                kps_xy   = res_pose.keypoints[0].xy.cpu().numpy()
                kps_conf = res_pose.keypoints[0].conf.cpu().numpy()
                for idx in [9, 10]:          # 手腕
                    if idx < len(kps_xy) and kps_conf[idx] > 0.10:
                        x, y = kps_xy[idx]
                        if x > 0 and y > 0:
                            wrists.append((float(x), float(y)))
                for idx in [7, 8]:           # 手肘
                    if idx < len(kps_xy) and kps_conf[idx] > 0.13:
                        x, y = kps_xy[idx]
                        if x > 0 and y > 0:
                            elbows.append((float(x), float(y)))
            if wrists:
                result["wrist_xy"] = wrists

            # ── Step 2: Phone ROI ─────────────────────────────────────────────
            # 手腕可見 → ROI 以手腕為中心，往上延伸（手機在手腕上方）
            # 手腕不可見但手肘可見 → ROI 以手肘為中心，往下延伸（手機在手肘下方）
            pad = int(min(fh, fw_img) * PHONE_ROI_PAD)

            def _phone_roi_scan(ax, ay, extend_down: bool):
                if extend_down:
                    rx1 = max(0,      int(ax - pad * 1.1))
                    rx2 = min(fw_img, int(ax + pad * 1.1))
                    ry1 = max(0,      int(ay - pad * 0.5))
                    ry2 = min(fh,     int(ay + pad * 2.5))
                else:
                    rx1 = max(0,      int(ax - pad * 1.3))
                    rx2 = min(fw_img, int(ax + pad * 1.3))
                    ry1 = max(0,      int(ay - pad * 1.8))
                    ry2 = min(fh,     int(ay + pad * 1.5))
                if (rx2 - rx1) < 32 or (ry2 - ry1) < 32:
                    return
                roi    = frame[ry1:ry2, rx1:rx2]
                res_ph = model_phone(roi, verbose=False, conf=YOLO_PHONE_CONF,
                                     imgsz=YOLO_PHONE_IMGSZ)[0]
                if res_ph.boxes is None:
                    return
                for box in res_ph.boxes:
                    if int(box.cls[0]) != phone_cls:
                        continue
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    bx1 += rx1; by1 += ry1; bx2 += rx1; by2 += ry1
                    conf_v = float(box.conf[0])
                    logging.debug(
                        f"Phone ROI hit: conf={conf_v:.2f} "
                        f"anchor=({'elbow' if extend_down else 'wrist'})"
                        f"({ax:.0f},{ay:.0f})"
                    )
                    result["phone_detected"] = True
                    phone_boxes.append((bx1, by1, bx2, by2, conf_v))

            if wrists:
                for wx, wy in wrists:
                    _phone_roi_scan(wx, wy, extend_down=False)
            elif elbows:
                # 手肘 fallback：只取畫面上方 3/4 以內的手肘（手有抬起才有意義）
                for ex, ey in elbows:
                    if ey < fh * 0.75:
                        _phone_roi_scan(ex, ey, extend_down=True)
            else:
                # 全畫面 fallback：關節完全未偵測時，掃全幅畫面
                res_fb = model_phone(frame, verbose=False,
                                     conf=YOLO_PHONE_CONF, imgsz=YOLO_PHONE_IMGSZ)[0]
                if res_fb.boxes is not None:
                    for box in res_fb.boxes:
                        if int(box.cls[0]) != phone_cls:
                            continue
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        conf_v = float(box.conf[0])
                        logging.info(f"[phone] fullframe hit: conf={conf_v:.2f}")
                        result["phone_detected"] = True
                        phone_boxes.append((bx1, by1, bx2, by2, conf_v))

            with display_lock:
                display_state["phone_boxes"]   = phone_boxes
                display_state["wrist_xy"]      = wrists
                display_state["yolo_frame_id"] = frame_id

        except Exception as e:
            logging.error(f"YOLOv8 推論例外：{e}")

        put_nowait_safe(queue_decision, result)

    logging.info("YOLOv8 執行緒結束")
