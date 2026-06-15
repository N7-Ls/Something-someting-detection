"""
執行緒 1：影像擷取。
將每個影格同時送進 queue_pose / queue_face / queue_cig / queue_display。
"""
import cv2
import logging
import time
from config import CAMERA_INDEX
from state  import queue_pose, queue_face, queue_cig, queue_display, stop_event
from utils  import put_nowait_safe


def thread_capture():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logging.error("無法開啟攝影機")
        stop_event.set()
        return

    frame_id = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.warning("讀取影格失敗，跳過")
                time.sleep(0.01)
                continue
            frame  = cv2.rotate(frame, cv2.ROTATE_180)
            ts     = time.perf_counter()
            packet = (frame_id, ts, frame)
            put_nowait_safe(queue_pose,    packet)
            put_nowait_safe(queue_face,    packet)
            put_nowait_safe(queue_cig,     packet)
            put_nowait_safe(queue_display, (frame_id, frame))
            frame_id += 1
    except Exception as e:
        logging.error(f"擷取執行緒例外：{e}")
        stop_event.set()
    finally:
        cap.release()
        for _ in range(2):
            put_nowait_safe(queue_pose, None)
            put_nowait_safe(queue_face, None)
            put_nowait_safe(queue_cig,  None)
        put_nowait_safe(queue_display, None)
        logging.info("影像擷取執行緒結束")
