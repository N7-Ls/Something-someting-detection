"""
GoShare Driver Monitor — 入口點。
啟動 4 條執行緒，主執行緒負責顯示與鍵盤互動。
"""
import datetime
import logging
import os
import threading
import time
import cv2

from config import OUTPUT_DIR
from state  import (
    queue_display, stop_event, recalib_event,
    display_state, display_lock,
)
from display          import annotate
from thread_capture   import thread_capture
from thread_yolo      import thread_yolo
from thread_mediapipe import thread_mediapipe
from thread_decision  import thread_decision
from utils            import put_nowait_safe

logging.basicConfig(level=logging.INFO, format="[%(threadName)s] %(message)s")


def main():
    threads = [
        threading.Thread(target=thread_capture,   name="Capture",    daemon=True),
        threading.Thread(target=thread_yolo,       name="YOLOv8",     daemon=True),
        threading.Thread(target=thread_mediapipe,  name="MediaPipe",  daemon=True),
        threading.Thread(target=thread_decision,   name="Decision",   daemon=True),
    ]
    for t in threads:
        t.start()

    fps_counter  = 0
    fps_time     = time.time()
    fps_display  = 0.0
    video_writer = None

    try:
        while not stop_event.is_set():
            try:
                packet = queue_display.get(timeout=0.5)
            except Exception:
                continue
            if packet is None:
                break
            display_frame_id, frame = packet

            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_display = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time    = now

            with display_lock:
                state_snap = {k: (v.copy() if isinstance(v, list) else v)
                              for k, v in display_state.items()}

            annotated = annotate(frame, state_snap, fps_display, display_frame_id)

            if video_writer is not None:
                video_writer.write(annotated)
                vh, vw = annotated.shape[:2]
                cv2.circle(annotated, (vw - 22, 18), 9, (0, 0, 220), -1)
                cv2.putText(annotated, "REC", (vw - 60, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 220), 2)

            cv2.imshow("GoShare Driver Monitor", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                stop_event.set()
            elif key in (ord("r"), ord("R")):
                if video_writer is None:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    vts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    vpath = os.path.join(OUTPUT_DIR, f"monitor_{vts}.mp4")
                    vh, vw = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(vpath, fourcc, 15, (vw, vh))
                    logging.info(f"開始錄影：{vpath}")
                else:
                    video_writer.release()
                    video_writer = None
                    logging.info("停止錄影")
            elif key in (ord("c"), ord("C")):
                recalib_event.set()

    except KeyboardInterrupt:
        logging.info("收到 KeyboardInterrupt，正在關閉...")
        stop_event.set()
    finally:
        if video_writer is not None:
            video_writer.release()
            logging.info("錄影檔案已儲存")

    for _ in range(2):
        put_nowait_safe(queue_display, None)

    for t in threads:
        t.join(timeout=5.0)

    cv2.destroyAllWindows()
    logging.info("程式結束")


if __name__ == "__main__":
    main()
