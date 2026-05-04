"""
執行緒 3：MediaPipe FaceLandmarker（Tasks API 0.10+）。
計算 EAR、頭部姿態（yaw/pitch/roll）、嘴部中心、臉寬。
"""
import logging
import queue
import cv2
import mediapipe as mp
from config import (
    FACE_LANDMARKER_PATH,
    LEFT_EYE_IDX, RIGHT_EYE_IDX,
    MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX,
)
from state import (
    queue_face, queue_decision, stop_event,
    display_state, display_lock,
)
from utils import ear, head_pose, put_nowait_safe


def thread_mediapipe():
    BaseOptions        = mp.tasks.BaseOptions
    FaceLandmarker     = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode  = mp.tasks.vision.RunningMode

    try:
        options = FaceLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        landmarker = FaceLandmarker.create_from_options(options)
    except Exception as e:
        logging.error(f"MediaPipe 初始化失敗：{e}")
        return

    DRAW_IDX = LEFT_EYE_IDX + RIGHT_EYE_IDX + [1, 61, 291, 199]

    while not stop_event.is_set():
        try:
            packet = queue_face.get(timeout=1.0)
        except queue.Empty:
            continue
        if packet is None:
            break

        frame_id, ts, frame = packet
        result = {
            "source": "face", "frame_id": frame_id, "timestamp": ts,
            "ear_val": None, "yaw": None, "pitch": None, "roll": None,
            "mouth_xy": None, "face_width": None,
        }

        try:
            h, w = frame.shape[:2]

            with display_lock:
                display_state["mp_frames"] += 1

            rgb       = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_result = landmarker.detect(mp_image)

            if mp_result.face_landmarks:
                lm = mp_result.face_landmarks[0]

                ear_l = ear(lm, LEFT_EYE_IDX,  w, h)
                ear_r = ear(lm, RIGHT_EYE_IDX, w, h)
                result["ear_val"] = (ear_l + ear_r) / 2.0

                yaw, pitch, roll = head_pose(lm, w, h)
                result["yaw"], result["pitch"], result["roll"] = yaw, pitch, roll

                mx = (lm[MOUTH_TOP_IDX].x + lm[MOUTH_BOTTOM_IDX].x) / 2 * w
                my = (lm[MOUTH_TOP_IDX].y + lm[MOUTH_BOTTOM_IDX].y) / 2 * h
                result["mouth_xy"] = (float(mx), float(my))

                face_w = abs(lm[263].x - lm[33].x) * w * 3.0
                result["face_width"] = face_w

                face_pts = [(lm[i].x * w, lm[i].y * h) for i in DRAW_IDX]
                all_pts  = [(lm[i].x * w, lm[i].y * h) for i in range(len(lm))]

                with display_lock:
                    display_state["face_detected"]  = True
                    display_state["ear_val"]        = result["ear_val"]
                    display_state["yaw"]            = yaw
                    display_state["pitch"]          = pitch
                    display_state["roll"]           = roll
                    display_state["mouth_xy"]       = result["mouth_xy"]
                    display_state["face_pts"]       = face_pts
                    display_state["mesh_landmarks"] = all_pts
                    display_state["face_width"]     = face_w
                    display_state["face_frame_id"]  = frame_id
            else:
                with display_lock:
                    display_state["face_detected"]  = False
                    display_state["ear_val"]        = None
                    display_state["yaw"]            = None
                    display_state["pitch"]          = None
                    display_state["roll"]           = None
                    display_state["mouth_xy"]       = None
                    display_state["face_pts"]       = []
                    display_state["mesh_landmarks"] = None

        except Exception as e:
            logging.error(f"MediaPipe 推論例外：{e}")

        put_nowait_safe(queue_decision, result)

    landmarker.close()
    logging.info("MediaPipe 執行緒結束")
