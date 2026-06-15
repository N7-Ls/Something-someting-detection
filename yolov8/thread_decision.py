"""
執行緒 4：決策中心。
融合 YOLO + MediaPipe 結果，輸出 Level 0~3 介入指令，並寫入 CSV。
"""
import csv
import datetime
import logging
import os
import queue
import time
from collections import deque
import numpy as np
import state
from config import (
    OUTPUT_DIR, RECORD_CSV, CALIB_SECONDS,
    YAW_PITCH_LIMIT, PITCH_PHONE_LIMIT,
    EAR_THRESHOLD_RATIO, EAR_THRESHOLD_MIN, EAR_THRESHOLD_MAX,
    FUSE_TIME_WINDOW, WRIST_MOUTH_RATIO, CIG_MOUTH_RATIO,
    DURATION_DISTRACT, DURATION_SMOKE, DURATION_SMOKE_NOCIG,
    DURATION_PHONE, DURATION_FATIGUE,
    HOLD_DISTRACT, HOLD_SMOKE, HOLD_PHONE, HOLD_FATIGUE,
    PERCLOS_WINDOW_SEC, PERCLOS_THRESHOLD,
)
from state import (
    queue_decision, stop_event, recalib_event,
    display_state, display_lock,
    get_cam_offset, set_cam_offset,
    get_ear_threshold, set_ear_threshold,
)
from utils import wrap_angle, pixel_dist


def thread_decision():
    cache_yolo  = {}
    cache_face  = {}
    timers      = {"distract": None, "smoke": None, "phone": None, "fatigue": None}
    hold_timers = {"distract": None, "smoke": None, "phone": None, "fatigue": None}
    ear_history: deque = deque()   # (perf_counter timestamp, ear_value)
    HOLD_MAP    = {
        "distract": HOLD_DISTRACT,
        "smoke":    HOLD_SMOKE,
        "phone":    HOLD_PHONE,
        "fatigue":  HOLD_FATIGUE,
    }
    prev_level = 0

    def check_duration(key, condition, required_sec):
        now = time.perf_counter()
        if condition:
            hold_timers[key] = None
            if timers[key] is None:
                timers[key] = now
            return (now - timers[key]) >= required_sec
        else:
            if timers[key] is None:
                return False
            if hold_timers[key] is None:
                hold_timers[key] = now
            if (now - hold_timers[key]) >= HOLD_MAP[key]:
                timers[key] = None
                hold_timers[key] = None
                return False
            return (now - timers[key]) >= required_sec

    # ── 自動校準 ──
    calib_pitches = []
    calib_ears    = []
    calib_start   = time.perf_counter()
    calibrated    = False

    def _finish_calibration():
        if len(calib_pitches) >= 10:
            # 角度圓形平均（circular mean）：正確處理 ±180° 邊界
            # 直接 median 在 +177 / -167 同時出現時會跑到錯的一側
            rads     = np.radians(calib_pitches)
            new_off  = float(np.degrees(np.arctan2(np.mean(np.sin(rads)),
                                                    np.mean(np.cos(rads)))))
            set_cam_offset(new_off)
            status = f"校準完成：offset={new_off:+.1f}°"
            logging.info(f"校準完成：{new_off:.1f}°（樣本數 {len(calib_pitches)}）")
        else:
            new_off = get_cam_offset()
            status  = f"校準樣本不足，沿用預設 {new_off:+.1f}°"
            logging.warning("校準樣本不足，使用預設 PITCH_CAM_OFFSET")

        if len(calib_ears) >= 10:
            baseline_ear = float(np.median(calib_ears))
            new_ear_thr  = baseline_ear * EAR_THRESHOLD_RATIO
            new_ear_thr  = min(max(new_ear_thr, EAR_THRESHOLD_MIN), EAR_THRESHOLD_MAX)
            set_ear_threshold(new_ear_thr)
            status += f"，EAR閾值={new_ear_thr:.3f}（基準{baseline_ear:.3f}）"
            logging.info(f"EAR 閾值校準完成：{new_ear_thr:.3f}（基準 {baseline_ear:.3f}，樣本數 {len(calib_ears)}）")
        else:
            status += f"，EAR閾值沿用預設 {get_ear_threshold():.3f}"
            logging.warning("EAR 校準樣本不足，沿用預設 EAR_THRESHOLD")

        status += "（C 鍵重新校準）"
        with display_lock:
            display_state["calib_status"] = status

    # ── CSV 設定 ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"monitor_{ts_str}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp",
        "raw_pitch", "corr_pitch", "yaw", "roll", "ear",
        "cam_offset",
        "phone_bbox", "phone_gaze", "phone_wrist",
        "smoke", "fatigue", "distract",
        "alert_level", "alert_msg",
    ])
    logging.info(f"感測數值記錄至：{csv_path}")

    try:
        while not stop_event.is_set():
            # ── 重新校準請求 ──
            if recalib_event.is_set():
                recalib_event.clear()
                calib_pitches.clear()
                calib_ears.clear()
                calib_start = time.perf_counter()
                calibrated  = False
                with display_lock:
                    display_state["calib_status"] = f"重新校準中… {CALIB_SECONDS:.0f}s"
                logging.info("重新校準開始")

            try:
                result = queue_decision.get(timeout=1.0)
            except queue.Empty:
                if not calibrated and (time.perf_counter() - calib_start) >= CALIB_SECONDS:
                    _finish_calibration()
                    calibrated = True
                continue
            if result is None:
                break

            fid = result["frame_id"]
            if result["source"] == "yolo":
                cache_yolo[fid] = result
            else:
                cache_face[fid] = result

            for c in (cache_yolo, cache_face):
                if len(c) > 30:
                    del c[min(c.keys())]

            yolo = cache_yolo.get(fid)
            if yolo is None and cache_yolo:
                latest = cache_yolo[max(cache_yolo.keys())]
                # perf_counter 與 thread_capture 的 ts 同源，不受 NTP 影響
                if time.perf_counter() - latest["timestamp"] < 0.5:
                    yolo = latest
            face = (cache_face.get(fid) or
                    (cache_face[max(cache_face.keys())] if cache_face else None))

            # ── 校準期間收集 raw pitch / EAR ──
            raw_pitch = face["pitch"] if (face and face["pitch"] is not None) else None
            if not calibrated:
                remaining = CALIB_SECONDS - (time.perf_counter() - calib_start)
                if raw_pitch is not None:
                    calib_pitches.append(raw_pitch)
                if face is not None and face["ear_val"] is not None:
                    calib_ears.append(face["ear_val"])
                if remaining <= 0:
                    _finish_calibration()
                    calibrated = True
                else:
                    with display_lock:
                        display_state["calib_status"] = f"校準中… {remaining:.0f}s（保持正常姿勢）"

            # ── corr_pitch（關鍵公式，勿更動符號與 wrap）──
            # 正值=低頭/手機，負值=抬頭/分心
            cam_off    = get_cam_offset()
            corr_pitch = (-wrap_angle(raw_pitch - cam_off)
                          if raw_pitch is not None else None)

            with display_lock:
                display_state["pitch_corr"] = corr_pitch

            # ── 條件判斷 ──
            distract_cond = (
                face is not None and face["yaw"] is not None and corr_pitch is not None
                and (abs(face["yaw"]) > YAW_PITCH_LIMIT or corr_pitch < -YAW_PITCH_LIMIT)
            )
            phone_by_bbox  = yolo is not None and yolo["phone_detected"]
            phone_by_gaze  = corr_pitch is not None and corr_pitch > PITCH_PHONE_LIMIT
            phone_by_wrist = False
            if yolo and face and yolo["wrist_xy"] and face["mouth_xy"]:
                face_y = face["mouth_xy"][1]
                fw     = face.get("face_width") or 80
                for wx, wy in yolo["wrist_xy"]:
                    # 手腕在嘴部上方，或低於嘴部不超過 2 倍臉寬（手舉至臉旁的正常持機範圍）
                    if wy < face_y + fw * 2.0:
                        phone_by_wrist = True
                        break
            phone_cond = phone_by_bbox or phone_by_gaze or phone_by_wrist

            # ── PERCLOS 疲勞偵測（滾動窗口）──
            _now_pc = time.perf_counter()
            if face is not None and face["ear_val"] is not None:
                ear_history.append((_now_pc, face["ear_val"]))
            while ear_history and (_now_pc - ear_history[0][0]) > PERCLOS_WINDOW_SEC:
                ear_history.popleft()
            if len(ear_history) >= 10:
                _ear_thr = get_ear_threshold()
                _closed  = sum(1 for _, e in ear_history if e < _ear_thr)
                _perclos = _closed / len(ear_history)
            else:
                _perclos = 0.0
            fatigue_cond = _perclos >= PERCLOS_THRESHOLD
            with display_lock:
                display_state["perclos"] = _perclos

            smoke_cond          = False
            wrist_mouth_dist_min = None
            if yolo and face:
                if abs(yolo["timestamp"] - face["timestamp"]) <= FUSE_TIME_WINDOW:
                    if yolo["wrist_xy"] and face["mouth_xy"] and face.get("face_width"):
                        threshold_px = face["face_width"] * WRIST_MOUTH_RATIO
                        # 垂直距離門檻：手腕須在嘴部高度 ±0.35×臉寬範圍內
                        # 防止撥髮、擦額頭等高位手部動作誤觸
                        vert_thr = face["face_width"] * 0.35
                        mouth_y  = face["mouth_xy"][1]
                        all_dists = [pixel_dist(w, face["mouth_xy"]) for w in yolo["wrist_xy"]]
                        wrist_mouth_dist_min = min(all_dists)
                        wrist_close = any(
                            pixel_dist(w, face["mouth_xy"]) < threshold_px
                            and abs(w[1] - mouth_y) < vert_thr
                            for w in yolo["wrist_xy"]
                        )
                        if state.cig_model_available:
                            # 額外空間過濾：cig BBox 中心必須在嘴部附近
                            # 排除遠離臉部的誤報（手機/手掌被誤判）
                            cig_near_mouth = False
                            cig_spatial_thr = face["face_width"] * CIG_MOUTH_RATIO
                            for cb in display_state.get("cig_boxes", []):
                                cx = (cb[0] + cb[2]) / 2
                                cy = (cb[1] + cb[3]) / 2
                                if pixel_dist((cx, cy), face["mouth_xy"]) < cig_spatial_thr:
                                    cig_near_mouth = True
                                    break
                            smoke_cond = yolo["cigarette_detected"] and wrist_close and cig_near_mouth
                        else:
                            smoke_cond = wrist_close

            with display_lock:
                display_state["wrist_mouth_dist"] = wrist_mouth_dist_min

            smoke_duration = DURATION_SMOKE if state.cig_model_available else DURATION_SMOKE_NOCIG
            trig_fatigue  = check_duration("fatigue",  fatigue_cond,  DURATION_FATIGUE)
            trig_smoke    = check_duration("smoke",    smoke_cond,    smoke_duration)
            trig_phone    = check_duration("phone",    phone_cond,    DURATION_PHONE)
            trig_distract = check_duration("distract", distract_cond, DURATION_DISTRACT)

            if trig_fatigue:
                level, msg = 3, "[警示] Level 3: 偵測到疲勞駕駛，請立即靠邊停車休息"
            elif trig_smoke:
                level, msg = 2, "[警示] Level 2: 偵測到抽菸行為，請專注駕駛"
            elif trig_phone:
                level = 2
                if phone_by_bbox:
                    msg = "[警示] Level 2: 偵測到使用手機，請放下手機專注駕駛"
                elif phone_by_wrist:
                    msg = "[警示] Level 2: 疑似使用手機，請放下手機專注駕駛"
                else:
                    msg = "[警示] Level 2: 疑似低頭分心，請注意前方路況"
            elif trig_distract:
                level, msg = 1, "[警示] Level 1: 視線偏離，請注意前方路況"
            else:
                level, msg = 0, ""

            if msg and level != prev_level:
                print(msg)
            prev_level = level

            with display_lock:
                display_state["alert_level"] = level
                display_state["alert_msg"]   = msg
                display_state["alert_flags"] = {
                    "phone":    trig_phone,
                    "smoke":    trig_smoke,
                    "fatigue":  trig_fatigue,
                    "distract": trig_distract,
                }

            if RECORD_CSV:
                csv_writer.writerow([
                    f"{time.time():.3f}",
                    f"{raw_pitch:.2f}"    if raw_pitch  is not None else "",
                    f"{corr_pitch:.2f}"   if corr_pitch is not None else "",
                    f"{face['yaw']:.2f}"  if face and face["yaw"]     is not None else "",
                    f"{face['roll']:.2f}" if face and face["roll"]    is not None else "",
                    f"{face['ear_val']:.4f}" if face and face["ear_val"] is not None else "",
                    f"{cam_off:.1f}",
                    int(phone_by_bbox), int(phone_by_gaze), int(phone_by_wrist),
                    int(smoke_cond), int(fatigue_cond), int(distract_cond),
                    level, msg,
                ])

    finally:
        csv_file.close()
        logging.info(f"CSV 已儲存：{csv_path}")
        logging.info("決策中心執行緒結束")
