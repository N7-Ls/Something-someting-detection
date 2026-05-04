"""
畫面標註：將 display_state 快照渲染到影格上。
"""
import cv2
from config import (
    EAR_THRESHOLD, YAW_PITCH_LIMIT, PITCH_PHONE_LIMIT, WRIST_MOUTH_RATIO,
)

LEVEL_COLOR = {
    0: (0, 200, 0),
    1: (0, 200, 255),
    2: (0, 120, 255),
    3: (0, 0, 255),
}


def annotate(frame, state: dict, fps: float, display_frame_id: int = 0):
    h, w = frame.shape[:2]
    out   = frame.copy()
    level = state["alert_level"]
    color = LEVEL_COLOR[level]

    # ── 外框顏色代表警戒等級 ──
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 4)

    # ── 手機 BBox ──
    for (x1, y1, x2, y2, conf) in state["phone_boxes"]:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(out, f"Phone {conf:.2f}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # ── 香菸 BBox ──
    for (x1, y1, x2, y2, conf) in state["cig_boxes"]:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 120, 255), 2)
        cv2.putText(out, f"Cig {conf:.2f}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)

    # ── 手腕關鍵點 ──
    for (wx, wy) in state["wrist_xy"]:
        cv2.circle(out, (int(wx), int(wy)), 8, (255, 80, 0), -1)
        cv2.putText(out, "Wrist", (int(wx) + 10, int(wy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)

    # ── 嘴部中心 ──
    if state["mouth_xy"]:
        mx, my = state["mouth_xy"]
        cv2.circle(out, (int(mx), int(my)), 6, (200, 0, 200), -1)

    # ── 全臉 Face Mesh ──
    mesh = state.get("mesh_landmarks")
    if mesh:
        for (px, py) in mesh:
            cv2.circle(out, (int(px), int(py)), 1, (0, 220, 200), -1)

    # ── 眼角/嘴角關鍵點 ──
    for (px, py) in state["face_pts"]:
        cv2.circle(out, (int(px), int(py)), 3, (255, 255, 255), -1)

    # ── 左側資訊面板 ──
    ear_v    = state["ear_val"]
    yaw_v    = state["yaw"]
    roll_v   = state["roll"]
    pitchc_v = state.get("pitch_corr")

    def ear_color(v):
        if v is None: return (180, 180, 180)
        return (0, 0, 255) if v < EAR_THRESHOLD else (0, 255, 80)

    def yaw_color(v):
        if v is None: return (180, 180, 180)
        return (0, 200, 255) if abs(v) > YAW_PITCH_LIMIT else (0, 255, 80)

    def pitchc_color(v):
        if v is None: return (180, 180, 180)
        return (0, 200, 255) if (v > PITCH_PHONE_LIMIT or v < -YAW_PITCH_LIMIT) else (0, 255, 80)

    face_ok  = state.get("face_detected", False)
    mp_cnt   = state.get("mp_frames", 0)
    face_txt = f"Face : OK  (#{mp_cnt})" if face_ok else f"Face : ND  (#{mp_cnt})"
    face_clr = (0, 255, 80) if face_ok else (0, 80, 255)

    fw   = state.get("face_width")
    wmd  = state.get("wrist_mouth_dist")
    thr  = (fw * WRIST_MOUTH_RATIO) if fw else None

    def wmd_color(d, t):
        if d is None or t is None: return (180, 180, 180)
        return (0, 0, 255) if d < t else (0, 255, 80)

    wmd_txt    = (f"W-M  : {wmd:.0f}/{thr:.0f}px"
                  if wmd is not None and thr is not None else "W-M  : --")
    pitchc_txt = (f"Ptch*: {pitchc_v:+.1f}" if pitchc_v is not None else "Ptch*: --")

    panel = [
        (face_txt,                                                              face_clr),
        (f"EAR  : {ear_v:.3f}"  if ear_v   is not None else "EAR  : --",     ear_color(ear_v)),
        (f"Yaw  : {yaw_v:+.1f}" if yaw_v   is not None else "Yaw  : --",     yaw_color(yaw_v)),
        (pitchc_txt,                                                            pitchc_color(pitchc_v)),
        (f"Roll : {roll_v:+.1f}"if roll_v  is not None else "Roll : --",     (180, 180, 180)),
        (wmd_txt,                                                               wmd_color(wmd, thr)),
        (f"FPS  : {fps:.1f}",                                                  (200, 200, 200)),
    ]

    # YOLO 影格落差：若 YOLO 比當前顯示幀落後超過 8 幀，以黃色警示
    yolo_fid = state.get("yolo_frame_id", 0)
    face_fid = state.get("face_frame_id", 0)
    yolo_lag = display_frame_id - yolo_fid
    face_lag = display_frame_id - face_fid
    lag_txt   = f"Sync : Y-{yolo_lag} F-{face_lag}"
    lag_color = (0, 200, 255) if (yolo_lag > 8 or face_lag > 8) else (100, 100, 100)
    panel.append((lag_txt, lag_color))

    LINE_H  = 28
    PANEL_W = 235
    PANEL_H = len(panel) * LINE_H + 10
    MARGIN  = 8

    overlay = out.copy()
    cv2.rectangle(overlay, (MARGIN, MARGIN), (MARGIN + PANEL_W, MARGIN + PANEL_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    for i, (txt, txt_color) in enumerate(panel):
        y_pos = MARGIN + LINE_H * (i + 1)
        cv2.putText(out, txt, (MARGIN + 6, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
        cv2.putText(out, txt, (MARGIN + 6, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.58, txt_color, 1)

    # ── 頂部校準狀態橫幅 ──
    calib_txt = state.get("calib_status", "")
    if calib_txt:
        is_calibrating = calib_txt.startswith("校準中")
        banner_color   = (0, 140, 255) if is_calibrating else (0, 160, 60)
        overlay3 = out.copy()
        cv2.rectangle(overlay3, (0, 0), (w, 28), banner_color, -1)
        cv2.addWeighted(overlay3, 0.75, out, 0.25, 0, out)
        cv2.putText(out, calib_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 3)
        cv2.putText(out, calib_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)

    # ── 底部警報訊息 ──
    if state["alert_msg"]:
        bar_h    = 42
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, h - bar_h), (w, h), color, -1)
        cv2.addWeighted(overlay2, 0.65, out, 0.35, 0, out)
        cv2.putText(out, state["alert_msg"], (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4)
        cv2.putText(out, state["alert_msg"], (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

    # ── Level 2：右上角速限徽章（30 km/h）──
    if level == 2:
        cx, cy, r = w - 72, 72, 58
        cv2.circle(out, (cx, cy), r, (0, 0, 200), -1)
        cv2.circle(out, (cx, cy), r, (255, 255, 255), 3)
        cv2.putText(out, "30",    (cx - 26, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(out, "km/h", (cx - 28, cy + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # ── Level 3：全螢幕 DANGER 覆蓋 + 時速歸零 ──
    if level == 3:
        danger_overlay = out.copy()
        cv2.rectangle(danger_overlay, (0, 0), (w, h), (0, 0, 160), -1)
        cv2.addWeighted(danger_overlay, 0.40, out, 0.60, 0, out)
        txt_d = "DANGER"
        (tw, _), _ = cv2.getTextSize(txt_d, cv2.FONT_HERSHEY_SIMPLEX, 2.8, 7)
        tx, ty = (w - tw) // 2, h // 2 - 10
        cv2.putText(out, txt_d, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 0, 0), 10)
        cv2.putText(out, txt_d, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 0, 255), 7)
        txt_s = "0  km/h"
        (sw, _), _ = cv2.getTextSize(txt_s, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
        cv2.putText(out, txt_s, ((w - sw) // 2, ty + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
        cv2.putText(out, txt_s, ((w - sw) // 2, ty + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

    return out
