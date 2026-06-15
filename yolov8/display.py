"""
畫面標註：將 display_state 快照渲染到影格上。
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
_font_cache: dict = {}

def _get_chinese_font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    candidates = [
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    font = ImageFont.load_default()
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    _font_cache[size] = font
    return font


def _put_chinese_text(frame, text: str, xy: tuple, font_size: int, color_bgr: tuple, outline_color=(0, 0, 0)):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _get_chinese_font(font_size)
    x, y = xy
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        draw.text((x + dx, y + dy), text, font=font,
                  fill=(outline_color[2], outline_color[1], outline_color[0]))
    r, g, b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
    draw.text((x, y), text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


LEVEL_COLOR = {
    0: (0, 200, 0),
    1: (0, 200, 255),
    2: (0, 120, 255),
    3: (0, 0, 255),
}


def annotate(frame, state: dict, fps: float, display_frame_id: int = 0,
             minimal: bool = False):
    """
    minimal=True：PyQt5 模式，只保留偵測框/關鍵點/外框色，省略文字面板與警報覆蓋
    （PyQt5 右側面板已顯示相同資訊，避免重複）
    """
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

    if minimal:
        return out

    # ── 頂部校準狀態橫幅 ──
    calib_txt = state.get("calib_status", "")
    if calib_txt:
        is_calibrating = calib_txt.startswith("校準中")
        banner_color   = (0, 140, 255) if is_calibrating else (0, 160, 60)
        overlay3 = out.copy()
        cv2.rectangle(overlay3, (0, 0), (w, 28), banner_color, -1)
        cv2.addWeighted(overlay3, 0.75, out, 0.25, 0, out)
        out = _put_chinese_text(out, calib_txt, (8, 4), 20, (255, 255, 255))

    # ── 底部警報訊息 ──
    if state["alert_msg"]:
        bar_h    = 42
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, h - bar_h), (w, h), color, -1)
        cv2.addWeighted(overlay2, 0.65, out, 0.35, 0, out)
        out = _put_chinese_text(out, state["alert_msg"], (12, h - 38), 24, (255, 255, 255))

    # ── Level 2：右上角警示三角 ──
    if level == 2:
        cx, cy, r = w - 72, 72, 58
        pts = np.array([
            [cx,      cy - r + 8],
            [cx - r + 8, cy + r - 8],
            [cx + r - 8, cy + r - 8],
        ], np.int32)
        cv2.fillPoly(out, [pts], (0, 100, 220))
        cv2.polylines(out, [pts], True, (255, 255, 255), 3)
        cv2.putText(out, "!",  (cx - 10, cy + r - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)

    # ── Level 3：全螢幕疲勞警示覆蓋 ──
    if level == 3:
        danger_overlay = out.copy()
        cv2.rectangle(danger_overlay, (0, 0), (w, h), (0, 0, 160), -1)
        cv2.addWeighted(danger_overlay, 0.40, out, 0.60, 0, out)
        ty = h // 2 - 20
        out = _put_chinese_text(out, "疲勞警示", (w // 2 - 80, ty - 10), 48, (255, 255, 255))
        out = _put_chinese_text(out, "建議立即停車休息", (w // 2 - 120, ty + 55), 28, (255, 220, 220))

    return out
