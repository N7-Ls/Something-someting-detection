"""
純工具函式，無副作用，不依賴 state。
"""
import cv2
import numpy as np
import queue
from config import FACE_3D_POINTS, FACE_2D_IDX


def ear(landmarks, eye_idx: list, w: int, h: int) -> float:
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_idx])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def head_pose(landmarks, w: int, h: int):
    """回傳 (yaw, pitch, roll)，單位：度。"""
    pts_2d = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in FACE_2D_IDX],
        dtype=np.float64,
    )
    focal = w
    cam_matrix = np.array(
        [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64
    )
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(
        FACE_3D_POINTS, pts_2d, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    # cv2.RQDecomp3x3 直接回傳度，不需 × 360
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]

    # 尤拉角分解歧義：(pitch, yaw, roll) 與 (pitch+180, 180-yaw, roll+180)
    # 代表同一個旋轉矩陣，RQDecomp3x3 會在兩者間跳動造成 corr_pitch 假性大幅跳變。
    # 取 |roll|<=90 的分支為正規表示，roll 接近 0 代表頭部沒有明顯側傾，較符合常態。
    if abs(roll) > 90.0:
        pitch = wrap_angle(pitch + 180.0)
        yaw   = wrap_angle(180.0 - yaw)
        roll  = wrap_angle(roll + 180.0)

    return yaw, pitch, roll


def pixel_dist(p1, p2) -> float:
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def wrap_angle(a: float) -> float:
    """正規化到 (-180, +180]，消除 solvePnP ±180° 邊界跳變。"""
    return ((a + 180.0) % 360.0) - 180.0


def put_nowait_safe(q: queue.Queue, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def put_latest_safe(q: queue.Queue, item):
    """佇列滿時丟棄舊項目、放入最新項目，避免消費者處理過時的積壓影格。"""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
