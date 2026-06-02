import cv2
import base64
import os
import tempfile
import threading
import time
import requests
import numpy as np
from preprocess_head import crop_head

CHECK_INTERVAL = 8.0   # 每幾秒自動偵測一次

API_URL = "https://uncommutatively-unpersuadable-an.ngrok-free.dev/api/chat"
API_KEY = "upiceollama"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
}

PROMPT = (
    "Step 1: Is the person wearing a helmet? If no, output 'Result: No helmet'.\n\n"
    "Step 2: Is the chin strap buckle clipped shut under the chin?\n"
    "- FASTENED: the buckle is closed and the strap connects both sides under the chin. "
    "The strap does NOT need to be tight — only the buckle must be clipped.\n"
    "- UNFASTENED: the buckle is open, or the straps hang separately on the sides of the face "
    "without connecting under the chin.\n\n"
    "Describe the buckle in one sentence, then output exactly one of:\n"
    "- 'Result: No helmet'\n"
    "- 'Result: Helmet on, strap fastened'\n"
    "- 'Result: Helmet on, strap unfastened'"
)

# 顏色對應 (BGR)
LABEL_COLOR = {
    "Fastened":     (0, 200, 0),
    "Not_fastened": (0, 120, 255),
    "No_helmet":    (0, 0, 220),
    "分析中...":    (180, 180, 0),
    "等待拍照":     (160, 160, 160),
    "錯誤":         (0, 0, 180),
}


def frame_to_base64(bgr_img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


def parse_label(answer: str) -> str:
    lower = answer.lower()
    if "no helmet" in lower:
        return "No_helmet"
    elif "strap fastened" in lower:
        return "Fastened"
    else:
        return "Not_fastened"


def call_api(cropped_bgr: np.ndarray) -> str:
    img_b64 = frame_to_base64(cropped_bgr)
    payload = {
        "model": "gemma3:27b",
        "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": 0.0},
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def crop_head_from_frame(bgr_frame: np.ndarray):
    """把 numpy frame 存成暫存檔，跑 crop_head，回傳裁切後的 BGR 圖。"""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, bgr_frame)

    try:
        cropped_path = crop_head(tmp_path)
        cropped_img = cv2.imread(cropped_path)
        if cropped_path != tmp_path and os.path.exists(cropped_path):
            os.remove(cropped_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return cropped_img


def draw_result(frame: np.ndarray, label: str, countdown: float = 0.0):
    """在畫面左上角畫結果框與倒數。"""
    color = LABEL_COLOR.get(label, (200, 200, 200))
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    label_map = {
        "Fastened":     "Helmet ON, Strap FASTENED",
        "Not_fastened": "Helmet ON, Strap UNFASTENED",
        "No_helmet":    "NO HELMET",
        "分析中...":    "Analyzing...",
        "等待中":       "Starting...",
        "錯誤":         "Error - check console",
    }
    display_text = label_map.get(label, label)
    cv2.putText(frame, display_text, (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    if label == "分析中...":
        sub = "Analyzing..."
    else:
        sub = f"Next check in {countdown:.0f}s  |  Q: quit"
    cv2.putText(frame, sub, (15, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 無法開啟鏡頭")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_label = "等待中"
    analyzing = False
    last_check = time.perf_counter() - CHECK_INTERVAL  # 啟動後立即觸發第一次

    def analyze(frame_snapshot: np.ndarray):
        nonlocal current_label, analyzing
        analyzing = True
        current_label = "分析中..."
        try:
            cropped = crop_head_from_frame(frame_snapshot)
            if cropped is None:
                print("[WARN] 未偵測到人物，直接送原圖")
                cropped = frame_snapshot
            answer = call_api(cropped)
            print(f"\n[API 回應]\n{answer}\n")
            current_label = parse_label(answer)
            print(f"[結果] {current_label}")
        except Exception as e:
            print(f"[ERROR] {e}")
            current_label = "錯誤"
        finally:
            analyzing = False

    print(f"鏡頭已開啟，每 {CHECK_INTERVAL:.0f} 秒自動偵測一次。按 Q 離開。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 讀取鏡頭失敗")
            break
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        now = time.perf_counter()
        if not analyzing and (now - last_check) >= CHECK_INTERVAL:
            last_check = now
            snapshot = frame.copy()
            threading.Thread(target=analyze, args=(snapshot,), daemon=True).start()

        countdown = max(0.0, CHECK_INTERVAL - (now - last_check))
        display = frame.copy()
        draw_result(display, current_label, countdown)
        cv2.imshow("Helmet Detection", display)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
