"""
GoShare Driver Monitor — PyQt5 儀表板
取代 yolov8/main.py 的 cv2.imshow，提供完整圖形化監控介面。

執行：
    cd PyQT5
    python dashboard.py
"""
import os
import sys

# Fix Qt platform plugin not found on Windows when running outside conda activate
def _fix_qt_plugin_path():
    import importlib.util
    spec = importlib.util.find_spec("PyQt5")
    if spec and spec.submodule_search_locations:
        qt5_plugins = os.path.join(list(spec.submodule_search_locations)[0], "Qt5", "plugins")
        if os.path.isdir(qt5_plugins):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", qt5_plugins)
_fix_qt_plugin_path()

import queue as _queue
import threading
import time

import cv2
import numpy as np

# ── 將 yolov8 目錄加入路徑 ──────────────────────────────────────────────────
_YOLOV8_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolov8")
sys.path.insert(0, _YOLOV8_DIR)

import state as _state
from state import (
    queue_display, stop_event, recalib_event,
    display_state, display_lock,
)
from thread_capture   import thread_capture
from thread_yolo      import thread_yolo
from thread_mediapipe import thread_mediapipe
from thread_decision  import thread_decision
from display          import annotate

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QFrame,
    QGroupBox, QGridLayout, QPushButton, QSizePolicy,
)
from PyQt5.QtCore  import Qt, QTimer
from PyQt5.QtGui   import QImage, QPixmap, QFont

import logging
logging.basicConfig(level=logging.INFO, format="[%(threadName)s] %(message)s")

# ── 常數 ────────────────────────────────────────────────────────────────────
_PANEL_W    = 290
_REFRESH_MS = 33      # ~30 fps UI 刷新


# ── VideoWidget ──────────────────────────────────────────────────────────────
class VideoWidget(QLabel):
    """顯示 OpenCV BGR frame 的 QLabel，自動縮放並保持長寬比。"""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #000000;")
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._src_pixmap = None

    def set_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg  = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._src_pixmap = QPixmap.fromImage(qimg)
        self._rescale()

    def resizeEvent(self, _e):
        self._rescale()

    def _rescale(self):
        if self._src_pixmap:
            scaled = self._src_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)


# ── AlertIcon ────────────────────────────────────────────────────────────────
class AlertIcon(QWidget):
    """單一違規項目的 emoji + 文字 + 指示燈。"""

    _COLORS = {
        "normal":    "#3a3a3a",
        "condition": "#e67e22",   # 橘：感測條件瞬間成立
        "triggered": "#e74c3c",   # 紅：計時器觸發（真正警報）
    }

    def __init__(self, emoji: str, label: str):
        super().__init__()
        self._cur_state = "normal"

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(6)

        emoji_lbl = QLabel(emoji)
        emoji_lbl.setFixedSize(32, 32)
        emoji_lbl.setAlignment(Qt.AlignCenter)
        emoji_lbl.setFont(QFont("Segoe UI Emoji", 16))

        self._text_lbl = QLabel(label)
        self._text_lbl.setFont(QFont("Segoe UI", 10))
        self._text_lbl.setStyleSheet("color: #aaaaaa;")

        self._dot = QFrame()
        self._dot.setFixedSize(12, 12)
        self._dot.setStyleSheet(
            f"background: {self._COLORS['normal']}; border-radius: 6px;"
        )

        lay.addWidget(emoji_lbl)
        lay.addWidget(self._text_lbl, 1)
        lay.addWidget(self._dot)

    def set_state(self, s: str):
        if s == self._cur_state:
            return
        self._cur_state = s
        color = self._COLORS.get(s, self._COLORS["normal"])
        self._dot.setStyleSheet(f"background: {color}; border-radius: 6px;")
        txt_color = "#ffffff" if s != "normal" else "#aaaaaa"
        self._text_lbl.setStyleSheet(f"color: {txt_color};")


# ── LevelBar ─────────────────────────────────────────────────────────────────
class LevelBar(QWidget):
    """警告等級彩色橫條 + 措施說明文字。"""

    _STYLE = {
        0: ("NORMAL",   "#27ae60", "正常行駛"),
        1: ("LEVEL  1", "#f39c12", "提示音 + 黃燈"),
        2: ("LEVEL  2", "#e67e22", "鎖定 30 km/h"),
        3: ("LEVEL  3", "#c0392b", "緊急喚醒 · 時速歸零"),
    }

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._level_lbl = QLabel("NORMAL")
        self._level_lbl.setAlignment(Qt.AlignCenter)
        self._level_lbl.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self._level_lbl.setFixedHeight(44)

        self._msg_lbl = QLabel("正常行駛")
        self._msg_lbl.setAlignment(Qt.AlignCenter)
        self._msg_lbl.setFont(QFont("Segoe UI", 9))
        self._msg_lbl.setWordWrap(True)

        lay.addWidget(self._level_lbl)
        lay.addWidget(self._msg_lbl)
        self.set_level(0)

    def set_level(self, level: int, msg: str = ""):
        text, color, default_msg = self._STYLE.get(level, self._STYLE[0])
        self._level_lbl.setText(text)
        self._level_lbl.setStyleSheet(
            f"background: {color}; color: white; border-radius: 5px; padding: 2px;"
        )
        # 從 thread_decision 的 msg 擷取可讀部分
        if msg and "[介入]" in msg:
            msg = msg.split(": ", 1)[-1]
        self._msg_lbl.setText(msg if msg else default_msg)
        self._msg_lbl.setStyleSheet(f"color: {color};")


# ── StatusPanel ───────────────────────────────────────────────────────────────
class StatusPanel(QWidget):
    """模型運行數值的格狀顯示面板。"""

    _FIELDS = [
        ("fps",   "FPS"),
        ("face",  "臉部"),
        ("ear",   "EAR"),
        ("yaw",   "Yaw"),
        ("pitch", "Ptch*"),
        ("sync",  "Sync"),
        ("calib", "校準"),
        ("cig",   "Cig模型"),
    ]

    def __init__(self):
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(2)
        grid.setColumnStretch(1, 1)

        self._vals: dict[str, QLabel] = {}
        for i, (key, lbl_txt) in enumerate(self._FIELDS):
            lbl = QLabel(lbl_txt + ":")
            lbl.setFont(QFont("Segoe UI", 8))
            lbl.setStyleSheet("color: #666666;")

            val = QLabel("--")
            val.setFont(QFont("Consolas", 8))
            val.setStyleSheet("color: #cccccc;")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            grid.addWidget(lbl, i, 0)
            grid.addWidget(val, i, 1)
            self._vals[key] = val

    def update(self, key: str, text: str, color: str = "#cccccc"):
        w = self._vals.get(key)
        if w:
            w.setText(text)
            w.setStyleSheet(f"color: {color};")


# ── DashboardWindow ───────────────────────────────────────────────────────────
class DashboardWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GoShare Driver Monitor")
        self.setMinimumSize(880, 580)
        self.setStyleSheet("QMainWindow { background: #1a1a1a; }")

        self._fps_count   = 0
        self._fps_time    = time.time()
        self._fps_val     = 0.0
        self._last_fid    = 0

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(_REFRESH_MS)

    # ── UI 建構 ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        root_w = QWidget()
        root_w.setStyleSheet("background: #1a1a1a;")
        self.setCentralWidget(root_w)

        root = QHBoxLayout(root_w)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── 左：鏡頭回饋 ─────────────────────────────────────────────────────
        self._video = VideoWidget()
        root.addWidget(self._video, 1)

        # ── 右：控制面板 ─────────────────────────────────────────────────────
        panel = QWidget()
        panel.setFixedWidth(_PANEL_W)
        panel.setStyleSheet("background: #1a1a1a;")
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(6)

        # 警告等級
        lvl_box = self._section("警告等級")
        self._level_bar = LevelBar()
        lvl_box.layout().addWidget(self._level_bar)
        pv.addWidget(lvl_box)

        # 違規指示燈
        icons_box = self._section("違規指示")
        self._icons = {
            "phone":    AlertIcon("📱", "手機操作"),
            "smoke":    AlertIcon("🚬", "吸菸行為"),
            "fatigue":  AlertIcon("😴", "疲勞駕駛"),
            "distract": AlertIcon("👁",  "視線分心"),
        }
        for ico in self._icons.values():
            icons_box.layout().addWidget(ico)
        pv.addWidget(icons_box)

        # 模型狀態
        stat_box = self._section("模型狀態")
        self._status = StatusPanel()
        stat_box.layout().addWidget(self._status)
        pv.addWidget(stat_box)

        # 按鈕
        btn_row = QHBoxLayout()
        btn_calib = QPushButton("重新校準 (C)")
        btn_calib.clicked.connect(lambda: recalib_event.set())
        btn_calib.setStyleSheet(self._btn_css("#2980b9"))

        btn_quit = QPushButton("結束 (Q)")
        btn_quit.clicked.connect(self._quit)
        btn_quit.setStyleSheet(self._btn_css("#c0392b"))

        btn_row.addWidget(btn_calib)
        btn_row.addWidget(btn_quit)
        pv.addLayout(btn_row)
        pv.addStretch()

        root.addWidget(panel)

    def _section(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setStyleSheet(
            "QGroupBox { color: #888888; border: 1px solid #383838; border-radius: 4px;"
            " margin-top: 10px; padding-top: 6px; font-size: 9px; font-family: 'Segoe UI'; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 10, 6, 6)
        lay.setSpacing(2)
        return box

    @staticmethod
    def _btn_css(color: str) -> str:
        return (
            f"QPushButton {{ background: {color}; color: white; border: none;"
            " border-radius: 4px; padding: 5px 10px; font-size: 9px; font-family: 'Segoe UI'; }"
            f"QPushButton:hover {{ background: {color}bb; }}"
        )

    # ── 定時刷新 ─────────────────────────────────────────────────────────────
    def _refresh(self):
        # 拉最新影格（排空過期幀）
        frame_raw    = None
        display_fid  = self._last_fid
        try:
            while True:
                pkt = queue_display.get_nowait()
                if pkt is None:
                    self._quit()
                    return
                display_fid, frame_raw = pkt
        except _queue.Empty:
            pass
        self._last_fid = display_fid

        # FPS 計算
        self._fps_count += 1
        now = time.time()
        if now - self._fps_time >= 1.0:
            self._fps_val    = self._fps_count / (now - self._fps_time)
            self._fps_count  = 0
            self._fps_time   = now

        # 快照 display_state（整份複製，避免長時間持鎖）
        with display_lock:
            snap = {
                k: (list(v) if isinstance(v, list) else
                    dict(v) if isinstance(v, dict) else v)
                for k, v in display_state.items()
            }

        # 更新影格
        if frame_raw is not None:
            annotated = annotate(frame_raw, snap, self._fps_val, display_fid, minimal=True)
            self._video.set_frame(annotated)

        # 更新各面板
        self._refresh_level(snap)
        self._refresh_icons(snap)
        self._refresh_status(snap)

    def _refresh_level(self, snap: dict):
        self._level_bar.set_level(
            snap.get("alert_level", 0),
            snap.get("alert_msg",   ""),
        )

    def _refresh_icons(self, snap: dict):
        flags    = snap.get("alert_flags", {})
        ear_v    = snap.get("ear_val")
        yaw_v    = snap.get("yaw")
        pc_v     = snap.get("pitch_corr")
        wmd      = snap.get("wrist_mouth_dist")
        fw       = snap.get("face_width")
        wmd_thr  = (fw * 0.55) if fw else None

        # 原始條件（瞬間，未計時）
        raw = {
            "phone":    len(snap.get("phone_boxes", [])) > 0
                        or (pc_v is not None and pc_v > 32),
            "smoke":    (wmd is not None and wmd_thr is not None and wmd < wmd_thr)
                        or len(snap.get("cig_boxes", [])) > 0,
            "fatigue":  ear_v is not None and ear_v < 0.20,
            "distract": (yaw_v is not None and abs(yaw_v) > 45)
                        or (pc_v is not None and pc_v < -45),
        }

        for key, icon in self._icons.items():
            if flags.get(key, False):
                icon.set_state("triggered")
            elif raw.get(key, False):
                icon.set_state("condition")
            else:
                icon.set_state("normal")

    def _refresh_status(self, snap: dict):
        s        = self._status
        face_ok  = snap.get("face_detected", False)
        mp_cnt   = snap.get("mp_frames", 0)
        ear_v    = snap.get("ear_val")
        yaw_v    = snap.get("yaw")
        pc_v     = snap.get("pitch_corr")
        yolo_fid = snap.get("yolo_frame_id", 0)
        face_fid = snap.get("face_frame_id", 0)
        calib    = snap.get("calib_status", "")

        s.update("fps",  f"{self._fps_val:.1f} fps")

        s.update("face",
                 f"OK  #{mp_cnt}" if face_ok else "未偵測",
                 "#2ecc71" if face_ok else "#e74c3c")

        s.update("ear",
                 f"{ear_v:.3f}" if ear_v is not None else "--",
                 "#e74c3c" if (ear_v is not None and ear_v < 0.20) else "#cccccc")

        s.update("yaw",
                 f"{yaw_v:+.1f}°" if yaw_v is not None else "--",
                 "#f39c12" if (yaw_v is not None and abs(yaw_v) > 45) else "#cccccc")

        s.update("pitch",
                 f"{pc_v:+.1f}°" if pc_v is not None else "--",
                 "#f39c12" if (pc_v is not None and (pc_v > 32 or pc_v < -45)) else "#cccccc")

        lag = abs(yolo_fid - face_fid)
        s.update("sync",
                 f"Y{yolo_fid} F{face_fid} Δ{lag}",
                 "#f39c12" if lag > 8 else "#cccccc")

        calib_short = calib.split("（")[0][:28] if "（" in calib else calib[:28]
        is_calib = calib.startswith("校準中") or calib.startswith("重新校準")
        s.update("calib", calib_short,
                 "#f39c12" if is_calib else "#2ecc71")

        s.update("cig",
                 "已載入" if _state.cig_model_available else "fallback",
                 "#2ecc71" if _state.cig_model_available else "#f39c12")

    # ── 退出 ─────────────────────────────────────────────────────────────────
    def _quit(self):
        stop_event.set()
        self._timer.stop()
        QApplication.quit()

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key_Q, Qt.Key_Escape):
            self._quit()
        elif k == Qt.Key_C:
            recalib_event.set()


# ── 入口點 ───────────────────────────────────────────────────────────────────
def main():
    threads = [
        threading.Thread(target=thread_capture,   name="Capture",   daemon=True),
        threading.Thread(target=thread_yolo,       name="YOLOv8",    daemon=True),
        threading.Thread(target=thread_mediapipe,  name="MediaPipe", daemon=True),
        threading.Thread(target=thread_decision,   name="Decision",  daemon=True),
    ]
    for t in threads:
        t.start()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 深色 Fusion palette
    from PyQt5.QtGui import QPalette, QColor as _QColor
    pal = QPalette()
    pal.setColor(QPalette.Window,          _QColor(26, 26, 26))
    pal.setColor(QPalette.WindowText,      _QColor(220, 220, 220))
    pal.setColor(QPalette.Base,            _QColor(35, 35, 35))
    pal.setColor(QPalette.AlternateBase,   _QColor(45, 45, 45))
    pal.setColor(QPalette.Text,            _QColor(220, 220, 220))
    pal.setColor(QPalette.Button,          _QColor(50, 50, 50))
    pal.setColor(QPalette.ButtonText,      _QColor(220, 220, 220))
    pal.setColor(QPalette.Highlight,       _QColor(41, 128, 185))
    pal.setColor(QPalette.HighlightedText, _QColor(255, 255, 255))
    app.setPalette(pal)

    win = DashboardWindow()
    win.show()

    ret = app.exec_()

    stop_event.set()
    for t in threads:
        t.join(timeout=3.0)

    sys.exit(ret)


if __name__ == "__main__":
    main()
