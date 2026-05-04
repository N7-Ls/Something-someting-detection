"""
GoShare Active Safety System - High-Fidelity UI Mockup
基於多工視覺融合架構之共享機車主動式安全防護系統

Press [1] for Normal Mode / Press [2] for Intervention Mode
"""

import sys
import os
import math
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QShortcut
)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QFont, QColor, QPainter, QPen, QPainterPath,
    QRadialGradient, QKeySequence, QPixmap
)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _tw(painter, text):
    """Get text pixel width from current painter font."""
    fm = painter.fontMetrics()
    if hasattr(fm, 'horizontalAdvance'):
        return fm.horizontalAdvance(text)
    return fm.width(text)


# ═══════════════════════════════════════════════════════════════
#  AI Vision Panel (Left Side)
# ═══════════════════════════════════════════════════════════════

class AIVisionPanel(QWidget):
    """Left panel: displays pre-rendered AI overlay images."""

    def __init__(self):
        super().__init__()
        self.is_warning = False
        self.setMinimumSize(520, 500)
        self._img_normal = QPixmap(os.path.join(_BASE_DIR, "normal_overlay.png"))
        self._img_smoking = QPixmap(os.path.join(_BASE_DIR, "smoking_overlay.png"))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        img = self._img_smoking if self.is_warning else self._img_normal
        p.fillRect(0, 0, w, h, QColor(10, 10, 14))

        if not img.isNull():
            iw, ih = img.width(), img.height()
            scale = max(w / iw, h / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            nx, ny = (w - nw) // 2, (h - nh) // 2
            scaled = img.scaled(nw, nh, Qt.IgnoreAspectRatio,
                                Qt.SmoothTransformation)
            p.drawPixmap(nx, ny, scaled)

    def set_state(self, warning):
        self.is_warning = warning
        self.update()


# ═══════════════════════════════════════════════════════════════
#  Dashboard Panel (Right Side) — Gogoro / GoShare Style
# ═══════════════════════════════════════════════════════════════

class DashboardPanel(QWidget):
    """Gogoro/GoShare-style virtual instrument cluster."""

    def __init__(self):
        super().__init__()
        self.speed = 45
        self.is_warning = False
        self.battery_pct = 78
        self.setMinimumSize(420, 500)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        cx = w / 2

        self._draw_bg(p, w, h)

        ring_cy = h * 0.36
        self._draw_speed_ring(p, cx, ring_cy, w, h)
        self._draw_speed_text(p, cx, ring_cy)
        self._draw_mode_label(p, cx, ring_cy, h)
        self._draw_battery(p, cx, h * 0.66)
        self._draw_go_button(p, cx, h * 0.82)
        self._draw_top_bar(p, w)

        if self.is_warning:
            self._draw_warning_elements(p, cx, ring_cy, w, h)

    def _draw_bg(self, p, w, h):
        bg = QRadialGradient(w / 2, h * 0.35, max(w, h) * 0.7)
        if self.is_warning:
            bg.setColorAt(0, QColor(28, 6, 6))
            bg.setColorAt(1, QColor(8, 2, 2))
        else:
            bg.setColorAt(0, QColor(16, 18, 24))
            bg.setColorAt(1, QColor(6, 6, 10))
        p.fillRect(0, 0, w, h, bg)

    def _draw_speed_ring(self, p, cx, cy, w, h):
        radius = min(w, h) * 0.22
        ring_w = 10
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        start_a = 225
        span = 270

        p.setPen(QPen(QColor(35, 37, 44), ring_w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, start_a * 16, -span * 16)

        frac = min(self.speed / 80.0, 1.0)

        if self.is_warning:
            ring_col = QColor(255, 45, 45)
            glow_col = QColor(255, 45, 45, 45)
        else:
            ring_col = QColor(235, 238, 245)
            glow_col = QColor(235, 238, 245, 30)

        p.setPen(QPen(glow_col, ring_w + 22, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, start_a * 16, int(-span * frac * 16))

        p.setPen(QPen(ring_col, ring_w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, start_a * 16, int(-span * frac * 16))

        num_ticks = 9
        for i in range(num_ticks):
            angle_deg = start_a - i * (span / (num_ticks - 1))
            angle_rad = math.radians(angle_deg)
            r_in = radius + 14
            r_out = radius + 24
            x1 = cx + r_in * math.cos(angle_rad)
            y1 = cy - r_in * math.sin(angle_rad)
            x2 = cx + r_out * math.cos(angle_rad)
            y2 = cy - r_out * math.sin(angle_rad)

            is_major = (i % 2 == 0)
            p.setPen(QPen(QColor(80, 82, 90) if is_major else QColor(50, 52, 58),
                          2 if is_major else 1))
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

            if is_major:
                lr = radius + 36
                lx = cx + lr * math.cos(angle_rad)
                ly = cy - lr * math.sin(angle_rad)
                val = str(i * 10)
                p.setFont(QFont("Segoe UI", 8))
                p.setPen(QColor(90, 92, 98))
                tw = _tw(p, val)
                p.drawText(int(lx - tw / 2), int(ly + 4), val)

    def _draw_speed_text(self, p, cx, cy):
        color = QColor(255, 50, 50) if self.is_warning else QColor(255, 255, 255)
        p.setPen(color)

        font = QFont("Segoe UI", 68, QFont.Bold)
        font.setLetterSpacing(QFont.AbsoluteSpacing, -2)
        p.setFont(font)
        txt = str(self.speed)
        tw = _tw(p, txt)
        asc = p.fontMetrics().ascent()
        p.drawText(int(cx - tw / 2), int(cy + asc / 3), txt)

        unit_col = QColor(140, 142, 148) if not self.is_warning else QColor(180, 70, 70)
        p.setPen(unit_col)
        p.setFont(QFont("Segoe UI", 13))
        utw = _tw(p, "km/h")
        p.drawText(int(cx - utw / 2), int(cy + asc / 3 + 28), "km/h")

    def _draw_mode_label(self, p, cx, ring_cy, h):
        if not self.is_warning:
            label, color = "NORMAL", QColor(100, 102, 108)
        else:
            label, color = "LIMITED", QColor(255, 60, 60)
        y = ring_cy + h * 0.20 + 30
        p.setFont(QFont("Segoe UI", 11, QFont.Bold))
        p.setPen(color)
        tw = _tw(p, label)
        p.drawText(int(cx - tw / 2), int(y), label)

    def _draw_battery(self, p, cx, y):
        bw, bh = 32, 52
        gap = 18

        for offset in [-gap - bw, gap]:
            bx = cx + offset
            by = y
            p.setPen(QPen(QColor(100, 102, 110), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(int(bx), int(by), bw, bh, 4, 4)

            tw = 12
            p.fillRect(int(bx + (bw - tw) / 2), int(by - 5), tw, 5,
                       QColor(100, 102, 110))

            fill_frac = self.battery_pct / 100.0
            fill_h = int((bh - 6) * fill_frac)
            fill_y = by + bh - 3 - fill_h
            fill_col = (QColor(76, 217, 100) if self.battery_pct > 20
                        else QColor(255, 59, 48))
            p.setPen(Qt.NoPen)
            p.setBrush(fill_col)
            p.drawRoundedRect(int(bx + 3), int(fill_y), bw - 6, fill_h, 2, 2)

            p.setPen(QPen(QColor(10, 10, 15), 1))
            for seg in range(1, 4):
                sy = by + bh - (bh - 6) * seg / 4
                p.drawLine(int(bx + 3), int(sy), int(bx + bw - 3), int(sy))

        p.setPen(QColor(170, 172, 178))
        p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        txt = f"{self.battery_pct}%"
        tw = _tw(p, txt)
        p.drawText(int(cx - tw / 2), int(y + bh + 24), txt)

    def _draw_go_button(self, p, cx, y):
        r = 30
        btn_col = (QColor(76, 217, 100) if not self.is_warning
                   else QColor(80, 80, 85))

        if not self.is_warning:
            glow = QRadialGradient(cx, y, r + 18)
            glow.setColorAt(0, QColor(76, 217, 100, 65))
            glow.setColorAt(1, QColor(76, 217, 100, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(glow)
            p.drawEllipse(int(cx - r - 18), int(y - r - 18),
                          (r + 18) * 2, (r + 18) * 2)

        p.setPen(Qt.NoPen)
        p.setBrush(btn_col)
        p.drawEllipse(int(cx - r), int(y - r), r * 2, r * 2)

        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont("Segoe UI", 16, QFont.Bold))
        tw = _tw(p, "GO")
        p.drawText(int(cx - tw / 2), int(y + 6), "GO")

    def _draw_top_bar(self, p, w):
        p.fillRect(0, 0, w, 42, QColor(0, 0, 0, 120))

        p.setFont(QFont("Segoe UI", 12))
        p.setPen(QColor(76, 217, 100))
        p.drawText(14, 28, "\u25cf Connected")

        p.setPen(QColor(180, 182, 188))
        t = datetime.now().strftime("%H:%M")
        tw = _tw(p, t)
        p.drawText(int(w / 2 - tw / 2), 28, t)

        p.setPen(QColor(100, 102, 108))
        p.drawText(w - 120, 28, "Jetson Nano")

    def _draw_warning_elements(self, p, cx, ring_cy, w, h):
        tri_y = ring_cy - h * 0.20
        sz = 18
        path = QPainterPath()
        path.moveTo(cx, tri_y - sz)
        path.lineTo(cx - sz * 0.9, tri_y + sz * 0.5)
        path.lineTo(cx + sz * 0.9, tri_y + sz * 0.5)
        path.closeSubpath()

        p.setPen(QPen(QColor(255, 50, 50), 2))
        p.setBrush(QColor(255, 50, 50, 35))
        p.drawPath(path)

        p.setPen(QPen(QColor(255, 50, 50), 2.5))
        p.drawLine(int(cx), int(tri_y - sz * 0.4),
                   int(cx), int(tri_y + sz * 0.1))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 50, 50))
        p.drawEllipse(int(cx - 2), int(tri_y + sz * 0.25), 5, 5)

        p.setPen(QColor(255, 50, 50))
        p.setFont(QFont("Segoe UI", 13, QFont.Bold))
        txt = "SPEED LIMITED"
        tw = _tw(p, txt)
        p.drawText(int(cx - tw / 2), int(tri_y + sz + 24), txt)

        py = h * 0.82 + 50
        pr = 14
        p.setPen(QPen(QColor(255, 50, 50, 170), 2.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(int(cx - pr), int(py - pr), pr * 2, pr * 2)
        p.drawLine(int(cx - pr * 0.7), int(py + pr * 0.7),
                   int(cx + pr * 0.7), int(py - pr * 0.7))

        p.setPen(QPen(QColor(255, 40, 40, 55), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(1, 1, w - 2, h - 2)

    def set_state(self, speed, warning):
        self.speed = speed
        self.is_warning = warning
        self.update()


# ═══════════════════════════════════════════════════════════════
#  Main Window
# ═══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "GoShare Active Safety System \u2014 AI Dashboard Mockup")
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)
        self.setStyleSheet("QMainWindow { background-color: #0a0a0e; }")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QWidget()
        cl = QHBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        left = QWidget()
        left.setStyleSheet("background-color: #111318;")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 10, 6, 10)

        header_l = QLabel("  面部偵測整合儀表板示意圖")
        header_l.setStyleSheet(
            "color: #00ffd0; font-family: Consolas; font-size: 13px;"
            " font-weight: bold; padding: 4px 0;")
        ll.addWidget(header_l)

        self.vision = AIVisionPanel()
        ll.addWidget(self.vision, stretch=1)

        sub = QLabel(
            "  YOLOv8-Pose  \u00b7  MediaPipe Face Mesh  \u00b7  ResNet-34")
        sub.setStyleSheet(
            "color: #444; font-family: Consolas; font-size: 10px;"
            " padding: 2px 0;")
        ll.addWidget(sub)

        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background-color: #222;")

        right = QWidget()
        right.setStyleSheet("background-color: #0a0a0e;")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        self.dashboard = DashboardPanel()
        rl.addWidget(self.dashboard, stretch=1)

        cl.addWidget(left, stretch=55)
        cl.addWidget(sep)
        cl.addWidget(right, stretch=45)
        root.addWidget(content, stretch=1)

        bar = QWidget()
        bar.setFixedHeight(56)
        bar.setStyleSheet(
            "background-color: #13151a; border-top: 1px solid #252730;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(20, 0, 20, 0)

        self.mode_label = QLabel()
        self.mode_label.setFont(QFont("Consolas", 13, QFont.Bold))
        bl.addWidget(self.mode_label)
        bl.addStretch()

        btn_n = QPushButton("  Normal Mode  [1]  ")
        btn_n.setCursor(Qt.PointingHandCursor)
        btn_n.setStyleSheet(
            "QPushButton { background: #152215; color: #00ff88;"
            " border: 1px solid #00ff88; border-radius: 5px;"
            " padding: 7px 20px; font: bold 12px Consolas; }"
            " QPushButton:hover { background: #1e3a1e; }")
        btn_n.clicked.connect(self.set_normal)

        btn_w = QPushButton("  Intervention Mode  [2]  ")
        btn_w.setCursor(Qt.PointingHandCursor)
        btn_w.setStyleSheet(
            "QPushButton { background: #2a1212; color: #ff4444;"
            " border: 1px solid #ff4444; border-radius: 5px;"
            " padding: 7px 20px; font: bold 12px Consolas; }"
            " QPushButton:hover { background: #3e1a1a; }")
        btn_w.clicked.connect(self.set_warning)

        bl.addWidget(btn_n)
        bl.addWidget(btn_w)

        hint = QLabel("  Press 1 / 2 to switch mode")
        hint.setStyleSheet("color: #555; font: 11px Consolas;")
        bl.addWidget(hint)

        root.addWidget(bar)

        QShortcut(QKeySequence("1"), self, self.set_normal)
        QShortcut(QKeySequence("2"), self, self.set_warning)

        self.set_normal()

    def set_normal(self):
        self.vision.set_state(False)
        self.dashboard.set_state(45, False)
        self.mode_label.setText("MODE: NORMAL OPERATION")
        self.mode_label.setStyleSheet(
            "color: #00ff88; font: bold 13px Consolas;")

    def set_warning(self):
        self.vision.set_state(True)
        self.dashboard.set_state(30, True)
        self.mode_label.setText("MODE: LEVEL 2 INTERVENTION \u26a0")
        self.mode_label.setStyleSheet(
            "color: #ff4444; font: bold 13px Consolas;")

    def export_png(self, filename):
        """Render current state to a PNG file."""
        self.repaint()
        app = QApplication.instance()
        app.processEvents()
        pixmap = self.grab()
        pixmap.save(filename, "PNG")
        print(f"  Saved: {filename}")


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()

    if "--save" in sys.argv:
        win.show()
        app.processEvents()

        win.set_normal()
        app.processEvents()
        win.export_png("normal_mode.png")

        win.set_warning()
        app.processEvents()
        win.export_png("intervention_mode.png")

        print("\n  Done! Two images saved in current directory.")
        sys.exit(0)
    else:
        win.show()
        sys.exit(app.exec_())
