"""
core/alert.py — Class AlertManager, quản lý âm thanh cảnh báo và overlay trên frame.

Phát âm thanh qua pygame.mixer khi phát hiện ngủ gật hoặc ngáp,
và vẽ overlay cảnh báo đẹp lên frame với hiệu ứng nhấp nháy.
"""

import os
import time

import cv2
import numpy as np
import pygame

from core.config import (
    ALARM_PATH, CLR_RED, CLR_YELLOW, CLR_WHITE,
    CLR_BG_DARK, CLR_ALERT_BORDER, CLR_ALERT_YAWN_BORDER
)


class AlertManager:
    """Quản lý cảnh báo âm thanh và overlay hình ảnh."""

    def __init__(self, alarm_path=None):
        """
        Khởi tạo AlertManager.

        Args:
            alarm_path: Đường dẫn file alarm.wav.
                       Nếu None, dùng ALARM_PATH từ config.

        Raises:
            FileNotFoundError: Nếu file âm thanh không tồn tại.
        """
        if alarm_path is None:
            alarm_path = ALARM_PATH

        # Kiểm tra file âm thanh tồn tại
        if not os.path.isfile(alarm_path):
            raise FileNotFoundError(
                f"Không tìm thấy file âm thanh tại: {alarm_path}\n"
                f"Vui lòng đặt file alarm.wav vào thư mục sounds/\n"
                f"Hoặc chạy generate_alarm.py để tạo file âm thanh mẫu."
            )

        # Khởi tạo pygame mixer
        pygame.mixer.init()

        # Load file âm thanh cảnh báo
        self._alarm_sound = pygame.mixer.Sound(alarm_path)

        # Cờ trạng thái cảnh báo
        self._is_drowsy_alerting = False
        self._is_yawn_alerting = False

        # Thời điểm bắt đầu cảnh báo (dùng cho hiệu ứng nhấp nháy)
        self._alert_start_time = 0.0

    def trigger_drowsy_alert(self):
        """
        Kích hoạt cảnh báo ngủ gatts.

        Phát âm thanh alarm lặp vô hạn.
        Đặt cờ và ghi nhận thời điểm bắt đầu.
        """
        if not self._is_drowsy_alerting:
            self._alarm_sound.play(loops=-1)
            self._is_drowsy_alerting = True
            self._alert_start_time = time.time()

    def trigger_yawn_alert(self):
        """Kích hoạt cảnh báo ngáp."""
        if not self._is_yawn_alerting and not self._is_drowsy_alerting:
            self._alarm_sound.play(loops=-1)
            self._is_yawn_alerting = True
            self._alert_start_time = time.time()

    def stop_alert(self):
        """Dừng tất cả âm thanh cảnh báo và reset cờ."""
        if self._is_drowsy_alerting or self._is_yawn_alerting:
            self._alarm_sound.stop()
            self._is_drowsy_alerting = False
            self._is_yawn_alerting = False

    @property
    def is_alerting(self):
        """Kiểm tra xem có đang cảnh báo hay không."""
        return self._is_drowsy_alerting or self._is_yawn_alerting

    def _get_blink_factor(self):
        """
        Tính hệ số nhấp nháy (0.0 ~ 1.0) cho hiệu ứng cảnh báo.

        Dùng hàm sin để tạo hiệu ứng pulse nhấp nháy mượt mà.

        Returns:
            float: Hệ số alpha (0 = mờ, 1 = đậm).
        """
        elapsed = time.time() - self._alert_start_time
        # Nhấp nháy 3Hz (3 lần/giây)
        return 0.5 + 0.5 * abs(np.sin(elapsed * np.pi * 3))

    def draw_alert_overlay(self, frame, alert_type):
        """
        Vẽ overlay cảnh báo đẹp lên frame.

        - alert_type="drowsy": border đỏ nhấp nháy + text cảnh báo lớn.
        - alert_type="yawn": border vàng cam + text cảnh báo ngáp.

        Args:
            frame: numpy.ndarray — ảnh BGR để vẽ overlay lên.
            alert_type: str — "drowsy" hoặc "yawn".

        Returns:
            numpy.ndarray: Frame đã vẽ overlay.
        """
        h, w = frame.shape[:2]
        blink = self._get_blink_factor()

        if alert_type == "drowsy":
            # Border đỏ nhấp nháy toàn frame
            border_color = tuple(int(c * blink) for c in CLR_ALERT_BORDER)
            thickness = int(4 + 4 * blink)  # Độ dày nhấp nháy

            # Vẽ border 4 cạnh
            cv2.rectangle(frame, (0, 0), (w, h), border_color, thickness)

            # Vẽ thêm viền mờ bên trong (glow effect)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), CLR_RED, thickness + 4)
            cv2.addWeighted(overlay, 0.15 * blink, frame, 1 - 0.15 * blink, 0, frame)

            # Panel cảnh báo giữa frame
            self._draw_alert_panel(
                frame, w, h,
                title="⚠ CANH BAO!",
                subtitle="BUON NGU!",
                color=CLR_RED,
                blink=blink
            )

        elif alert_type == "yawn":
            # Border vàng cam nhấp nháy
            border_color = tuple(int(c * blink) for c in CLR_ALERT_YAWN_BORDER)
            thickness = int(3 + 3 * blink)

            cv2.rectangle(frame, (0, 0), (w, h), border_color, thickness)

            # Panel cảnh báo ngáp
            self._draw_alert_panel(
                frame, w, h,
                title="⚠ CANH BAO!",
                subtitle="DANG NGAP!",
                color=CLR_YELLOW,
                blink=blink
            )

        return frame

    def _draw_alert_panel(self, frame, w, h, title, subtitle, color, blink):
        """
        Vẽ panel cảnh báo dạng card giữa frame.

        Args:
            frame: Ảnh BGR.
            w, h: Chiều rộng, cao frame.
            title: Text tiêu đề cảnh báo.
            subtitle: Text phụ cảnh báo.
            color: Màu chính (BGR).
            blink: Hệ số nhấp nháy (0~1).
        """
        # Kích thước panel
        panel_w, panel_h = 420, 120
        px = (w - panel_w) // 2
        py = (h - panel_h) // 2

        # Vẽ nền panel (semi-transparent dark)
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), CLR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Viền panel
        panel_border = tuple(min(int(c * blink) + 30, 255) for c in color)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), panel_border, 2, cv2.LINE_AA)

        # Góc bo tròn (vẽ 4 góc nhỏ)
        corner_len = 20
        for cx, cy in [(px, py), (px + panel_w, py), (px, py + panel_h), (px + panel_w, py + panel_h)]:
            dx = corner_len if cx == px else -corner_len
            dy = corner_len if cy == py else -corner_len
            cv2.line(frame, (cx, cy), (cx + dx, cy), panel_border, 3, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy), panel_border, 3, cv2.LINE_AA)

        # Text tiêu đề
        font_title = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(title, font_title, 1.0, 2)
        cv2.putText(
            frame, title,
            (px + (panel_w - tw) // 2, py + 45),
            font_title, 1.0, color, 2, cv2.LINE_AA
        )

        # Text phụ
        font_sub = cv2.FONT_HERSHEY_SIMPLEX
        (sw, sh), _ = cv2.getTextSize(subtitle, font_sub, 0.8, 2)
        cv2.putText(
            frame, subtitle,
            (px + (panel_w - sw) // 2, py + 90),
            font_sub, 0.8, CLR_WHITE, 2, cv2.LINE_AA
        )

    def cleanup(self):
        """Dọn dẹp tài nguyên pygame khi thoát chương trình."""
        self.stop_alert()
        pygame.mixer.quit()
