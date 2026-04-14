"""
core/utils.py — Hàm tiện ích: vẽ landmarks, convex hull, dashboard HUD đẹp.

Cung cấp các hàm vẽ trực quan lên frame OpenCV
để hiển thị dashboard chuyên nghiệp với panel, gradient, EAR bar.
"""

import cv2
import numpy as np

from core.config import (
    EAR_THRESH, MAR_THRESH,
    EAR_CONSEC_FRAMES, YAWN_CONSEC_FRAMES,
    CLR_BG_DARK, CLR_BG_PANEL, CLR_GREEN, CLR_RED,
    CLR_BLUE, CLR_YELLOW, CLR_WHITE, CLR_CYAN, CLR_ORANGE,
    CLR_GRAY, CLR_DARK_GRAY,
    CLR_BAR_LOW, CLR_BAR_MID, CLR_BAR_HIGH
)


# ============================================================
# Vẽ landmarks & hull
# ============================================================

def draw_landmarks(frame, shape, color=CLR_BLUE):
    """
    Vẽ 68 điểm landmark trên khuôn mặt lên frame.

    Vẽ dạng chấm nhỏ màu xanh dương, nối các điểm theo nhóm
    (mắt, mũi, miệng, hàm) bằng đường mờ.

    Args:
        frame: numpy.ndarray — ảnh BGR để vẽ lên.
        shape: numpy.ndarray shape (68, 2) — tọa độ 68 điểm landmark.
        color: tuple — màu BGR của điểm landmark (mặc định: xanh dương).

    Returns:
        numpy.ndarray: Frame đã vẽ landmarks.
    """
    # Vẽ các điểm (vector hóa tọa độ)
    for i in range(68):
        x, y = int(shape[i, 0]), int(shape[i, 1])
        cv2.circle(frame, (x, y), 2, color, -1, cv2.LINE_AA)

    # Nối đường theo nhóm — một lệnh polylines mỗi nhóm (nhanh hơn nhiều cv2.line)
    connections = [
        # Hàm
        list(range(0, 17)),
        # Lông mày trái
        list(range(17, 22)),
        # Lông mày phải
        list(range(22, 27)),
        # Mũi
        list(range(27, 31)),
        # Cầu mũi
        list(range(31, 36)),
        # Mắt trái
        list(range(36, 42)) + [36],
        # Mắt phải
        list(range(42, 48)) + [42],
        # Môi ngoài
        list(range(48, 60)) + [48],
        # Môi trong
        list(range(60, 68)) + [60],
    ]

    dim_color = tuple(max(c // 2, 0) for c in color)
    for group in connections:
        pts = shape[group].astype(np.int32).reshape(-1, 1, 2)
        closed = len(group) > 1 and int(group[0]) == int(group[-1])
        cv2.polylines(frame, [pts], closed, dim_color, 1, cv2.LINE_AA)

    return frame


def draw_eye_hull(frame, eye, color=CLR_GREEN):
    """
    Vẽ convex hull quanh mắt lên frame.

    Vẽ đường viền bao quanh mắt (convex hull) với
    semi-transparent fill để trực quan hóa vùng mắt.

    Args:
        frame: numpy.ndarray — ảnh BGR để vẽ lên.
        eye: numpy.ndarray shape (6, 2) — tọa độ 6 điểm landmark của một mắt.
        color: tuple — màu BGR (mặc định: xanh lá).

    Returns:
        numpy.ndarray: Frame đã vẽ convex hull mắt.
    """
    hull = cv2.convexHull(eye)

    # Fill semi-transparent
    overlay = frame.copy()
    cv2.drawContours(overlay, [hull], -1, color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Viền ngoài
    cv2.drawContours(frame, [hull], -1, color, 1, cv2.LINE_AA)

    return frame


def draw_mouth_hull(frame, mouth, color=CLR_ORANGE):
    """
    Vẽ convex hull quanh miệng lên frame.

    Args:
        frame: numpy.ndarray — ảnh BGR để vẽ lên.
        mouth: numpy.ndarray shape (20, 2) — tọa độ 20 điểm miệng.
        color: tuple — màu BGR (mặc định: cam).

    Returns:
        numpy.ndarray: Frame đã vẽ convex hull miệng.
    """
    hull = cv2.convexHull(mouth)

    # Fill semi-transparent
    overlay = frame.copy()
    cv2.drawContours(overlay, [hull], -1, color, -1)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    # Viền ngoài
    cv2.drawContours(frame, [hull], -1, color, 1, cv2.LINE_AA)

    return frame


def draw_face_box(frame, face, color=CLR_CYAN, label="Face"):
    """
    Vẽ bounding box quanh khuôn mặt với label.

    Vẽ box bo tròn góc + label phía trên.

    Args:
        frame: numpy.ndarray — ảnh BGR.
        face: dlib.rectangle — vùng mặt.
        color: tuple — màu BGR (mặc định: cyan).
        label: str — nhãn hiển thị.

    Returns:
        numpy.ndarray: Frame đã vẽ bounding box.
    """
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    # Đảm bảo tọa độ không âm
    x1, y1 = max(0, x1), max(0, y1)

    # Vẽ box chính
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Góc bo tròn (corner accents)
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    for cx, cy, dx, dy in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, 3, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, 3, cv2.LINE_AA)

    # Label phía trên box
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), CLR_BG_DARK, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4), font, 0.45, color, 1, cv2.LINE_AA)

    return frame


# ============================================================
# Dashboard HUD
# ============================================================

def draw_ear_bar(frame, ear_value, thresh=EAR_THRESH):
    """
    Vẽ thanh bar EAR gradient ở cạnh phải frame.

    Thanh cao = EAR cao = mắt mở (xanh).
    Thanh thấp = EAR thấp = mắt nhắm (đỏ).
    Gradient: đỏ → vàng → xanh tùy theo giá trị EAR.

    Args:
        frame: numpy.ndarray — ảnh BGR để vẽ lên.
        ear_value: float — giá trị EAR hiện tại (0.0 ~ 0.5).
        thresh: float — ngưỡng EAR.

    Returns:
        numpy.ndarray: Frame đã vẽ thanh bar EAR.
    """
    h, w = frame.shape[:2]

    # Kích thước thanh bar
    bar_w = 24
    bar_h = h - 120
    bar_x = w - bar_w - 30
    bar_y = 70

    # Vẽ nền panel cho bar
    panel_x = bar_x - 12
    panel_y = bar_y - 35
    panel_w = bar_w + 24
    panel_h = bar_h + 55
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), CLR_BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), CLR_DARK_GRAY, 1, cv2.LINE_AA)

    # Tiêu đề "EAR"
    font_sm = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "EAR", (bar_x - 2, bar_y - 12), font_sm, 0.4, CLR_GRAY, 1, cv2.LINE_AA)

    # Vẽ nền thanh bar (xám đậm)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 35, 35), -1)

    # Chuẩn hóa EAR về [0, 1] (giả sử EAR tối đa ~0.45)
    ear_norm = min(max(ear_value / 0.45, 0.0), 1.0)
    fill_h = int(bar_h * ear_norm)

    # Gradient fill: một strip numpy + gán ROI (thay vòng for hàng trăm cv2.line)
    if fill_h > 0:
        i_arr = np.arange(fill_h, dtype=np.float64)
        ratio = i_arr / float(bar_h)
        mask_low = ratio < 0.5
        t_low = np.clip(ratio * 2.0, 0.0, 1.0)
        t_high = np.clip((ratio - 0.5) * 2.0, 0.0, 1.0)
        c_lo = np.array(CLR_BAR_LOW, dtype=np.float64)
        c_mid = np.array(CLR_BAR_MID, dtype=np.float64)
        c_hi = np.array(CLR_BAR_HIGH, dtype=np.float64)
        row_rgb = np.empty((fill_h, 3), dtype=np.float64)
        row_rgb[mask_low] = c_lo + (c_mid - c_lo) * t_low[:, np.newaxis][mask_low]
        row_rgb[~mask_low] = c_mid + (c_hi - c_mid) * t_high[:, np.newaxis][~mask_low]
        y0 = bar_y + bar_h - fill_h
        # Gán có broadcast (fill_h, 1, 3) → vùng (fill_h, inner_w, 3)
        frame[y0 : y0 + fill_h, bar_x + 1 : bar_x + bar_w - 1] = row_rgb.astype(
            np.uint8
        )[:, np.newaxis, :]

    # Đường ngưỡng tham chiếu
    thresh_norm = min(max(thresh / 0.45, 0.0), 1.0)
    thresh_y = bar_y + bar_h - int(bar_h * thresh_norm)
    cv2.line(frame, (bar_x - 4, thresh_y), (bar_x + bar_w + 4, thresh_y), CLR_WHITE, 1, cv2.LINE_AA)
    # Nhãn ngưỡng
    cv2.putText(frame, f"{thresh:.2f}", (bar_x - 4, thresh_y - 5), font_sm, 0.3, CLR_GRAY, 1, cv2.LINE_AA)

    # Giá trị EAR hiện tại dưới bar
    ear_color = CLR_GREEN if ear_value >= thresh else CLR_RED
    cv2.putText(frame, f"{ear_value:.2f}", (bar_x - 2, bar_y + bar_h + 18), font_sm, 0.4, ear_color, 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, ear, mar, status, fps, ear_counter, yawn_counter,
             ear_thresh=EAR_THRESH, mar_thresh=MAR_THRESH,
             ear_consec=EAR_CONSEC_FRAMES, yawn_consec=YAWN_CONSEC_FRAMES):
    """
    Vẽ dashboard HUD chuyên nghiệp lên frame.

    Bố cục:
    - Panel trái trên: EAR, MAR, Status
    - Panel phải trên: FPS, counter
    - Thanh bar EAR cạnh phải

    Args:
        frame: numpy.ndarray — ảnh BGR.
        ear: float — giá trị EAR hiện tại.
        mar: float — giá trị MAR hiện tại.
        status: str — trạng thái ("NORMAL", "DROWSY", "YAWNING").
        fps: float — FPS hiện tại.
        ear_counter: int — số frame liên tiếp EAR thấp.
        yawn_counter: int — số frame liên tiếp MAR cao.
        ear_thresh: float — ngưỡng EAR.
        mar_thresh: float — ngưỡng MAR.
        ear_consec: int — số frame cần đạt để cảnh báo drowsy.
        yawn_consec: int — số frame cần đạt để cảnh báo yawn.

    Returns:
        numpy.ndarray: Frame đã vẽ HUD.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_sm = cv2.FONT_HERSHEY_SIMPLEX

    # === Panel trái trên: thông số ===
    panel_w, panel_h = 220, 130
    panel_x, panel_y = 10, 10

    # Vẽ nền panel (semi-transparent)
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), CLR_BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Viền panel
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), CLR_DARK_GRAY, 1, cv2.LINE_AA)

    # Tiêu đề panel
    cv2.putText(frame, "DROWSINESS DETECTION",
                (panel_x + 8, panel_y + 18), font_sm, 0.38, CLR_CYAN, 1, cv2.LINE_AA)
    cv2.line(frame, (panel_x + 5, panel_y + 24), (panel_x + panel_w - 5, panel_y + 24), CLR_DARK_GRAY, 1, cv2.LINE_AA)

    # --- EAR ---
    ear_color = CLR_GREEN if ear >= ear_thresh else CLR_RED
    y_pos = panel_y + 44
    cv2.putText(frame, "EAR", (panel_x + 10, y_pos), font_sm, 0.42, CLR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{ear:.3f}", (panel_x + 60, y_pos), font_sm, 0.55, ear_color, 2, cv2.LINE_AA)

    # EAR mini bar
    bar_x = panel_x + 140
    bar_w = 65
    bar_h = 10
    cv2.rectangle(frame, (bar_x, y_pos - 9), (bar_x + bar_w, y_pos - 9 + bar_h), (35, 35, 35), -1)
    ear_fill = int(bar_w * min(max(ear / 0.45, 0.0), 1.0))
    cv2.rectangle(frame, (bar_x, y_pos - 9), (bar_x + ear_fill, y_pos - 9 + bar_h), ear_color, -1)
    cv2.line(frame, (bar_x + int(bar_w * ear_thresh / 0.45), y_pos - 9),
             (bar_x + int(bar_w * ear_thresh / 0.45), y_pos + 1), CLR_WHITE, 1, cv2.LINE_AA)

    # --- MAR ---
    mar_color = CLR_GREEN if mar < mar_thresh else CLR_ORANGE
    y_pos = panel_y + 70
    cv2.putText(frame, "MAR", (panel_x + 10, y_pos), font_sm, 0.42, CLR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{mar:.3f}", (panel_x + 60, y_pos), font_sm, 0.55, mar_color, 2, cv2.LINE_AA)

    # MAR mini bar
    cv2.rectangle(frame, (bar_x, y_pos - 9), (bar_x + bar_w, y_pos - 9 + bar_h), (35, 35, 35), -1)
    mar_fill = int(bar_w * min(max(mar / 1.0, 0.0), 1.0))
    cv2.rectangle(frame, (bar_x, y_pos - 9), (bar_x + mar_fill, y_pos - 9 + bar_h), mar_color, -1)

    # --- Status ---
    if status == "NORMAL":
        status_color = CLR_GREEN
        status_icon = "[OK]"
    elif status == "DROWSY":
        status_color = CLR_RED
        status_icon = "[!!]"
    else:
        status_color = CLR_YELLOW
        status_icon = "[~]"

    y_pos = panel_y + 96
    cv2.putText(frame, "Status", (panel_x + 10, y_pos), font_sm, 0.42, CLR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{status_icon} {status}", (panel_x + 75, y_pos), font_sm, 0.5, status_color, 2, cv2.LINE_AA)

    # --- Counter nhỏ ---
    y_pos = panel_y + 120
    cv2.putText(frame, f"E:{ear_counter}/{ear_consec}", (panel_x + 10, y_pos), font_sm, 0.32, CLR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Y:{yawn_counter}/{yawn_consec}", (panel_x + 110, y_pos), font_sm, 0.32, CLR_GRAY, 1, cv2.LINE_AA)

    # === Panel phải trên: FPS ===
    fps_w, fps_h = 90, 30
    fps_x = w - fps_w - 60
    fps_y = 10

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (fps_x, fps_y), (fps_x + fps_w, fps_y + fps_h), CLR_BG_PANEL, -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (fps_x, fps_y), (fps_x + fps_w, fps_y + fps_h), CLR_DARK_GRAY, 1, cv2.LINE_AA)

    fps_color = CLR_GREEN if fps >= 25 else (CLR_YELLOW if fps >= 15 else CLR_RED)
    cv2.putText(frame, f"FPS {fps:.0f}", (fps_x + 8, fps_y + 22), font_sm, 0.55, fps_color, 2, cv2.LINE_AA)

    return frame
