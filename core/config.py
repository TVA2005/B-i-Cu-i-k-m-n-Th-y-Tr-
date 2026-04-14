"""
core/config.py — Hằng số cấu hình cho hệ thống phát hiện ngủ gật tài xế.

Chứa các ngưỡng EAR, MAR, số frame liên tiếp để cảnh báo,
thông số camera, kích thước frame, và bảng màu dashboard.
"""

import os

# === Đường dẫn gốc dự án (tính từ vị trí file này) ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Ngưỡng phát hiện ===
EAR_THRESH = 0.25          # Ngưỡng EAR để coi là mắt nhắm
MAR_THRESH = 0.6           # Ngưỡng MAR để coi là đang ngáp

# === Số frame liên tiếp ===
EAR_CONSEC_FRAMES = 20     # Số frames liên tiếp EAR < thresh → cảnh báo ngủ gật
YAWN_CONSEC_FRAMES = 15    # Số frames liên tiếp MAR > thresh → cảnh báo ngáp

# === Camera ===
CAMERA_INDEX = 0           # Chỉ số camera mặc định (0 = webcam tích hợp)

# === Kích thước frame ===
FRAME_WIDTH = 640          # Chiều rộng frame sau resize (tăng tốc xử lý)
FRAME_HEIGHT = 480         # Chiều cao frame sau resize

# === Đường dẫn (dùng đường dẫn tương đối từ BASE_DIR) ===
MODEL_PATH = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")
ALARM_PATH = os.path.join(BASE_DIR, "sounds", "alarm.wav")

# === Phát hiện mặt ===
FACE_UPSAMPLE = 1          # Số lần upsample cho HOG detector (0= nhanh, 1= chuẩn, 2= chính xác chậm)
TRACKING_QUALITY_THRESH = 7.0   # Ngưỡng chất lượng correlation tracker (thấp hơn -> coi như mất track)
REDETECT_INTERVAL = 10          # Số frame/lần ép detect lại để bám mặt ổn định hơn khi quay đầu

# === Smoothing EAR/MAR (tránh nhiễu) ===
SMOOTHING_WINDOW = 5       # Số frame dùng cho moving average

# === Bảng màu dashboard (BGR) ===
# Màu chính
CLR_BG_DARK = (30, 30, 30)          # Nền tối dashboard
CLR_BG_PANEL = (40, 40, 50)         # Nền panel
CLR_GREEN = (0, 220, 80)            # Xanh lá — trạng thái bình thường
CLR_RED = (0, 60, 255)              # Đỏ — trạng thái cảnh báo
CLR_BLUE = (220, 130, 50)           # Xanh dương — landmarks
CLR_YELLOW = (60, 255, 255)        # Vàng — cảnh báo ngáp
CLR_WHITE = (240, 240, 240)        # Trắng — text chung
CLR_CYAN = (240, 200, 60)          # Cyan — accent
CLR_ORANGE = (0, 165, 255)         # Cam — cảnh báo trung bình
CLR_GRAY = (160, 160, 160)         # Xám — text phụ
CLR_DARK_GRAY = (80, 80, 80)       # Xám đậm — viền

# Màu gradient cho EAR bar (từ đỏ → vàng → xanh)
CLR_BAR_LOW = (0, 60, 255)         # Đỏ — EAR thấp
CLR_BAR_MID = (0, 255, 255)        # Vàng — EAR trung bình
CLR_BAR_HIGH = (0, 220, 80)        # Xanh — EAR cao

# Màu border cảnh báo
CLR_ALERT_BORDER = (0, 0, 255)     # Đỏ đậm — border cảnh báo ngủ gật
CLR_ALERT_YAWN_BORDER = (0, 200, 255)  # Vàng cam — border cảnh báo ngáp
