# Driver Drowsiness Detection System

Hệ thống cảnh báo tài xế ngủ gật theo thời gian thực sử dụng webcam.

## Tính năng nâng cấp

- **Dashboard HUD chuyên nghiệp** với panel, gradient bar, hiệu ứng mờ (glassmorphism)
- **Face tracking** giữa các frame để ổn định kết quả phát hiện
- **EAR/MAR smoothing** để giảm nhiễu, giá trị ổn định hơn
- **Upsampling HOG detector** để phát hiện mặt chính xác hơn
- **Convex hull** cho mắt và miệng với semi-transparent fill
- **Overlay cảnh báo đẹp** với hiệu ứng nhấp nháy và panel cảnh báo
- 68 điểm landmark trên khuôn mặt + bounding box bo tròn góc

## Cài đặt

### 1. Cài đặt Python packages

```bash
pip install -r requirements.txt
```

**Lưu ý cho Windows:** Nếu `pip install dlib` bị lỗi build, dùng prebuilt wheel:

```bash
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-20.0.99-cp314-cp314-win_amd64.whl
```

Tải wheel từ: https://github.com/z-mahmud22/Dlib_Windows_Python3.x

### 2. Tải Dlib model

Tải file `shape_predictor_68_face_landmarks.dat` từ:

https://github.com/davisking/dlib-models

Giải nén và đặt vào thư mục `models/`:

```
drowsiness_detection/
└── models/
    └── shape_predictor_68_face_landmarks.dat
```

### 3. Tạo file âm thanh cảnh báo

```bash
python generate_alarm.py
```

## Chạy chương trình

```bash
python main.py
```

Nhấn **Q** để thoát.

## Cấu trúc dự án mới

```
drowsiness_detection/
├── main.py                  # Entry point
├── core/                    # Package chính
│   ├── __init__.py          # Exports
│   ├── config.py            # Hằng số cấu hình
│   ├── detector.py          # FaceDetector với tracking
│   ├── alert.py             # AlertManager với overlay đẹp
│   └── utils.py             # Dashboard HUD, vẽ landmarks
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── sounds/
│   └── alarm.wav
├── generate_alarm.py        # Tạo âm thanh mẫu
├── requirements.txt
└── README.md
```

## Cấu hình tùy chỉnh

Chỉnh sửa `core/config.py` để thay đổi:

| Tham số | Giá trị mặc định | Mô tả |
|---------|------------------|-------|
| EAR_THRESH | 0.25 | Ngưỡng EAR coi là mắt nhắm |
| MAR_THRESH | 0.6 | Ngưỡng MAR coi là ngáp |
| EAR_CONSEC_FRAMES | 20 | Số frame liên tiếp mắt nhắm → cảnh báo |
| YAWN_CONSEC_FRAMES | 15 | Số frame liên tiếp ngáp → cảnh báo |
| FACE_UPSAMPLE | 1 | Upsampling cho HOG (0=nhanh, 1=chuẩn, 2=chậm) |
| SMOOTHING_WINDOW | 5 | Số frame dùng cho moving average |
