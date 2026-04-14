"""
core/detector.py — Class FaceDetector, logic phát hiện mặt và tính EAR/MAR.

Sử dụng Dlib HOG face detector + shape predictor 68 điểm
để phát hiện mặt và tính toán các chỉ số mắt (EAR) và miệng (MAR).
Hỗ trợ upsampling để bắt mặt chính xác hơn,
và face tracking để ổn định kết quả giữa các frame.
"""

import os
from collections import deque

import dlib
import numpy as np

from core.config import (
    MODEL_PATH,
    FACE_UPSAMPLE,
    SMOOTHING_WINDOW,
    TRACKING_QUALITY_THRESH,
)


def _euclidean(a, b):
    """Khoảng cách L2 giữa hai điểm 2D (nhanh hơn scipy cho vòng lặp nhỏ mỗi frame)."""
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


def _dlib_safe_model_path(path: str) -> str:
    """Dlib C++ trên Windows hay lỗi mở file nếu đường dẫn có Unicode — dùng short path 8.3."""
    if os.name != "nt":
        return path
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
        if n and n < len(buf):
            return buf.value
    except Exception:
        pass
    return path


class FaceDetector:
    """Phát hiện mặt và tính toán EAR/MAR từ webcam frame."""

    # Chỉ số landmark cho mắt trái và mắt phải (6 điểm mỗi mắt)
    LEFT_EYE_IDX = np.array(list(range(36, 42)))
    RIGHT_EYE_IDX = np.array(list(range(42, 48)))

    # Chỉ số landmark cho miệng (20 điểm: 48-67)
    MOUTH_IDX = np.array(list(range(48, 68)))

    # Chỉ số landmark cho lông mày (để vẽ)
    LEFT_BROW_IDX = np.array(list(range(17, 22)))
    RIGHT_BROW_IDX = np.array(list(range(22, 27)))

    # Chỉ số landmark cho hàm (để vẽ bounding box)
    JAW_IDX = np.array(list(range(0, 17)))

    def __init__(self, predictor_path=None, upsample=FACE_UPSAMPLE):
        """
        Khởi tạo FaceDetector.

        Args:
            predictor_path: Đường dẫn file shape_predictor_68_face_landmarks.dat.
                           Nếu None, dùng MODEL_PATH từ config.
            upsample: Số lần upsample cho HOG detector (0=nhanh, 1=chuẩn, 2=chính xác chậm).

        Raises:
            FileNotFoundError: Nếu file model không tồn tại.
        """
        if predictor_path is None:
            predictor_path = MODEL_PATH

        # Kiểm tra file model tồn tại
        if not os.path.isfile(predictor_path):
            raise FileNotFoundError(
                f"Không tìm thấy model Dlib tại: {predictor_path}\n"
                f"Vui lòng tải shape_predictor_68_face_landmarks.dat từ:\n"
                f"https://github.com/davisking/dlib-models\n"
                f"Và đặt vào thư mục models/"
            )

        # Khởi tạo HOG face detector với upsampling
        self._detector = dlib.get_frontal_face_detector()
        self._upsample = upsample

        # Khởi tạo shape predictor từ file model (đường dẫn an toàn cho dlib trên Windows)
        self._predictor = dlib.shape_predictor(_dlib_safe_model_path(predictor_path))

        # Correlation tracker để track mặt giữa các frame (tăng ổn định)
        self._tracker = None
        self._tracking = False
        self._tracking_quality = 0.0

        # Bộ đệm smoothing EAR/MAR (moving average O(1) mỗi frame)
        self._ear_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self._mar_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self._ear_sum = 0.0
        self._mar_sum = 0.0

    def detect(self, frame):
        """
        Phát hiện các khuôn mặt trong frame grayscale.

        Sử dụng HOG detector với upsampling để tăng độ chính xác.
        Nếu đang tracking, ưu tiên dùng tracker.

        Args:
            frame: Ảnh grayscale (numpy array, shape HxW).

        Returns:
            list[dlib.rectangle]: Danh sách các vùng mặt phát hiện được.
        """
        # Dlib HOG detector với upsampling
        faces = self._detector(frame, self._upsample)
        return faces

    def start_tracking(self, frame, face):
        """
        Bắt đầu theo dõi (track) một khuôn mặt.

        Dùng dlib correlation tracker để theo dõi mặt
        giữa các frame, giúp ổn định kết quả.

        Args:
            frame: Ảnh BGR hoặc grayscale.
            face: dlib.rectangle — vùng mặt cần track.
        """
        self._tracker = dlib.correlation_tracker()
        self._tracker.start_track(frame, face)
        self._tracking = True
        self._tracking_quality = TRACKING_QUALITY_THRESH

    def update_tracking(self, frame):
        """
        Cập nhật vị trí mặt đang track.

        Args:
            frame: Ảnh BGR hoặc grayscale.

        Returns:
            dlib.drectangle hoặc None: Vị trí mặt mới, hoặc None nếu mất track.
        """
        if not self._tracking or self._tracker is None:
            return None

        try:
            quality = float(self._tracker.update(frame))
            self._tracking_quality = quality
            if quality < TRACKING_QUALITY_THRESH:
                self._tracking = False
                return None
            pos = self._tracker.get_position()
            # Chuyển drectangle thành rectangle thông thường
            rect = dlib.rectangle(
                int(pos.left()), int(pos.top()),
                int(pos.right()), int(pos.bottom())
            )
            return rect
        except Exception:
            self._tracking = False
            self._tracking_quality = 0.0
            return None

    def stop_tracking(self):
        """Dừng theo dõi mặt."""
        self._tracking = False
        self._tracker = None
        self._tracking_quality = 0.0

    @property
    def is_tracking(self):
        """Kiểm tra có đang track mặt hay không."""
        return self._tracking

    @property
    def tracking_quality(self):
        """Chất lượng tracker hiện tại (số càng cao càng tốt)."""
        return self._tracking_quality

    def get_shape(self, frame, face):
        """
        Lấy 68 điểm landmark cho một khuôn mặt.

        Args:
            frame: Ảnh grayscale (numpy array).
            face: dlib.rectangle — vùng mặt phát hiện được.

        Returns:
            dlib.full_object_detection: Đối tượng chứa 68 điểm landmark.
        """
        shape = self._predictor(frame, face)
        return shape

    def shape_to_np(self, shape):
        """
        Chuyển dlib shape thành numpy array (68x2).

        Tối ưu: dùng vectorized numpy thay vì loop.

        Args:
            shape: dlib.full_object_detection — 68 điểm landmark.

        Returns:
            numpy.ndarray: Mảng shape (68, 2) chứa tọa độ (x, y) các điểm.
        """
        coords = np.array(
            [(shape.part(i).x, shape.part(i).y) for i in range(68)],
            dtype=np.int32
        )
        return coords

    def get_eye_landmarks(self, shape_np):
        """
        Trích xuất tọa độ landmark của 2 mắt từ shape numpy array.

        Args:
            shape_np: numpy.ndarray shape (68, 2) — tọa độ 68 điểm landmark.

        Returns:
            tuple: (left_eye, right_eye) — mỗi phần tử là numpy array shape (6, 2).
        """
        return shape_np[self.LEFT_EYE_IDX], shape_np[self.RIGHT_EYE_IDX]

    def get_mouth_landmarks(self, shape_np):
        """
        Trích xuất tọa độ landmark của miệng từ shape numpy array.

        Args:
            shape_np: numpy.ndarray shape (68, 2) — tọa độ 68 điểm landmark.

        Returns:
            numpy.ndarray: Mảng shape (20, 2) chứa tọa độ 20 điểm miệng (landmark 48-67).
        """
        return shape_np[self.MOUTH_IDX]

    def compute_EAR(self, eye):
        """
        Tính Eye Aspect Ratio (EAR) cho một mắt.

        Công thức: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Khi mắt mở: EAR cao (khoảng 0.25-0.35).
        Khi mắt nhắm: EAR thấp (gần 0).

        Args:
            eye: numpy.ndarray shape (6, 2) — tọa độ 6 điểm landmark của một mắt.

        Returns:
            float: Giá trị EAR (0.0 ~ 0.4).
        """
        vertical_1 = _euclidean(eye[1], eye[5])  # ||p2-p6||
        vertical_2 = _euclidean(eye[2], eye[4])  # ||p3-p5||
        horizontal = _euclidean(eye[0], eye[3])  # ||p1-p4||

        if horizontal < 1e-6:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def compute_avg_EAR(self, left_eye, right_eye):
        """
        Tính EAR trung bình của 2 mắt (có smoothing).

        Dùng moving average để giảm nhiễu, giúp EAR ổn định hơn.

        Args:
            left_eye: numpy.ndarray shape (6, 2) — landmark mắt trái.
            right_eye: numpy.ndarray shape (6, 2) — landmark mắt phải.

        Returns:
            float: EAR trung bình đã smooth.
        """
        left_ear = self.compute_EAR(left_eye)
        right_ear = self.compute_EAR(right_eye)
        raw_ear = (left_ear + right_ear) / 2.0

        if len(self._ear_buffer) == self._ear_buffer.maxlen:
            self._ear_sum -= self._ear_buffer[0]
        self._ear_buffer.append(raw_ear)
        self._ear_sum += raw_ear

        return self._ear_sum / len(self._ear_buffer)

    def compute_MAR(self, mouth):
        """
        Tính Mouth Aspect Ratio (MAR) cho miệng (có smoothing).

        Công thức: MAR = (||50-58|| + ||51-57||) / (2 * ||48-54||)

        Khi miệng đóng: MAR thấp.
        Khi ngáp (miệng mở to): MAR cao (> 0.6).

        Args:
            mouth: numpy.ndarray shape (20, 2) — tọa độ 20 điểm miệng (landmark 48-67).

        Returns:
            float: Giá trị MAR đã smooth.
        """
        vertical_1 = _euclidean(mouth[2], mouth[10])  # ||50-58||
        vertical_2 = _euclidean(mouth[3], mouth[9])   # ||51-57||
        horizontal = _euclidean(mouth[0], mouth[6])   # ||48-54||

        if horizontal < 1e-6:
            return 0.0

        raw_mar = (vertical_1 + vertical_2) / (2.0 * horizontal)

        if len(self._mar_buffer) == self._mar_buffer.maxlen:
            self._mar_sum -= self._mar_buffer[0]
        self._mar_buffer.append(raw_mar)
        self._mar_sum += raw_mar

        return self._mar_sum / len(self._mar_buffer)

    def reset_smoothing(self):
        """Reset bộ đệm smoothing (khi mất mặt)."""
        self._ear_buffer.clear()
        self._mar_buffer.clear()
        self._ear_sum = 0.0
        self._mar_sum = 0.0
