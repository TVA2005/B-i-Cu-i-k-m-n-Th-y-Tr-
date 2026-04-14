"""
main.py — Entry point, vòng lặp webcam chính.

Khởi tạo camera, detector, alert manager và chạy vòng lặp
phát hiện ngủ gật/ngáp theo thời gian thực.

Tối ưu:
- Face tracking giữa các frame để ổn định
- Smoothing EAR/MAR để giảm nhiễu
- Dashboard HUD đẹp với panel và gradient bar
"""

import time
import cv2

from core.config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    EAR_THRESH, MAR_THRESH,
    EAR_CONSEC_FRAMES, YAWN_CONSEC_FRAMES,
    REDETECT_INTERVAL,
)
from core.detector import FaceDetector
from core.alert import AlertManager
from core.utils import (
    draw_landmarks, draw_eye_hull, draw_mouth_hull,
    draw_ear_bar, draw_hud, draw_face_box
)


class DrowsinessApp:
    """Ứng dụng phát hiện ngủ gật tài xế với tracking và dashboard đẹp."""

    def __init__(self):
        """
        Khởi tạo ứng dụng.

        Raises:
            FileNotFoundError: Nếu model Dlib không tồn tại.
            RuntimeError: Nếu không mở được camera.
        """
        # Khởi tạo detector
        self._detector = FaceDetector()
        print("[OK] FaceDetector đã sẵn sàng")

        # Khởi tạo alert manager
        self._alert_mgr = AlertManager()
        print("[OK] AlertManager đã sẵn sàng")

        # Mở camera
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Không thể mở camera tại chỉ số {CAMERA_INDEX}.\n"
                f"Kiểm tra camera đã được kết nối chưa, "
                f"hoặc thử thay CAMERA_INDEX trong core/config.py."
            )

        # Đặt kích thước frame
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"[OK] Camera đã mở: {FRAME_WIDTH}x{FRAME_HEIGHT}")

        # Bộ đếm frame liên tiếp
        self._ear_counter = 0
        self._yawn_counter = 0

        # Trạng thái hiện tại
        self._status = "NORMAL"

        # Biến tính FPS
        self._fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._global_frame_idx = 0

    def _update_fps(self):
        """Cập nhật FPS mỗi giây."""
        self._frame_count += 1
        elapsed = time.time() - self._fps_start_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = time.time()

    def _reset_counters(self, ear, mar, has_valid_face):
        """
        Cập nhật bộ đếm frame.

        Args:
            ear: float — giá trị EAR.
            mar: float — giá trị MAR.
            has_valid_face: bool — có detect được mặt hợp lệ hay không.
        """
        if not has_valid_face:
            # Mất mặt (quay đi/che mặt) thì không suy luận buồn ngủ để tránh cảnh báo giả.
            self._ear_counter = 0
            self._yawn_counter = 0
            return

        if ear < EAR_THRESH:
            self._ear_counter += 1
        else:
            self._ear_counter = 0

        if mar > MAR_THRESH:
            self._yawn_counter += 1
        else:
            self._yawn_counter = 0

    def _check_alerts(self):
        """Kiểm tra và kích hoạt cảnh báo."""
        if self._ear_counter >= EAR_CONSEC_FRAMES:
            self._status = "DROWSY"
            self._alert_mgr.trigger_drowsy_alert()
            return "drowsy"

        if self._yawn_counter >= YAWN_CONSEC_FRAMES:
            self._status = "YAWNING"
            self._alert_mgr.trigger_yawn_alert()
            return "yawn"

        self._status = "NORMAL"
        self._alert_mgr.stop_alert()
        return None

    def _get_face(self, gray, color_frame, force_detect=False):
        """
        Lấy khuôn mặt: dùng tracking nếu có, hoặc detect mới.

        Args:
            gray: Ảnh grayscale để detect.
            color_frame: Ảnh BGR để track.

        Returns:
            dlib.rectangle hoặc None: Vùng mặt.
        """
        tracked = None

        # Ưu tiên tracking (nhanh và ổn định)
        if self._detector.is_tracking:
            tracked = self._detector.update_tracking(color_frame)
            if tracked is not None:
                if not force_detect:
                    return tracked

        # Nếu đến chu kỳ hoặc mất track, detect lại để căn chỉnh box mặt.
        faces = self._detector.detect(gray)

        if len(faces) > 0:
            # Lấy mặt lớn nhất
            face = max(faces, key=lambda f: f.width() * f.height())
            # Bắt đầu tracking
            self._detector.start_tracking(color_frame, face)
            return face

        # Detector có thể hụt khi quay nghiêng; nếu tracker vẫn giữ được mặt thì dùng tạm.
        if tracked is not None:
            return tracked

        # Không phát hiện mặt → reset tracking và smoothing
        self._detector.stop_tracking()
        self._detector.reset_smoothing()
        return None

    def _process_frame(self, frame):
        """
        Xử lý một frame: detect/tracking, tính EAR/MAR, vẽ HUD.

        Args:
            frame: Ảnh BGR từ camera.

        Returns:
            Frame đã xử lý.
        """
        # Resize về kích thước chuẩn
        h, w = frame.shape[:2]
        if w != FRAME_WIDTH:
            scale = FRAME_WIDTH / w
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))

        # Chuyển sang grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._global_frame_idx += 1

        # Giá trị mặc định
        ear = 0.0
        mar = 0.0
        has_valid_face = False

        # Lấy khuôn mặt (tracking hoặc detect)
        force_detect = (self._global_frame_idx % REDETECT_INTERVAL == 0)
        face = self._get_face(gray, frame, force_detect=force_detect)

        if face is not None:
            # Lấy 68 điểm landmark
            shape = self._detector.get_shape(gray, face)
            shape_np = self._detector.shape_to_np(shape)
            has_valid_face = True

            # Vẽ landmarks và hull
            draw_landmarks(frame, shape_np)

            # Lấy landmarks
            left_eye, right_eye = self._detector.get_eye_landmarks(shape_np)
            mouth = self._detector.get_mouth_landmarks(shape_np)

            # Vẽ convex hull mắt và miệng
            draw_eye_hull(frame, left_eye)
            draw_eye_hull(frame, right_eye)
            draw_mouth_hull(frame, mouth)

            # Vẽ bounding box mặt
            draw_face_box(frame, face, label="Driver")

            # Tính EAR và MAR (đã có smoothing)
            ear = self._detector.compute_avg_EAR(left_eye, right_eye)
            mar = self._detector.compute_MAR(mouth)

        # Cập nhật counter và kiểm tra cảnh báo
        self._reset_counters(ear, mar, has_valid_face)
        alert_type = self._check_alerts()

        # Vẽ dashboard HUD
        draw_hud(frame, ear, mar, self._status, self._fps,
                 self._ear_counter, self._yawn_counter)

        # Vẽ thanh bar EAR
        draw_ear_bar(frame, ear)

        # Vẽ overlay cảnh báo
        if alert_type is not None:
            self._alert_mgr.draw_alert_overlay(frame, alert_type)

        return frame

    def run(self):
        """Chạy vòng lặp chính."""
        print("\n=== HỆ THỐNG CẢNH BÁO NGỦ GẬT ===")
        print("Nhấn [Q] để thoát\n")
        print("Đang khởi động...")

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    print("[!] Lỗi: Không thể đọc frame từ camera!")
                    break

                self._update_fps()
                processed_frame = self._process_frame(frame)

                cv2.imshow("Driver Drowsiness Detection", processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n[!] Đã nhận Ctrl+C, đang thoát...")

        finally:
            self._cleanup()

    def _cleanup(self):
        """Giải phóng tài nguyên."""
        print("[*] Đang dọn dẹp...")
        self._alert_mgr.cleanup()
        self._cap.release()
        cv2.destroyAllWindows()
        print("[OK] Đã thoát chương trình.")


if __name__ == "__main__":
    try:
        app = DrowsinessApp()
        app.run()
    except FileNotFoundError as e:
        print(f"[!] LỖI: {e}")
    except RuntimeError as e:
        print(f"[!] LỖI: {e}")
    except Exception as e:
        print(f"[!] LỖI KHÔNG XÁC ĐỊNH: {e}")
