"""
generate_alarm.py — Script tạo file alarm.wav mẫu.

Tạo âm thanh cảnh báo dạng sine wave (beep) 
để sử dụng trong hệ thống phát hiện ngủ gật.
Chỉ cần chạy 1 lần trước khi chạy main.py.
"""

import os
import numpy as np
from scipy.io.wavfile import write


def generate_alarm(output_path="sounds/alarm.wav"):
    """
    Tạo file alarm.wav dạng sine wave beep cảnh báo.

    Tạo chuỗi beep ngắt quãng (0.3s bật, 0.2s tắt) 
    trong 3 giây, tần số 880Hz (nốt A5 — âm thanh cảnh báo rõ ràng).

    Args:
        output_path: Đường dẫn file wav đầu ra (mặc định: sounds/alarm.wav).
    """
    # Thông số âm thanh
    sample_rate = 44100     # Tần số lấy mẫu (Hz)
    duration = 3.0          # Tổng thời gian (giây)
    frequency = 880.0       # Tần số sine wave (Hz) — nốt A5
    beep_on = 0.3           # Thời gian beep bật (giây)
    beep_off = 0.2          # Thời gian beep tắt (giây)

    # Tạo mảng thời gian
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Tạo sine wave
    signal = np.sin(2 * np.pi * frequency * t)

    # Tạo envelope bật/tắt (beep pattern)
    envelope = np.zeros_like(t)
    period = beep_on + beep_off
    for i, time_val in enumerate(t):
        # Vị trí trong chu kỳ hiện tại
        pos_in_period = time_val % period
        if pos_in_period < beep_on:
            envelope[i] = 1.0

    # Nhân signal với envelope
    signal = signal * envelope

    # Fade in/out để tránh tiếng xé (click noise)
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    for i in range(fade_samples):
        factor = i / fade_samples
        signal[i] *= factor
        signal[-(i + 1)] *= factor

    # Chuẩn hóa về khoảng [-1, 1] và chuyển sang int16
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    signal_int16 = np.int16(signal * 32767)

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ghi file wav
    write(output_path, sample_rate, signal_int16)
    print(f"Da tao file am thanh tai: {output_path}")


if __name__ == "__main__":
    generate_alarm()
