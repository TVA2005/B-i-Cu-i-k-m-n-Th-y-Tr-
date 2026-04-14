"""
core — Package chính của hệ thống phát hiện ngủ gật tài xế.

Chứa các module: config, detector, alert, utils.
"""

from core.config import *
from core.detector import FaceDetector
from core.alert import AlertManager
from core.utils import (
    draw_landmarks,
    draw_eye_hull,
    draw_ear_bar,
    draw_hud,
    draw_face_box,
    draw_mouth_hull,
)
