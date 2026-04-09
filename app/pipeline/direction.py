import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class DirectionEstimator:
    def __init__(
        self,
        forward_threshold_yaw: float = 20.0,
        forward_threshold_pitch_low: float = 0.2,
        forward_threshold_pitch_high: float = 3.0,
        eye_open_threshold: float = 0.3,
    ):
        self.forward_threshold_yaw = forward_threshold_yaw
        self.forward_threshold_pitch_low = forward_threshold_pitch_low
        self.forward_threshold_pitch_high = forward_threshold_pitch_high
        self.eye_open_threshold = eye_open_threshold

    def estimate(
        self, landmarks: np.ndarray, image_size: Tuple[int, int]
    ) -> Optional[Tuple[float, float, float, float, bool]]:
        if landmarks is None or len(landmarks) < 468:
            return None

        h, w = image_size

        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        eye_center_x = (left_eye[0] + right_eye[0]) / 2

        eye_distance = abs(right_eye[0] - left_eye[0])
        if eye_distance < 1:
            return None

        yaw = -((nose_tip[0] - eye_center_x) / eye_distance) * 90

        eye_to_nose = nose_tip[1] - left_eye[1]
        nose_to_mouth = ((left_mouth[1] + right_mouth[1]) / 2) - nose_tip[1]

        if nose_to_mouth <= 0 or eye_to_nose <= 0:
            return None

        pitch_ratio = eye_to_nose / nose_to_mouth

        yaw_deg = float(np.clip(yaw, -90, 90))
        pitch_deg = float((pitch_ratio - 1.0) * 90)
        roll_deg = 0.0

        is_forward = (
            abs(yaw_deg) < self.forward_threshold_yaw
            and self.forward_threshold_pitch_low
            <= pitch_ratio
            <= self.forward_threshold_pitch_high
        )

        return (yaw_deg, pitch_deg, roll_deg, pitch_ratio, is_forward)

    def get_eye_state(self, blendshapes: Dict[str, float]) -> str:
        eye_blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
        eye_blink_right = blendshapes.get("eyeBlinkRight", 0.0)

        avg_blink = (eye_blink_left + eye_blink_right) / 2.0

        if avg_blink < self.eye_open_threshold:
            return "Eyes Open"
        else:
            return "Eyes Closed"

    def get_direction_label(self, yaw: float, pitch_ratio: float) -> str:
        labels = []

        yaw_threshold = self.forward_threshold_yaw
        pitch_low = self.forward_threshold_pitch_low
        pitch_high = self.forward_threshold_pitch_high

        if abs(yaw) >= yaw_threshold:
            if yaw > 0:
                labels.append("Right")
            else:
                labels.append("Left")

        if pitch_ratio > pitch_high:
            labels.append("Down")
        elif pitch_ratio < pitch_low:
            labels.append("Up")

        if not labels:
            return "Forward"

        return " ".join(labels)
