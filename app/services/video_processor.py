import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from app.pipeline import (
    FaceDetector,
    FaceTracker,
    LandmarkExtractor,
    DirectionEstimator,
)

KEY_LANDMARKS = {
    "nose": 1,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

LANDMARK_COLORS = {
    "nose": (255, 255, 0),
    "left_eye": (0, 255, 255),
    "right_eye": (0, 255, 255),
    "left_mouth": (255, 0, 255),
    "right_mouth": (255, 0, 255),
}


class VideoProcessor:
    def __init__(self):
        self.detector = FaceDetector()
        self.tracker = FaceTracker()
        self.landmark_extractor = LandmarkExtractor()
        self.direction_estimator = DirectionEstimator(
            forward_threshold_yaw=20.0,
            forward_threshold_pitch_low=0.2,
            forward_threshold_pitch_high=3.0,
        )
        self.frame_count = 0

    def _draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ):
        x1, y1, x2, y2 = bbox

        for name, idx in KEY_LANDMARKS.items():
            if idx < len(landmarks):
                lx, ly, lz = landmarks[idx]

                frame_x = int(x1 + lx)
                frame_y = int(y1 + ly)

                color = LANDMARK_COLORS[name]
                cv2.circle(frame, (frame_x, frame_y), 4, color, -1)
                cv2.circle(frame, (frame_x, frame_y), 4, (255, 255, 255), 1)

                cv2.putText(
                    frame,
                    name.replace("_", " ").title(),
                    (frame_x + 6, frame_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                )

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        self.frame_count += 1

        detections = self.detector.detect_with_tracking(frame, self.frame_count)

        face_statuses = []

        for detection in detections:
            track_id = detection["track_id"]
            cropped_face = detection["cropped_face"]
            bbox = detection["bbox"]

            if track_id < 0 or cropped_face.size == 0:
                continue

            result = self.landmark_extractor.extract(cropped_face)

            if result is None:
                continue

            landmarks, blendshapes = result

            direction_result = self.direction_estimator.estimate(
                landmarks, cropped_face.shape[:2]
            )

            if direction_result is None:
                continue

            yaw, pitch, roll, pitch_ratio, is_forward = direction_result
            direction_label = self.direction_estimator.get_direction_label(
                yaw, pitch_ratio
            )
            eye_state = self.direction_estimator.get_eye_state(blendshapes)

            label = direction_label

            color = (0, 255, 0) if is_forward else (0, 0, 255)

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = f"ID {track_id}: {label}"
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            eye_text = eye_state
            eye_color = (0, 255, 255) if "Open" in eye_state else (0, 0, 255)
            cv2.putText(
                frame,
                eye_text,
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                eye_color,
                1,
            )

            self._draw_landmarks(frame, landmarks, bbox)

            face_statuses.append(
                {
                    "id": track_id,
                    "is_forward": is_forward,
                    "label": label,
                    "eye_state": eye_state,
                    "yaw": round(yaw, 1),
                    "pitch": round(pitch, 1),
                    "pitch_ratio": round(pitch_ratio, 2),
                    "roll": round(roll, 1),
                    "bbox": bbox,
                }
            )

        return frame, face_statuses

    def reset(self):
        self.tracker.reset()
        self.frame_count = 0

    def close(self):
        self.landmark_extractor.close()
