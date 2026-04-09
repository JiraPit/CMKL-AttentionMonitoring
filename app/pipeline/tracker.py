from typing import Dict, Any


class FaceTracker:
    def __init__(self):
        self.frame_count = 0

    def update(self, detections: list) -> list:
        self.frame_count += 1
        return detections

    def reset(self):
        self.frame_count = 0

    def get_active_tracks(self) -> list:
        return []
