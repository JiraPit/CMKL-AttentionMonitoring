from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class FaceDetector:
    def __init__(self):
        self.model = YOLO("yolo11n-face.pt")
        self.model.conf = 0.3
        self.model.iou = 0.45
        self._next_id = 1
        self._tracks: Dict[int, Dict] = {}
        self._iou_threshold = 0.3

    def _compute_iou(
        self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def detect_with_tracking(
        self, frame: np.ndarray, frame_count: int
    ) -> List[Dict[str, Any]]:
        results = self.model(frame, verbose=False)

        current_boxes = []
        if results[0].boxes:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                current_boxes.append(tuple(xyxy))

        matched_ids = []
        used_boxes = set()

        for track_id, track_data in list(self._tracks.items()):
            best_iou = 0
            best_idx = -1

            for idx, box in enumerate(current_boxes):
                if idx in used_boxes:
                    continue
                iou = self._compute_iou(track_data["bbox"], box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= self._iou_threshold and best_idx >= 0:
                used_boxes.add(best_idx)
                self._tracks[track_id]["bbox"] = current_boxes[best_idx]
                self._tracks[track_id]["age"] = 0
                matched_ids.append(track_id)

        for idx, box in enumerate(current_boxes):
            if idx in used_boxes:
                continue

            new_id = self._next_id
            self._next_id += 1
            self._tracks[new_id] = {"bbox": box, "age": 0}
            matched_ids.append(new_id)

        for track_id in list(self._tracks.keys()):
            self._tracks[track_id]["age"] += 1
            if self._tracks[track_id]["age"] > 30:
                del self._tracks[track_id]

        detections = []
        h, w = frame.shape[:2]

        for track_id in matched_ids:
            x1, y1, x2, y2 = self._tracks[track_id]["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                cropped_face = frame[y1:y2, x1:x2]
                detections.append(
                    {
                        "track_id": track_id,
                        "bbox": (x1, y1, x2, y2),
                        "cropped_face": cropped_face,
                    }
                )

        return detections

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        results = self.model(frame, verbose=False)

        cropped_faces = []
        h, w = frame.shape[:2]

        if results[0].boxes:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    cropped_face = frame[y1:y2, x1:x2]
                    cropped_faces.append((cropped_face, (x1, y1, x2, y2)))

        return cropped_faces

    def reset(self):
        self._tracks = {}
        self._next_id = 1
