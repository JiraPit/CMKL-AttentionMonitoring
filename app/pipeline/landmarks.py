import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple, Dict
import cv2
import os


MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def download_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request

        print(f"Downloading MediaPipe face landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


class LandmarkExtractor:
    def __init__(self):
        download_model()

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=10,
            output_face_blendshapes=True,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def extract(
        self, cropped_face: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_face)

        result = self.landmarker.detect(mp_image)

        if result.face_landmarks is None or len(result.face_landmarks) == 0:
            return None

        face_landmarks = result.face_landmarks[0]
        h, w = cropped_face.shape[:2]

        landmarks = np.zeros((len(face_landmarks), 3), dtype=np.float32)

        for idx, landmark in enumerate(face_landmarks):
            landmarks[idx] = [landmark.x * w, landmark.y * h, landmark.z * w]

        blendshapes = {}
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for blendshape in result.face_blendshapes[0]:
                blendshapes[blendshape.category_name] = blendshape.score

        return (landmarks, blendshapes)

    def get_key_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        key_indices = [1, 33, 263, 61, 291, 234]
        return landmarks[key_indices]

    def close(self):
        self.landmarker.close()
