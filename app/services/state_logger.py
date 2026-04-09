import csv
import os
from datetime import datetime
from typing import Dict, Any, List


class StateLogger:
    def __init__(self, session_id: str = None):
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.session_id = session_id
        self.data_dir = "session_data"
        self.csv_path = os.path.join(self.data_dir, f"{session_id}.csv")

        os.makedirs(self.data_dir, exist_ok=True)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "frame_count",
                        "face_id",
                        "is_forward",
                        "direction_label",
                        "eye_state",
                        "yaw",
                        "pitch",
                        "pitch_ratio",
                    ]
                )

    def log(self, frame_count: int, face_statuses: List[Dict[str, Any]]):
        timestamp = datetime.now().isoformat()

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for face in face_statuses:
                writer.writerow(
                    [
                        timestamp,
                        frame_count,
                        face["id"],
                        1 if face["is_forward"] else 0,
                        face["label"],
                        face["eye_state"],
                        face["yaw"],
                        face["pitch"],
                        face["pitch_ratio"],
                    ]
                )

    def get_session_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.csv_path):
            return []

        data = []
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["frame_count"] = int(row["frame_count"])
                row["face_id"] = int(row["face_id"])
                row["is_forward"] = bool(int(row["is_forward"]))
                row["yaw"] = float(row["yaw"])
                row["pitch"] = float(row["pitch"])
                row["pitch_ratio"] = float(row["pitch_ratio"])
                data.append(row)

        return data

    def get_face_ids(self) -> List[int]:
        data = self.get_session_data()
        return sorted(set(row["face_id"] for row in data))

    @staticmethod
    def get_all_sessions() -> List[Dict[str, Any]]:
        data_dir = "session_data"
        if not os.path.exists(data_dir):
            return []

        sessions = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".csv"):
                session_id = filename.replace(".csv", "")
                file_path = os.path.join(data_dir, filename)
                mtime = os.path.getmtime(file_path)
                sessions.append(
                    {
                        "session_id": session_id,
                        "file": filename,
                        "modified": datetime.fromtimestamp(mtime).isoformat(),
                    }
                )

        return sorted(sessions, key=lambda x: x["modified"], reverse=True)
