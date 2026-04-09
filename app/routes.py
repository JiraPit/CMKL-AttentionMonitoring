import cv2
import numpy as np
import base64
import json
import threading
from datetime import datetime
from flask import Blueprint, render_template, request, Response, stream_with_context
from app.services import VideoProcessor, StateLogger

bp = Blueprint("main", __name__)

processor = VideoProcessor()
state_logger = None
video_capture = None
video_lock = threading.Lock()
is_processing = False


@bp.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@bp.route("/report")
def report():
    return render_template("report.html")


@bp.route("/upload", methods=["POST"])
def upload_video():
    global video_capture, is_processing, state_logger

    if "video" not in request.files:
        return {"error": "No video file provided"}, 400

    video_file = request.files["video"]

    if video_file.filename == "":
        return {"error": "No video file selected"}, 400

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        video_file.save(tmp_file.name)
        tmp_path = tmp_file.name

    with video_lock:
        if video_capture is not None:
            video_capture.release()
        video_capture = cv2.VideoCapture(tmp_path)
        processor.reset()
        state_logger = StateLogger()
        is_processing = True

    os.unlink(tmp_path)

    return {"status": "Processing started", "session_id": state_logger.session_id}


@bp.route("/webcam/start", methods=["POST"])
def start_webcam():
    global video_capture, is_processing, state_logger

    with video_lock:
        if video_capture is not None:
            video_capture.release()
        video_capture = cv2.VideoCapture(0)
        processor.reset()
        state_logger = StateLogger()
        is_processing = True

    return {"status": "Webcam started", "session_id": state_logger.session_id}


@bp.route("/webcam/stop", methods=["POST"])
def stop_webcam():
    global video_capture, is_processing

    with video_lock:
        is_processing = False
        if video_capture is not None:
            video_capture.release()
            video_capture = None

    return {"status": "Stopped"}


@bp.route("/video_feed")
def video_feed():
    def generate():
        global video_capture, is_processing, state_logger

        while True:
            with video_lock:
                if video_capture is None or not is_processing:
                    continue

                ret, frame = video_capture.read()

                if not ret:
                    is_processing = False
                    break

            processed_frame, face_statuses = processor.process_frame(frame)

            if state_logger is not None:
                state_logger.log(processor.frame_count, face_statuses)

            _, buffer = cv2.imencode(
                ".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            total_faces = len(face_statuses)
            forward_faces = sum(1 for f in face_statuses if f["is_forward"])
            attention_rate = (
                (forward_faces / total_faces * 100) if total_faces > 0 else 0
            )

            serializable_statuses = []
            for f in face_statuses:
                serializable_statuses.append(
                    {
                        "id": int(f["id"]),
                        "is_forward": bool(f["is_forward"]),
                        "label": str(f["label"]),
                        "eye_state": str(f["eye_state"]),
                        "yaw": float(f["yaw"]),
                        "pitch": float(f["pitch"]),
                        "pitch_ratio": float(f["pitch_ratio"]),
                        "roll": float(f["roll"]),
                        "bbox": [int(x) for x in f["bbox"]],
                    }
                )

            data = {
                "frame": frame_base64,
                "face_statuses": serializable_statuses,
                "stats": {
                    "total_faces": int(total_faces),
                    "forward_faces": int(forward_faces),
                    "attention_rate": float(round(attention_rate, 1)),
                },
            }

            yield f"data: {json.dumps(data)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@bp.route("/sessions", methods=["GET"])
def get_sessions():
    sessions = StateLogger.get_all_sessions()
    return {"sessions": sessions}


@bp.route("/session/<session_id>", methods=["GET"])
def get_session(session_id):
    logger = StateLogger(session_id)
    data = logger.get_session_data()
    face_ids = logger.get_face_ids()
    return {"session_id": session_id, "face_ids": face_ids, "data": data}


@bp.route("/status")
def status():
    return {"is_processing": is_processing, "has_video": video_capture is not None}


@bp.route("/")
def index():
    return {
        "message": "CMKL Attention Monitoring API",
        "endpoints": ["/dashboard", "/report"],
    }
