# CMKL Attention Monitoring

Real-time attention monitoring system using computer vision to track face orientation and eye state. Detects if people are paying attention (facing forward with eyes open) from webcam feeds or uploaded videos.

## How It Works

The pipeline processes video frames through three stages:

1. Face Detection & Tracking - YOLO11n-face detects faces, with IoU (Intersection over Union) tracking to maintain consistent face IDs across frames
2. Landmark Extraction - MediaPipe extracts 468 facial landmarks + blendshapes
3. Attention Analysis - Head pose estimation (yaw/pitch) determines if face is forward-facing; eye blendshapes detect open/closed state

## Usage

```bash
# Install dependencies
uv sync

# Start the Flask server
uv run python run.py
```

Server runs at http://localhost:5000

Web Interface:

- /dashboard - Real-time monitoring with live video feed and face status
- /report - Historical data visualization with interactive charts

API:

- POST /upload - Upload video for processing
- POST /webcam/start / POST /webcam/stop - Control webcam processing
- GET /video_feed - Server-Sent Events video stream
- GET /sessions - List all logged sessions

## Results

### CSV Data

Session data is saved to `session_data/<timestamp>.csv`. Example:

| timestamp                  | frame_count | face_id | is_forward | direction_label | eye_state | yaw | pitch | pitch_ratio |
| -------------------------- | ----------- | ------- | ---------- | --------------- | --------- | --- | ----- | ----------- |
| 2026-04-27T14:30:22.123456 | 1           | 1       | 1          | Forward         | Eyes Open | 5.2 | 2.1   | 1.5         |

### Visualization

Access the Report page at /report to view interactive Chart.js graphs:

- Attention Over Time (forward/not-forward per frame)
- Eyes State Over Time (open/closed per frame)
- Filter by session and face ID
- Summary statistics (attention rate, eyes open rate)
