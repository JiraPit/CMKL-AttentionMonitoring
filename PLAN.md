# CMKL Attention Monitoring System

## Overview
A Flask-based attention monitoring website that processes video frames to detect faces, track them, extract facial landmarks, and estimate head direction in real-time.

## Modes
1. **Webcam Mode**: Live camera feed processing
2. **Upload Mode**: Upload and process video files

## Processing Pipeline

```
FULL FRAME → YOLOv8s → Cropped Faces → ByteTrack → MediaPipe FaceMesh
                ↓                                        ↓
         Bounding boxes                          Key landmarks
         (for overlay)                            (nose, chin, eyes,
                                                    mouth corners)
                                                        ↓
                                                   Direction vector
                                                   (yaw, pitch, roll)
                                                        ↓
                           ┌─────────────────────────────────────┐
                           │  FULL FRAME + Overlay              │
                           │  • Green box = yaw/pitch < ±15°     │
                           │  • Red box = yaw/pitch >= ±15°     │
                           └─────────────────────────────────────┘
```

### Step 1: Face Detection (YOLOv8s)
- Model: `yolov8s-face.pt`
- Input: Full video frame
- Output: List of face bounding boxes + cropped face images

### Step 2: Face Tracking (ByteTrack)
- Algorithm: IOU-based matching with sort
- Input: Cropped face images
- Output: Persistent face IDs per detection

### Step 3: Landmark Extraction (MediaPipe)
- Model: FaceMesh
- Input: Cropped face images (not full frame)
- Output: 468 3D facial landmarks

### Step 4: Direction Estimation
- Method: PnP solvePnP with 6 key landmarks
- Key Landmarks Used:
  | Landmark | Index | Purpose |
  |----------|-------|---------|
  | Nose tip | 1 | Primary reference |
  | Chin | 234 | Bottom anchor |
  | Left eye | 33 | Left anchor |
  | Right eye | 263 | Right anchor |
  | Left mouth | 61 | Mouth anchor |
  | Right mouth | 291 | Mouth anchor |
- Output: yaw (left/right), pitch (up/down), roll (tilt)
- Forward threshold: `abs(yaw) < 15° AND abs(pitch) < 15°`

## Project Structure
```
CMKL-AttentionMonitoring/
├── pyproject.toml              # uv dependencies
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── routes.py                # /dashboard, /upload endpoints
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── detector.py          # YOLOv8s face detection
│   │   ├── tracker.py           # ByteTrack tracking
│   │   ├── landmarks.py         # MediaPipe FaceMesh on cropped
│   │   └── direction.py         # Head pose from face mesh landmarks
│   ├── services/
│   │   └── video_processor.py   # Orchestrates pipeline per frame
│   └── templates/
│       └── dashboard.html       # Main dashboard
├── static/
│   ├── css/style.css
│   └── js/app.js                # Webcam/upload handling + SSE
└── requirements.txt
```

## Dependencies (uv)
| Package | Purpose |
|---------|---------|
| flask | Web framework |
| ultralytics | YOLOv8s face detection |
| ByteTrack | Multi-face tracking |
| mediapipe | Face mesh landmarks |
| opencv-python | Image/video processing |
| numpy | Numerical operations |

## Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Mode: (•) Webcam    ( ) Upload Video                            │
├─────────────────────────────────┬───────────────────────────────┤
│                                 │  Face Status                  │
│     Video Preview               │  ┌─────────────────────────┐ │
│     (full frame with boxes)     │  │ ID #1: Forward ✓        │ │
│                                 │  │ ID #2: Looking Left ←    │ │
│     ┌───┐  ┌───┐               │  │ ID #3: Forward ✓        │ │
│     │ 1 │  │ 2 │               │  └─────────────────────────┘ │
│     └───┘  └───┘               │                               │
│         ┌───┐                  │  Summary                      │
│         │ 3 │                  │  - Total Faces: 3            │
│         └───┘                  │  - Attentive: 2 (67%)         │
├─────────────────────────────────┴───────────────────────────────┤
│  [Start Webcam]  [Upload & Process]  [Stop]                     │
└─────────────────────────────────────────────────────────────────┘
```

### Dashboard Features
- **Mode Toggle**: Switch between webcam (live) and upload (video file)
- **Video Preview**: Full frame with colored bounding boxes and face IDs
- **Status Panel**: Real-time list of detected faces showing:
  - Face ID (from ByteTrack)
  - Status: "Forward" (green) / "Not Forward" (red with direction indicator)
  - Yaw/Pitch values
- **Stats Summary**: Total faces, attentive count, attention percentage

## API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/dashboard` | GET | Render main dashboard |
| `/upload` | POST | Upload video file (max 100MB) |
| `/video_feed` | GET | SSE stream of processed frames |
| `/webcam/start` | POST | Initialize webcam session |
| `/webcam/stop` | POST | Stop webcam session |

## Technical Specifications
| Aspect | Value |
|--------|-------|
| Video size limit | 100MB max |
| Forward threshold | ±15° for both yaw and pitch |
| Webcam source | Client-side default webcam |
| Persistence | None (in-memory only) |

## Color Coding
| Color | Meaning |
|-------|---------|
| Green (#00FF00) | Face facing forward (within threshold) |
| Red (#FF0000) | Face not facing forward (outside threshold) |
