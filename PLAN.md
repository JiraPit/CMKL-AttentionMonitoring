# CMKL Attention Monitoring System

## Overview
A Flask-based attention monitoring website that processes video frames to detect faces, track them, extract facial landmarks, and estimate head direction in real-time.

## Modes
1. **Webcam Mode**: Live camera feed processing
2. **Upload Mode**: Upload and process video files

## Processing Pipeline

```
FULL FRAME → YOLO11n-face → Cropped Faces → ByteTrack → MediaPipe FaceMesh
                ↓                                              ↓
         Bounding boxes                                Key landmarks + Blendshapes
         (for overlay)                                  (nose, eyes, mouth)
                                                               ↓
                                                          Direction vector + Eye state
                                                          (yaw, pitch, roll)
                                                               ↓
                                      ┌─────────────────────────────────────┐
                                      │  FULL FRAME + Overlay              │
                                      │  • Green box = Forward            │
                                      │  • Red box = Not Forward          │
                                      │  • Eye state below box            │
                                      └─────────────────────────────────────┘
```

### Step 1: Face Detection (YOLOv11n)
- Model: `yolo11n-face.pt` (YOLOv11n face detection from HuggingFace)
- Input: Full video frame
- Output: List of face bounding boxes + cropped images

### Step 2: Face Tracking (ByteTrack)
- Algorithm: IOU-based matching
- Input: Face detections from step 1
- Output: Persistent face IDs per detection

### Step 3: Landmark Extraction (MediaPipe)
- Model: FaceMesh with blendshapes
- Input: Cropped face images (not full frame)
- Output: 468 3D facial landmarks + blendshapes

### Step 4: Direction Estimation
- Method: Ratio-based with eye landmarks
- Key Landmarks Used:
  | Landmark | Index | Purpose |
  |----------|-------|---------|
  | Nose tip | 1 | Primary reference |
  | Left eye | 33 | Left anchor |
  | Right eye | 263 | Right anchor |
  | Left mouth | 61 | Mouth anchor |
  | Right mouth | 291 | Mouth anchor |
- Eye State: Based on `eyeBlinkLeft` and `eyeBlinkRight` blendshapes
- Pitch Ratio: eye_to_nose / nose_to_mouth (forward: 0.2 - 3.0)

## Project Structure
```
CMKL-AttentionMonitoring/
├── pyproject.toml              # uv dependencies
├── yolo11n-face.pt             # YOLO face detection model
├── face_landmarker.task         # MediaPipe landmark model
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── routes.py                # API routes
│   ├── pipeline/
│   │   ├── detector.py          # YOLOv11 face detection
│   │   ├── tracker.py           # ByteTrack tracking
│   │   ├── landmarks.py         # MediaPipe FaceMesh + Blendshapes
│   │   └── direction.py         # Head pose & eye state estimation
│   ├── services/
│   │   ├── video_processor.py  # Frame processing pipeline
│   │   └── state_logger.py      # CSV logging for persistence
│   └── templates/
│       ├── dashboard.html        # Main dashboard
│       └── report.html          # Report page with graphs
├── static/
│   ├── css/style.css
│   └── js/app.js                # Client-side logic
└── session_data/                # CSV logs (auto-created)
    └── YYYYMMDD_HHMMSS.csv
```

## Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Attention Monitoring System          [View Reports]              │
│  Mode: (•) Webcam    ( ) Upload Video                            │
├─────────────────────────────────┬───────────────────────────────┤
│                                 │  Face Status                  │
│     Video Preview               │  ┌─────────────────────────┐ │
│     (full frame with boxes)     │  │ ID #1: Forward ✓        │ │
│                                 │  │ Eyes Open               │ │
│     ┌───┐  ┌───┐               │  │ ID #2: Looking Left ←    │ │
│     │ 1 │  │ 2 │               │  │ Eyes Closed               │ │
│     └───┘  └───┘               │  └─────────────────────────┘ │
│         ┌───┐                  │                               │
│         │ 3 │                  │  Summary                      │
│         └───┘                  │  - Total Faces: 3            │
├─────────────────────────────────┴───────────────────────────────┤
│  [Start Webcam]  [Upload & Process]  [Stop]                     │
└─────────────────────────────────────────────────────────────────┘
```

## Report Page

```
┌─────────────────────────────────────────────────────────────────┐
│  ← Back to Dashboard                                             │
│  Attention Monitoring Report                                      │
├─────────────────────────────────────────────────────────────────┤
│  [Select Session ▼]                                              │
│  [Face #1] [Face #2] [Face #3]                                  │
├─────────────────────────────────────────────────────────────────┤
│  │ Total Frames │ Avg Attention │ Avg Eyes Open │ Detections │ │
│  │     1250     │     67.3%    │     89.2%     │    1250   │ │
├─────────────────────────────────┬───────────────────────────────┤
│  Attention Over Time            │  Eyes State Over Time          │
│  ┌─────────────────────────┐   │  ┌─────────────────────────┐ │
│  │    📈 Graph             │   │  │    📈 Graph             │ │
│  │    (Forward/Not)        │   │  │    (Open/Closed)        │ │
│  └─────────────────────────┘   │  └─────────────────────────┘ │
└─────────────────────────────────┴───────────────────────────────┘
```

## API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/dashboard` | GET | Render main dashboard |
| `/report` | GET | Render report page with graphs |
| `/upload` | POST | Upload video file |
| `/video_feed` | GET | SSE stream of processed frames |
| `/webcam/start` | POST | Initialize webcam session |
| `/webcam/stop` | POST | Stop webcam session |
| `/sessions` | GET | List all saved sessions |
| `/session/<id>` | GET | Get session data for graphs |

## Technical Specifications
| Aspect | Value |
|--------|-------|
| Video size limit | 100MB max |
| Forward threshold | Yaw < 20°, Pitch ratio 0.2 - 3.0 |
| Eye threshold | Blink < 0.3 = Open |
| Webcam source | Client-side default webcam |
| Persistence | CSV files in session_data/ |

## Color Coding
| Color | Meaning |
|-------|---------|
| Green | Face facing forward / Eyes open |
| Red | Face not facing forward / Eyes closed |
| Yellow | Nose landmark |
| Cyan | Eye landmarks |
| Magenta | Mouth landmarks |
