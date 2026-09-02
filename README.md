# 🚦 AI-Powered Real-Time Road Safety & Accident Detection System

A deep-learning-based road safety and accident detection pipeline built according to the 4-person team architecture contract.

> **Scope**: Person 1 (Video Preprocessing + YOLO Object Detection) → Person 2 (ByteTrack Tracking + Motion Features) → **STOP**  
> *(Person 3 & Person 4 are intentionally out-of-scope and not implemented)*

---

## 🏛️ Project Architecture

The implemented pipeline connects Person 1 and Person 2 into one continuous, robust stream:

```text
       ┌────────────────────────┐
       │   Road Video / CCTV    │
       │    (data/raw/*.mp4)    │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │        PERSON 1        │
       │  VideoProcessor (CV2)  │  --> Processed Video (outputs/detections/processed_road_test.mp4)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │        PERSON 1        │
       │  YOLODetector (YOLO11) │  --> Annotated Video & Detections (CSV & JSON)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │        PERSON 2        │
       │ MultiObjectTracker     │  --> ByteTrack ID Persistence & Trajectory Trails
       │ (ByteTrack + Motion)   │  --> Kinematic Speed & Direction Analysis
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │     FINAL OUTPUTS      │
       │   tracked_video.mp4    │
       │   trajectories.csv     │  --> Handoff to Person 3 (CNN+LSTM Sequence Buffer)
       │  motion_features.json  │
       │   tracking_report.txt  │
       └───────────┬────────────┘
                   │
              [STOP HERE]
  (Person 3 & Person 4 NOT IMPLEMENTED)
```

---

## 📁 Repository Structure

```text
dl_project/project/
│
├── data/
│   └── raw/                       # Raw input videos (mock_accident.mp4, road_test.mp4, etc.)
│
├── preprocessing/                 # Person 1: Level 1 Video Preprocessing
│   ├── __init__.py
│   └── video_processor.py         # VideoProcessor class (OpenCV reading, sizing, metadata)
│
├── detection/                     # Person 1: Level 2 YOLO Object Detection
│   ├── __init__.py
│   ├── config.py                  # Detection configuration (classes, confidence, paths)
│   ├── detection_schema.json      # Person 1 detection schema contract
│   ├── yolo_detector.py           # YOLODetector class (YOLO11n, road classes)
│   └── run_detection.py           # Person 1 standalone CLI runner
│
├── tracking/                      # Person 2: Level 3 Object Tracking & Motion
│   ├── __init__.py
│   ├── tracker.py                 # MultiObjectTracker (ByteTrack wrapper)
│   ├── trajectory.py              # TrajectoryTracker (historical centers & displacement)
│   ├── motion_features.py         # MotionFeatureExtractor (speed, direction, status, anomalies)
│   └── run_tracking.py            # Person 2 standalone CLI runner
│
├── models/                        # Pretrained models & checkpoints
│   └── yolo11n.pt                 # Pretrained YOLOv11 nano weights
│
├── outputs/                       # Generated deliverables
│   ├── detections/                # Person 1 outputs (processed video, annotated video, CSV, JSON)
│   └── tracking/                  # Person 2 outputs (tracked video, trajectories.csv, report)
│
├── tests/                         # Automated test suite
│   ├── __init__.py
│   ├── test_person_1.py           # Unit tests for VideoProcessor and YOLODetector
│   ├── test_person_2.py           # Unit tests for TrajectoryTracker, Motion, and ByteTrack
│   └── test_pipeline.py           # Integration tests (valid video, different video, missing video)
│
├── pipeline.py                    # End-to-end integration pipeline runner
├── main.py                        # Primary entrypoint
├── requirements.txt               # UTF-8 encoded project dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Activate Python 3.11 / 3.12 Virtual Environment**:
   ```powershell
   .venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### 1. Run Complete End-to-End Pipeline (Person 1 → Person 2 → STOP)
```powershell
python main.py --video data/raw/real_road_test.mp4
```
Optional flags:
- `--max-frames <N>`: Process only first N frames (useful for rapid testing).
- `--conf <threshold>`: Detection confidence threshold (default: 0.40).
- `--output-dir <path>`: Custom output folder.

### 2. Run Person 1 Standalone
```powershell
# Preprocessing only:
python preprocessing/video_processor.py --input data/raw/real_road_test.mp4 --output outputs/detections/processed_road_test.mp4

# YOLO Object Detection:
python detection/run_detection.py --input data/raw/real_road_test.mp4 --output-dir outputs/detections
```

### 3. Run Person 2 Standalone
```powershell
python tracking/run_tracking.py --input data/raw/real_road_test.mp4 --output-dir outputs/tracking
```

---

## 🧪 Running Automated Tests

Run the full test suite with verbose output:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

This validates:
- **Video preprocessing** (resolution, FPS, durations, error handling).
- **YOLO detections** (schema fields, road classes: `car`, `motorcycle`, `bus`, `truck`, `bicycle`, `person`).
- **Object tracking** (ByteTrack persistent IDs, center calculations, trajectory trails).
- **Motion features** (direction estimation, speed calculation, anomaly detection).
- **Pipeline integration** (Valid video test, Different video test, Missing video error handling).

---

## 🤝 Person 2 → Person 3 Handoff Contract

Person 2 finishes all tracking and kinematic feature extraction and stores the result at:
- **`outputs/tracking/trajectories.csv`**
- **`outputs/tracking/motion_features.json`**
- **`outputs/tracking/tracked_video.mp4`**

### Schema of `trajectories.csv`:
| Column | Type | Description |
|---|---|---|
| `frame_id` | `int` | 1-indexed video frame number |
| `timestamp_sec` | `float` | Timestamp in seconds calculated from video FPS |
| `track_id` | `int` | Persistent object identity across frames |
| `class` | `str` | Detected road class (`car`, `motorcycle`, `bus`, etc.) |
| `x1, y1, x2, y2` | `int` | Object bounding box pixel coordinates |
| `center_x, center_y` | `int` | Object centroid in pixels |
| `confidence` | `float` | YOLO confidence score |
| `displacement_px` | `float` | Euclidean pixel distance from previous frame |
| `direction` | `str` | Cardinal movement: `Up`, `Down`, `Left`, `Right`, `Stationary` |
| `speed_mps` | `float` | Estimated speed in meters per second |
| `status` | `str` | `Moving` or `Stationary` |
| `alerts` | `str` | Kinematic warning flags (e.g. `HIGH_SPEED`, `DIRECTION_REVERSAL`) |

### How Person 3 Can Use This:
1. **Sequence Buffer (Level 5/6)**: Person 3 can ingest `trajectories.csv` alongside video frames to sample sequences of 16/32 frames per clip.
2. **Kinematic Context**: Person 3's temporal model (CNN+LSTM) can use the bounding boxes and trajectory dynamics to focus feature extraction on high-speed or interacting vehicles.
3. **Person 3 & 4 Status**: Intentionally not implemented. Person 3 can build on top of these stable IDs and motion features.
