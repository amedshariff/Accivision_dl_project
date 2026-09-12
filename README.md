# 🚦 AI-Powered Real-Time Road Safety & Accident Detection System

A complete deep-learning-based accident detection pipeline:
**Input Road Video → OpenCV Preprocessing → YOLO Detection → ByteTrack Tracking → CNN-LSTM Classifier → Accident Verdict (Yes/No)**

---

## 🏛️ Pipeline Overview

```text
       ┌────────────────────────┐
       │   Road Video / CCTV    │
       │  (Accident or Normal)  │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ 1. OpenCV Preprocess   │  --> Uniform 16-frame sampling, resize (224x224), ImageNet normalization
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ 2. YOLO Detection      │  --> YOLOv11 nano: cars, motorcycles, buses, trucks, bicycles, pedestrians
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ 3. Object Tracking     │  --> ByteTrack IDs, trajectories, velocity in m/s, direction, motion alerts
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ 4. CNN-LSTM Classifier │  --> Pretrained ResNet18 (spatial) + 2-layer LSTM (temporal dynamics)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                  FINAL VERDICT OUTPUT                  │
       │  • ACCIDENT DETECTED (or NORMAL TRAFFIC)               │
       │  • Accident Probability Score (e.g. 97.86%)            │
       │  • Output Video with Real-Time HUD Gauge               │
       └────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure & Component Mapping

### 1. OpenCV Preprocessing
* `preprocessing/video_processor.py` — Ingests video, extracts FPS, dimensions, frame count; writes standardized MP4.
* `preprocessing/video_preprocessing.py` — Coordinates extraction, resizing, and normalization for DL input.
* `preprocessing/frame_extraction.py` — Uniform temporal sampling of 16 frames per video.
* `preprocessing/resize_frames.py` — Standardizes frames to 224x224.
* `preprocessing/normalize_frames.py` — ImageNet normalization and tensor stacking (1, 16, 3, 224, 224).
* `preprocessing/optical_flow.py` — Computes Farneback dense motion vectors.
* `preprocessing/utils.py` — Video metadata inspection helpers.

### 2. YOLO Object Detection
* `detection/yolo_detector.py` — Core YOLODetector class wrapping YOLOv11.
* `detection/weights/yolo11n.pt` — Pretrained YOLOv11 nano model weights.
* `detection/detection_config.py` — Confidence thresholds and target road classes.
* `detection/draw_bounding_boxes.py` — Renders bounding boxes and confidence labels.
* `detection/detect_objects.py` — Single-frame detection wrapper.
* `detection/run_detection.py` — Standalone detection CLI.

### 3. Object Tracking & Kinematics
* `tracking/tracker.py` — MultiObjectTracker class integrating ByteTrack frame association.
* `tracking/trajectory.py` — TrajectoryTracker maintaining rolling centroid history and motion trails.
* `tracking/motion_features.py` — Kinematics engine: velocity (m/s), direction, anomaly flags.
* `tracking/speed_estimation.py` — Pixel-to-meter speed estimation.
* `tracking/tracking_config.py` — Tracking parameters.
* `tracking/run_tracking.py` — Standalone tracking CLI.

### 4. CNN–LSTM Temporal Classifier
* `models/cnn_model.py` — Pretrained ResNet18 spatial feature extractor (outputs 512-dim embedding per frame).
* `models/lstm_model.py` — 2-layer LSTM temporal network with dropout and classification head.
* `models/cnn_lstm_model.py` — Composite end-to-end model (B, 16, 3, 224, 224) -> Probability.
* `models/saved_models/best_model.pth` — Trained model weights checkpoint for out-of-the-box inference.
* `training/dataset_loader.py` — PyTorch VideoSequenceDataset yielding video tensors and labels.
* `training/train_cnn_lstm.py` — Training script with Adam optimizer and validation tracking.
* `training/loss_functions.py` — Weighted BCE loss prioritizing accident recall.
* `inference/accident_predictor.py` — Prediction API returning verdict (Accident vs. Normal) and probability.
* `inference/video_inference.py` — Generates output video with HUD probability meter and status banner.
* `integration/accident_detection_pipeline.py` — Glues all 4 stages together into one continuous stream.

---

## 🚀 How to Run

### 1. Run Complete Pipeline on Any Video
```powershell
# Test on accident video:
python main.py --video data/raw/videos/accident/mock_accident.mp4

# Test on normal traffic video:
python main.py --video data/raw/videos/normal/real_road_test.mp4
```

### 2. Run Automated Tests
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Open Detailed Project PDF
The project includes a complete PDF guide explaining every single file and technology:
```powershell
start docs/Accident_Detection_System_Complete_Guide.pdf
```
