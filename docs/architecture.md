# Architecture Overview

## 4-Stage Pipeline Flow
1. **OpenCV Preprocessing**: Frame reading, uniform temporal sampling (16 frames), resize to 224x224, ImageNet normalization.
2. **YOLO Detection**: YOLOv11 nano detecting road-safety classes (car, motorcycle, bus, truck, bicycle, person).
3. **Object Tracking**: ByteTrack maintaining persistent identities across frames and extracting kinematics (speed, direction, displacement).
4. **CNN-LSTM Classifier**: ResNet18 spatial feature extractor + 2-layer LSTM temporal network classifying sequences into Accident vs. Normal traffic.
