"""
Generate Detailed Project PDF Guide
Generates docs/Accident_Detection_System_Complete_Guide.pdf
"""
import os
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))
        page_text = f"Page {self._pageNumber} of {page_count}"
        header_text = "AI-Powered Real-Time Road Safety & Accident Detection System"
        self.drawString(54, 750, header_text)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        
        self.line(54, 50, 558, 50)
        self.drawRightString(558, 38, page_text)
        self.drawString(54, 38, "Accivision Technical Architecture & Implementation Guide")
        self.restoreState()

def build_pdf(output_pdf_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=64,
        bottomMargin=64
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=16,
        spaceAfter=10,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=15,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#742A2A"),
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        'DocCallout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1A365D")
    )

    story = []

    # Title Block
    story.append(Spacer(1, 10))
    story.append(Paragraph("AI-Powered Real-Time Road Safety & Accident Detection System", title_style))
    story.append(Paragraph("<b>End-to-End Technical Guide & System Blueprint</b><br/>Architecture, Workflow, File Mapping, and Pretrained Model Usage", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2B6CB0"), spaceAfter=15))

    # Executive Overview
    story.append(Paragraph("1. System Architecture & End-to-End Pipeline", h1_style))
    story.append(Paragraph(
        "The Accident Detection System processes raw road and CCTV footage (both accident sequences and normal driving) "
        "through a 4-tier computer vision and deep learning pipeline. The architecture seamlessly combines real-time "
        "object detection and tracking with temporal sequence classification:",
        body_style
    ))

    # 4-stage pipeline summary box
    pipeline_data = [
        ["Stage", "Core Technology", "Primary Purpose", "Output Data"],
        ["1. OpenCV Preprocessing", "OpenCV (cv2)", "Frame ingestion, uniform sampling (16 frames), resize to 224x224, ImageNet normalization", "Normalized tensor (B, 16, 3, 224, 224)"],
        ["2. YOLO Detection", "YOLOv11 nano", "Detects vehicles (cars, motorcycles, buses, trucks), bicycles, and pedestrians", "Bounding boxes (x1, y1, x2, y2), confidences, classes"],
        ["3. Object Tracking", "ByteTrack + Kinematics", "Maintains persistent IDs across occlusion; computes displacement, direction, speed in m/s", "Track IDs, motion vectors, trajectories.csv"],
        ["4. CNN-LSTM Classifier", "ResNet18 + LSTM", "Extracts spatial feature embeddings and analyzes temporal dynamics over consecutive frames", "Accident verdict (Yes/No), probability %, video HUD"]
    ]
    t_pipeline = Table(pipeline_data, colWidths=[110, 110, 150, 134])
    t_pipeline.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('LEADING', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F7FAFC"), colors.white]),
    ]))
    story.append(t_pipeline)
    story.append(Spacer(1, 15))

    # Section 2: Detailed File Mapping
    story.append(Paragraph("2. File Mapping — Which Files Belong to What", h1_style))
    story.append(Paragraph(
        "The project is structured into clear functional domains. Below is the precise mapping of files to each pipeline stage:",
        body_style
    ))

    file_map_data = [
        ["Component", "File Path", "Specific Responsibility"],
        ["1. OpenCV Preprocessing", "preprocessing/video_processor.py", "Inspects FPS, dimensions, frame count; writes standardized MP4."],
        ["", "preprocessing/video_preprocessing.py", "Extracts, resizes, and normalizes sequences for DL model input."],
        ["", "preprocessing/frame_extraction.py", "Uniform temporal sampling (exactly 16 frames per video clip)."],
        ["", "preprocessing/resize_frames.py", "Standardizes all frames to 224x224 pixels."],
        ["", "preprocessing/normalize_frames.py", "Applies ImageNet mean/std normalization and tensor conversion."],
        ["", "preprocessing/optical_flow.py", "Computes Farneback dense motion energy between frames."],
        ["", "preprocessing/utils.py", "Helper functions for video metadata extraction."],
        ["2. YOLO Detection", "detection/yolo_detector.py", "Core YOLODetector class wrapping YOLOv11 for road objects."],
        ["", "detection/weights/yolo11n.pt", "Pretrained YOLOv11 nano model weights."],
        ["", "detection/detection_config.py", "Confidence thresholds (0.40) and target road classes."],
        ["", "detection/draw_bounding_boxes.py", "Draws labels, boxes, and confidence banners on video frames."],
        ["", "detection/detect_objects.py", "Convenience functional wrapper for single-frame detection."],
        ["", "detection/run_detection.py", "CLI execution script for Person 1 detection."],
        ["3. Object Tracking", "tracking/tracker.py", "MultiObjectTracker integrating ByteTrack frame association."],
        ["", "tracking/trajectory.py", "TrajectoryTracker managing rolling center points and trails."],
        ["", "tracking/motion_features.py", "Computes speed (m/s), direction (Up/Down/Left/Right), anomaly alerts."],
        ["", "tracking/speed_estimation.py", "Calibrated conversion from pixels/second to meters/second."],
        ["", "tracking/tracking_config.py", "Tracker configuration parameters (history length = 30)."],
        ["", "tracking/run_tracking.py", "CLI execution script for Person 2 tracking deliverables."],
        ["4. CNN-LSTM Classifier", "models/cnn_model.py", "Pretrained ResNet18 spatial feature extractor (512-dim output)."],
        ["", "models/lstm_model.py", "2-layer LSTM temporal network with dropout and classification head."],
        ["", "models/cnn_lstm_model.py", "Composite end-to-end model (B, 16, 3, 224, 224) -> Probability."],
        ["", "models/saved_models/best_model.pth", "Trained model checkpoint for immediate inference."],
        ["", "training/dataset_loader.py", "PyTorch VideoSequenceDataset yielding sequence tensors and labels."],
        ["", "training/train_cnn_lstm.py", "Training loop with Adam optimizer and validation tracking."],
        ["", "training/loss_functions.py", "Weighted BCE loss prioritizing accident recall."],
        ["", "inference/accident_predictor.py", "High-level prediction API returning accident verdict & confidence."],
        ["", "inference/video_inference.py", "Renders output video with HUD probability gauge & status bar."],
        ["Integration & Entry", "integration/accident_detection_pipeline.py", "Full end-to-end pipeline connecting Stages 1 -> 2 -> 3 -> 4."],
        ["", "main.py", "Top-level user CLI entrypoint (python main.py --video <path>)."]
    ]
    t_file_map = Table(file_map_data, colWidths=[120, 160, 224])
    t_file_map.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('LEADING', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F7FAFC"), colors.white]),
    ]))
    story.append(t_file_map)
    story.append(PageBreak())

    # Section 3: Deep Technical Breakdown
    story.append(Paragraph("3. Deep Technical Breakdown: What is Used & How It is Used", h1_style))

    # Stage 1 Breakdown
    story.append(Paragraph("Stage 1: OpenCV Preprocessing", h2_style))
    story.append(Paragraph(
        "<b>What is used:</b> OpenCV (<code>cv2.VideoCapture</code>, <code>cv2.resize</code>, <code>cv2.cvtColor</code>), "
        "NumPy contiguous arrays, and PyTorch tensor transformations.",
        body_style
    ))
    story.append(Paragraph(
        "<b>How it is used:</b> Raw road videos often have varying resolutions (1080p, 720p, 360p) and frame rates (10 to 30 FPS). "
        "The preprocessor inspects video properties and extracts a uniform temporal sample of exactly 16 frames spanning the clip duration. "
        "Each frame is resized to 224x224 pixels and transformed using ImageNet normalization (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]). "
        "This ensures consistent tensor dimensions <code>(1, 16, 3, 224, 224)</code> ready for deep neural network consumption.",
        body_style
    ))

    # Stage 2 Breakdown
    story.append(Paragraph("Stage 2: YOLO Road Object Detection", h2_style))
    story.append(Paragraph(
        "<b>What is used:</b> Pretrained YOLOv11 nano model weights (<code>detection/weights/yolo11n.pt</code>) powered by Ultralytics.",
        body_style
    ))
    story.append(Paragraph(
        "<b>How it is used:</b> Each frame is fed through the YOLO network with a confidence threshold of 0.40. "
        "The model is restricted specifically to road safety classes: <code>car</code>, <code>motorcycle</code>, <code>bus</code>, "
        "<code>truck</code>, <code>bicycle</code>, and <code>person</code>. Detections provide pixel bounding boxes (x1, y1, x2, y2), "
        "class names, and confidence probabilities. Irrelevant background objects (such as trees, signs, animals) are filtered out.",
        body_style
    ))

    # Stage 3 Breakdown
    story.append(Paragraph("Stage 3: Multi-Object Tracking & Motion Analysis", h2_style))
    story.append(Paragraph(
        "<b>What is used:</b> <b>ByteTrack</b> multi-object tracking algorithm, Euclidean centroid computation, "
        "and kinematic motion feature estimation.",
        body_style
    ))
    story.append(Paragraph(
        "<b>How it is used:</b> YOLO alone only detects objects per frame independently without remembering identities. "
        "ByteTrack assigns persistent numerical IDs (e.g. ID: 1, ID: 2) across consecutive frames, even during partial occlusions. "
        "For each tracked vehicle, the system computes: "
        "<br/>• <b>Centroid & Trails:</b> Centers (cx, cy) stored in a rolling 30-frame deque to draw visible trajectory lines."
        "<br/>• <b>Displacement:</b> Frame-to-frame pixel movement: Δd = √((x2 - x1)² + (y2 - y1)²)."
        "<br/>• <b>Speed:</b> Speed in m/s using actual video FPS: v = (Δd × FPS) / pixels_per_meter."
        "<br/>• <b>Direction:</b> Cardinal movement: 'Up', 'Down', 'Left', 'Right', or 'Stationary'."
        "<br/>• <b>Kinematic Alerts:</b> Flags sudden direction reversals, abnormal speed spikes, and rapid deceleration.",
        body_style
    ))

    # Stage 4 Breakdown
    story.append(Paragraph("Stage 4: CNN–LSTM Temporal Accident Classifier", h2_style))
    story.append(Paragraph(
        "<b>What is used:</b> Pretrained <code>ResNet18</code> CNN backbone + 2-layer <code>LSTM</code> recurrent network + "
        "Sigmoid classification head.",
        body_style
    ))
    story.append(Paragraph(
        "<b>How it is used:</b> An accident is a temporal event that unfolds over time (e.g., normal approach -> sudden collision/impact -> post-crash halt). "
        "Single-frame image models cannot distinguish an accident from a parked car. The hybrid CNN-LSTM solves this: "
        "<br/>1. <b>CNN Spatial Feature Extractor:</b> The 16 frames are passed through the frozen ResNet18 backbone. "
        "The final classification layer is removed, yielding a 512-dimensional feature vector per frame: <code>(B, 16, 512)</code>."
        "<br/>2. <b>LSTM Temporal Aggregator:</b> The sequence of 16 vectors is fed into the 2-layer LSTM (Hidden dim: 128). "
        "The LSTM learns how visual features change over time."
        "<br/>3. <b>Classification Head:</b> The final hidden representation passes through a Linear layer with Sigmoid activation, "
        "outputting the probability of an accident P ∈ [0.0, 1.0]. "
        "If P ≥ 0.50, the verdict is <b>ACCIDENT DETECTED</b>; otherwise <b>NORMAL TRAFFIC</b>.",
        body_style
    ))

    # Pretrained Model Explanation Callout
    story.append(Spacer(1, 10))
    callout_data = [[
        Paragraph(
            "<b>Why We Use a Pretrained Model (The Core Engineering Decision):</b><br/>"
            "• <b>CNN Spatial Backbone:</b> We use a <b>pretrained ResNet18</b> (trained on ImageNet). "
            "Training a deep CNN from scratch requires hundreds of thousands of road images and weeks of GPU compute. "
            "A pretrained CNN already understands low-level and high-level visual features (edges, contours, vehicle shapes, road textures). "
            "Freezing this backbone allows rapid feature extraction even on standard laptops without GPUs.<br/>"
            "• <b>LSTM Sequence Model:</b> While the CNN backbone is pretrained, the <b>LSTM head is trained specifically on accident vs. normal video sequences</b>. "
            "This gives you the best of both worlds: high-level visual features from ImageNet transfer learning combined with customized temporal accident detection.",
            callout_style
        )
    ]]
    t_callout = Table(callout_data, colWidths=[504])
    t_callout.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#3182CE")),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_callout)
    story.append(PageBreak())

    # Section 4: How to Run
    story.append(Paragraph("4. How to Run & User Guide", h1_style))
    story.append(Paragraph("Follow these exact steps to run the system:", body_style))

    story.append(Paragraph("<b>Step 1: Activate Virtual Environment</b>", h2_style))
    story.append(Paragraph("<code>.venv\\Scripts\\activate</code>", code_style))

    story.append(Paragraph("<b>Step 2: Run End-to-End Prediction on Any Video</b>", h2_style))
    story.append(Paragraph(
        "To test on an accident video and get the final answer immediately:<br/>"
        "<code>python main.py --video data/raw/videos/accident/mock_accident.mp4</code>",
        code_style
    ))
    story.append(Paragraph(
        "To test on a normal traffic video:<br/>"
        "<code>python main.py --video data/raw/videos/normal/real_road_test.mp4</code>",
        code_style
    ))

    story.append(Paragraph("<b>Sample Terminal Output:</b>", h2_style))
    sample_out = (
        "=================================================================<br/>"
        "  ROAD SAFETY AI — FULL PIPELINE (OPENCV -> YOLO -> TRACKING -> CNN-LSTM)<br/>"
        "=================================================================<br/>"
        "[Stage 1] OpenCV Preprocessing...<br/>"
        "  Input: data/raw/videos/accident/mock_accident.mp4<br/>"
        "  Properties: 640x360 @ 10.0 FPS | 60 frames (~6.0s)<br/>"
        "[Stage 2 & 3] YOLO Detection & Object Tracking...<br/>"
        "  Unique entities tracked: 2<br/>"
        "[Stage 4] CNN + LSTM Temporal Accident Classifier...<br/>"
        "=================================================================<br/>"
        "  FINAL ACCIDENT PREDICTION VERDICT<br/>"
        "=================================================================<br/>"
        "  VERDICT:              <b>ACCIDENT DETECTED</b><br/>"
        "  ACCIDENT PROBABILITY: <b>97.86%</b><br/>"
        "  IS ACCIDENT:          <b>True</b><br/>"
        "  OUTPUT VIDEO:         <b>outputs/predictions/predicted_mock_accident.mp4</b><br/>"
        "================================================================="
    )
    story.append(Paragraph(sample_out, code_style))
    story.append(Spacer(1, 10))

    # Section 5: Verification & Deliverables
    story.append(Paragraph("5. Deliverables & Output Verification", h1_style))
    story.append(Paragraph(
        "When the pipeline executes, the following deliverables are saved in the <code>outputs/</code> folder:",
        body_style
    ))

    deliv_data = [
        ["Output Folder", "Output File", "Description"],
        ["outputs/predictions/", "predicted_<video>.mp4", "Video displaying bounding boxes, tracking IDs, and real-time HUD accident probability meter."],
        ["outputs/tracked_videos/", "tracked_video.mp4", "Annotated video with persistent IDs, velocities in m/s, and historical motion trails."],
        ["outputs/tracked_videos/", "trajectories.csv", "Full tabular kinematics dataset (frame, timestamp, track ID, speed, direction, status)."],
        ["outputs/tracked_videos/", "tracking_report.txt", "Text summary report detailing unique vehicles, maximum speeds, and dominant directions."],
        ["models/saved_models/", "best_model.pth", "Trained PyTorch weights checkpoint for the CNN+LSTM accident classifier."]
    ]
    t_deliv = Table(deliv_data, colWidths=[120, 160, 224])
    t_deliv.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('LEADING', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F7FAFC"), colors.white]),
    ]))
    story.append(t_deliv)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {output_pdf_path}")

if __name__ == "__main__":
    out_path = "docs/Accident_Detection_System_Complete_Guide.pdf"
    if len(sys.argv) > 1:
        out_path = sys.argv[1]
    build_pdf(out_path)
