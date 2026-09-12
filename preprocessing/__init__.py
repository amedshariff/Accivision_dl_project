"""
Preprocessing Package
Handles video loading, frame extraction, resizing, normalization, and optical flow.
"""
from preprocessing.video_processor import VideoProcessor
from preprocessing.video_preprocessing import VideoPreprocessor
from preprocessing.frame_extraction import extract_frames
from preprocessing.resize_frames import resize_frames, resize_frame
from preprocessing.normalize_frames import normalize_sequence, normalize_frame_tensor
from preprocessing.optical_flow import compute_sequence_motion_energy, compute_optical_flow_magnitude
from preprocessing.utils import get_video_metadata

__all__ = [
    "VideoProcessor",
    "VideoPreprocessor",
    "extract_frames",
    "resize_frames",
    "resize_frame",
    "normalize_sequence",
    "normalize_frame_tensor",
    "compute_sequence_motion_energy",
    "compute_optical_flow_magnitude",
    "get_video_metadata"
]
