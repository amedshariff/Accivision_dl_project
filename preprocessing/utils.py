"""
Utils Module (Preprocessing)
"""
import cv2
from typing import Dict, Any

def get_video_metadata(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = count / fps if fps > 0 else 0.0
    cap.release()
    return {
        "fps": round(fps, 2),
        "width": w,
        "height": h,
        "frame_count": count,
        "duration_sec": round(dur, 2)
    }
