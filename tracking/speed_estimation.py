"""
Speed Estimation Module
"""
def estimate_speed(displacement_px: float, fps: float, pixels_per_meter: float = 80.0) -> float:
    """Calculates speed in meters per second."""
    if pixels_per_meter <= 0 or fps <= 0:
        return 0.0
    speed_px_per_sec = displacement_px * fps
    return round(speed_px_per_sec / pixels_per_meter, 2)
