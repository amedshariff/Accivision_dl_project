"""
Trajectory Tracking Module (Person 2 - Level 3)
Maintains position history for tracked objects, tracks center points,
and computes frame-to-frame displacements and cumulative path distances.
"""

import math
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Optional


class TrajectoryTracker:
    """
    Maintains per-object historical positions and computes geometric trajectory metrics.
    """

    def __init__(self, history_length: int = 30):
        self.history_length = history_length
        # Maps track_id -> deque of (center_x, center_y)
        self.trajectories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_length))
        # Maps track_id -> total distance traveled in pixels
        self.total_distances: Dict[int, float] = defaultdict(float)

    def update(self, track_id: int, center: Tuple[int, int]) -> Tuple[float, float]:
        """
        Updates trajectory history for an object.
        Returns:
            displacement_pixels: Displacement from immediate previous frame
            total_distance_pixels: Cumulative distance traversed
        """
        history = self.trajectories[track_id]
        displacement = 0.0

        if len(history) > 0:
            prev_point = history[-1]
            dx = center[0] - prev_point[0]
            dy = center[1] - prev_point[1]
            displacement = math.hypot(dx, dy)
            self.total_distances[track_id] += displacement

        history.append(center)
        return round(displacement, 2), round(self.total_distances[track_id], 2)

    def get_trail(self, track_id: int) -> List[Tuple[int, int]]:
        """
        Returns list of (x, y) coordinates representing object's historical path.
        """
        return list(self.trajectories.get(track_id, []))

    def get_total_distance(self, track_id: int) -> float:
        return self.total_distances.get(track_id, 0.0)

    def reset(self):
        self.trajectories.clear()
        self.total_distances.clear()
