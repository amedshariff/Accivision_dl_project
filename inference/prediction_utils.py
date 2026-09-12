"""
Prediction Utils Module
"""
from typing import List, Dict, Any

def smooth_probabilities(probabilities: List[float], window_size: int = 3) -> List[float]:
    """Applies moving average filter to temporal probability scores."""
    if len(probabilities) < window_size:
        return probabilities
    smoothed = []
    for i in range(len(probabilities)):
        start = max(0, i - window_size + 1)
        smoothed.append(float(sum(probabilities[start:i+1]) / len(probabilities[start:i+1])))
    return smoothed
