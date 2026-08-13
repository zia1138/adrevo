"""Evaluator for minimizing max/min distance ratio (dim=3, 14 points)."""

import json

import numpy as np
import scipy.spatial.distance
from main import min_max_dist_dim3_14

NUM_POINTS = 14
DIMENSION = 3
if __name__ == "__main__":
    points = min_max_dist_dim3_14()

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    error_msg = None
    is_valid = True

    if points.shape != (NUM_POINTS, DIMENSION):
        is_valid = False
        error_msg = f"Invalid shape: {points.shape}, expected ({NUM_POINTS}, {DIMENSION})"

    if is_valid:
        try:
            pairwise_distances = scipy.spatial.distance.pdist(points)
            min_distance = np.min(pairwise_distances)
            max_distance = np.max(pairwise_distances)
            inv_ratio_squared = (min_distance / max_distance) ** 2 if max_distance > 0 else 0
            combined_score = float(inv_ratio_squared)
        except Exception as e:
            is_valid = False
            error_msg = str(e)
            combined_score = 0.0
    else:
        combined_score = 0.0

    result = {
        "correct": is_valid,
        "error": error_msg,
        "combined_score": combined_score,
    }
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
