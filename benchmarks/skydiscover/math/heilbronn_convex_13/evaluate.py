"""Evaluator for the Heilbronn problem for convex regions (13 points)."""

import json
import itertools

import numpy as np
from scipy.spatial import ConvexHull
from main import heilbronn_convex13

#BENCHMARK = 0.030936889034895654
NUM_POINTS = 13
def triangle_area(p1, p2, p3):
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2


if __name__ == "__main__":
    points = heilbronn_convex13()

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    error_msg = None
    is_valid = True

    if points.shape != (NUM_POINTS, 2):
        is_valid = False
        error_msg = f"Invalid shape: {points.shape}, expected ({NUM_POINTS}, 2)"

    if is_valid:
        try:
            min_triangle_area = min(
                triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)
            )
            convex_hull_area = ConvexHull(points).volume
            min_area_normalized = min_triangle_area / convex_hull_area
            #combined_score = float(min_area_normalized / BENCHMARK)
            combined_score = float(min_area_normalized)
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
