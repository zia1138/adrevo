"""Evaluator for the Heilbronn triangle problem (11 points in equilateral triangle)."""

import json
import itertools

import numpy as np
from main import heilbronn_triangle11

NUM_POINTS = 11
TOL = 1e-6


def triangle_area(a, b, c):
    return abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2


def check_inside_triangle(points, tol=1e-6):
    for x, y in points:
        cond1 = y >= -tol
        cond2 = np.sqrt(3) * x <= np.sqrt(3) - y + tol
        cond3 = y <= np.sqrt(3) * x + tol
        if not (cond1 and cond2 and cond3):
            return False, f"Point ({x}, {y}) is outside the equilateral triangle."
    return True, None


if __name__ == "__main__":
    points = heilbronn_triangle11()

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    error_msg = None
    is_valid = True

    if points.shape != (NUM_POINTS, 2):
        is_valid = False
        error_msg = f"Invalid shape: {points.shape}, expected ({NUM_POINTS}, 2)"
    else:
        is_valid, error_msg = check_inside_triangle(points, TOL)

    if is_valid:
        a = np.array([0, 0])
        b = np.array([1, 0])
        c = np.array([0.5, np.sqrt(3) / 2])
        min_triangle_area = min(
            triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)
        )
        combined_score = float(min_triangle_area) / triangle_area(a, b, c)
    else:
        combined_score = 0.0

    result = {
        "correct": is_valid,
        "error": error_msg,
        "combined_score": combined_score,
    }
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
