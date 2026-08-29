"""Evaluator for the Heilbronn problem for convex regions (13 points)."""

import json
import itertools
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

#BENCHMARK = 0.030936889034895654
NUM_POINTS = 13
OUTPUT_FILE = Path("evo/heilbronn_convex_13.json")
def triangle_area(p1, p2, p3):
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2


if __name__ == "__main__":
    try:
        OUTPUT_FILE.unlink(missing_ok=True)
        subprocess.run(["uv", "run", "--directory", "evo", "python", "main.py"], check=True)
        points = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))["points"]
    except Exception as exc:
        Path("results.json").write_text(json.dumps({"correct": False, "error": str(exc), "combined_score": 0.0}, indent=4), encoding="utf-8")
        raise SystemExit

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
    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")
