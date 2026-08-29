"""Evaluator for minimizing max/min distance ratio (dim=3, 14 points)."""

import json
import subprocess
from pathlib import Path

import numpy as np
import scipy.spatial.distance

NUM_POINTS = 14
DIMENSION = 3
OUTPUT_FILE = Path("evo/minimizing_max_min_dist_3.json")
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
    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")
