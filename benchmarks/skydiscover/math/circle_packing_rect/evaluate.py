"""Evaluator for packing 21 circles in a rectangle of perimeter 4."""

import json
import subprocess
from pathlib import Path

import numpy as np


NUM_CIRCLES = 21
TOL = 1e-6
# If width + height <= 2, then 4r <= 2 for every contained circle.
MAX_RADIUS = 0.5
OUTPUT_FILE = Path("evo/circle_packing_rect.json")


def _as_float_array(candidate):
    """Return a validated float64 array, or an explanatory error."""
    if np.iscomplexobj(candidate):
        return None, "Packing must contain real numbers"

    try:
        circles = np.asarray(candidate, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        return None, f"Packing must be a numeric array: {exc}"

    if circles.shape != (NUM_CIRCLES, 3):
        return None, (
            f"Invalid shape: {circles.shape}, expected ({NUM_CIRCLES}, 3)"
        )

    if not np.all(np.isfinite(circles)):
        return None, "Packing contains NaN or infinite values"

    return circles, None


def minimum_circumscribing_rectangle(circles: np.ndarray):
    """Return the width and height of the circles' axis-aligned bounding box.

    Geometry is evaluated in extended precision so subtraction and addition
    cannot overflow for finite float64 submissions near the dtype limits.
    """
    work = circles.astype(np.longdouble, copy=False)
    radii = work[:, 2]
    width = np.max(work[:, 0] + radii) - np.min(work[:, 0] - radii)
    height = np.max(work[:, 1] + radii) - np.min(work[:, 1] - radii)
    return width, height


def validate_packing(candidate):
    circles, error = _as_float_array(candidate)
    if error is not None:
        return False, error

    work = circles.astype(np.longdouble, copy=False)
    centers = work[:, :2]
    radii = work[:, 2]
    tolerance = np.longdouble(TOL)

    negative = np.flatnonzero(radii < 0)
    if negative.size:
        i = int(negative[0])
        return False, f"Circle {i} has negative radius {float(radii[i])}"

    too_large = np.flatnonzero(radii > np.longdouble(MAX_RADIUS) + tolerance)
    if too_large.size:
        i = int(too_large[0])
        return False, f"Circle {i} has impossible radius {float(radii[i])}"

    for i in range(NUM_CIRCLES):
        for j in range(i + 1, NUM_CIRCLES):
            delta = centers[i] - centers[j]
            distance = np.hypot(delta[0], delta[1])
            required = radii[i] + radii[j]

            if not np.isfinite(distance) or not np.isfinite(required):
                return False, "Non-finite value produced by overlap calculation"

            if distance + tolerance < required:
                return False, (
                    f"Circles {i} and {j} overlap: dist={float(distance)}, "
                    f"r1+r2={float(required)}"
                )

    width, height = minimum_circumscribing_rectangle(circles)
    perimeter_sum = width + height
    if not np.all(np.isfinite([width, height, perimeter_sum])):
        return False, "Non-finite bounding rectangle"
    if width < 0 or height < 0 or perimeter_sum > 2 + tolerance:
        return False, "Circles not contained in rectangle of perimeter 4."

    score = np.sum(radii, dtype=np.longdouble)
    if not np.isfinite(score):
        return False, "Packing has a non-finite score"

    return True, None


def evaluate_candidate(candidate):
    """Validate a candidate and compute its score only after validation."""
    circles, error = _as_float_array(candidate)
    if error is not None:
        return False, error, 0.0

    is_valid, error = validate_packing(circles)
    if not is_valid:
        return False, error, 0.0

    score = float(np.sum(circles[:, 2], dtype=np.float64))
    if not np.isfinite(score):
        return False, "Packing has a non-finite score", 0.0

    return True, None, score


if __name__ == "__main__":
    try:
        OUTPUT_FILE.unlink(missing_ok=True)
        subprocess.run(["uv", "run", "--directory", "evo", "python", "main.py"], check=True)
        payload = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        is_valid, error_msg, combined_score = evaluate_candidate(payload["circles"])
    except Exception as exc:
        is_valid = False
        error_msg = f"Submission raised {type(exc).__name__}: {exc}"
        combined_score = 0.0

    result = {
        "correct": is_valid,
        "error": error_msg,
        "combined_score": combined_score,
    }
    # Refuse JavaScript's non-standard NaN and Infinity spellings.
    Path("results.json").write_text(json.dumps(result, indent=4, allow_nan=False), encoding="utf-8")
