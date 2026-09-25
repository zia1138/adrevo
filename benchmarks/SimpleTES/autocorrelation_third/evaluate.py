"""Trusted evaluator for the converted SimpleTES AC3 benchmark."""

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATE_DIR = Path("evo")
OUTPUT_FILE = CANDIDATE_DIR / "autocorrelation.json"
CANDIDATE_TIMEOUT_SECONDS = 70
MAX_ABSOLUTE_HEIGHT = 1e10
MIN_INTEGRAL_SQUARED = 1e-9


def evaluate_heights(value: Any) -> tuple[float, int]:
    """Validate candidate heights and independently compute the AC3 value."""
    if not isinstance(value, list):
        raise ValueError("heights must be a list")
    if not value:
        raise ValueError("heights must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("height values must be real numbers")

    heights = np.asarray(value, dtype=float)
    if heights.ndim != 1:
        raise ValueError("heights must be one-dimensional")
    if not np.all(np.isfinite(heights)):
        raise ValueError("heights must be finite")
    if float(np.max(np.abs(heights))) > MAX_ABSOLUTE_HEIGHT:
        raise ValueError(f"absolute height values must not exceed {MAX_ABSOLUTE_HEIGHT}")

    n_points = heights.size
    dx = 0.5 / n_points
    integral_squared = (float(np.sum(heights)) * dx) ** 2
    if integral_squared < MIN_INTEGRAL_SQUARED:
        raise ValueError("function integral is too close to zero")

    convolution = np.convolve(heights, heights, mode="full") * dx
    c3 = float(np.max(np.abs(convolution)) / integral_squared)
    if not np.isfinite(c3):
        raise ValueError("computed C3 is not finite")
    return c3, n_points


def run_candidate() -> dict[str, Any]:
    """Run the candidate in its own uv project and read its JSON output."""
    OUTPUT_FILE.unlink(missing_ok=True)
    subprocess.run(
        ["uv", "run", "-qq", "--directory", str(CANDIDATE_DIR), "python", "main.py"],
        check=True,
        timeout=CANDIDATE_TIMEOUT_SECONDS,
    )
    return json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))


def main() -> None:
    try:
        payload = run_candidate()
        if not isinstance(payload, dict):
            raise ValueError("candidate output must be a JSON object")
        c3, n_points = evaluate_heights(payload.get("heights"))
        result = {
            "correct": True,
            "error": None,
            "c3": c3,
            "n_points": n_points,
            "combined_score": c3,
        }
    except Exception as exc:
        result = {
            "correct": False,
            "error": f"{type(exc).__name__}: {exc}",
            "c3": 1e300,
            "n_points": 0,
            "combined_score": 1e300,
        }

    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
