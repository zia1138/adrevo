"""Trusted evaluator for the converted SimpleTES Erdős overlap benchmark."""

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATE_DIR = Path("evo")
OUTPUT_FILE = CANDIDATE_DIR / "erdos.json"
CANDIDATE_TIMEOUT_SECONDS = 1100


def evaluate_heights(value: Any) -> tuple[float, int]:
    """Validate, normalize, and independently compute the C5 overlap."""
    if not isinstance(value, list):
        raise ValueError("heights must be a list")
    if not value:
        raise ValueError("heights must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("height values must be real numbers")

    heights = np.asarray(value, dtype=np.float64)
    if heights.ndim != 1:
        raise ValueError("heights must be one-dimensional")
    if not np.all(np.isfinite(heights)):
        raise ValueError("heights must be finite")
    if np.any(heights < 0.0) or np.any(heights > 1.0):
        raise ValueError("heights must lie in [0, 1]")

    n_points = heights.size
    target_sum = n_points / 2.0
    current_sum = float(np.sum(heights))
    if current_sum != target_sum:
        if current_sum <= 0.0:
            raise ValueError("height sum must be positive")
        heights = heights * (target_sum / current_sum)
        if np.any(heights < 0.0) or np.any(heights > 1.0):
            raise ValueError("normalized heights do not lie in [0, 1]")

    dx = 2.0 / n_points
    correlation = np.correlate(heights, 1.0 - heights, mode="full") * dx
    c5 = float(np.max(correlation))
    if not np.isfinite(c5):
        raise ValueError("computed C5 is not finite")
    return c5, n_points


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
        c5, n_points = evaluate_heights(payload.get("heights"))
        result = {
            "correct": True,
            "error": None,
            "c5": c5,
            "n_points": n_points,
            "combined_score": c5,
        }
    except Exception as exc:
        result = {
            "correct": False,
            "error": f"{type(exc).__name__}: {exc}",
            "c5": 1e300,
            "n_points": 0,
            "combined_score": 1e300,
        }

    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
