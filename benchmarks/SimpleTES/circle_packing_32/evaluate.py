"""Trusted evaluator for the converted SimpleTES 32-circle benchmark."""

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


N = 32
CANDIDATE_DIR = Path("evo")
OUTPUT_FILE = CANDIDATE_DIR / "circle_packing.json"
CANDIDATE_TIMEOUT_SECONDS = 530
ATOL = 1e-12


def validate_packing(
    centers_value: Any,
    radii_value: Any,
) -> tuple[bool, str | None, np.ndarray | None]:
    """Validate untrusted candidate output and return normalized radii."""
    try:
        centers = np.asarray(centers_value, dtype=float)
        radii = np.asarray(radii_value, dtype=float)
    except (TypeError, ValueError) as exc:
        return False, f"Output is not numeric: {exc}", None

    if centers.shape != (N, 2):
        return False, f"Expected centers shape ({N}, 2), got {centers.shape}", None
    if radii.shape != (N,):
        return False, f"Expected radii shape ({N},), got {radii.shape}", None
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
        return False, "Centers and radii must contain only finite values", None
    if np.any(radii < 0):
        return False, "Radii must be non-negative", None

    boundary_clearance = np.minimum.reduce(
        (centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1])
    )
    outside = np.flatnonzero(radii > boundary_clearance + ATOL)
    if outside.size:
        index = int(outside[0])
        return False, f"Circle {index} lies outside the unit square", None

    for left in range(N):
        deltas = centers[left + 1 :] - centers[left]
        distances = np.linalg.norm(deltas, axis=1)
        required = radii[left] + radii[left + 1 :]
        overlaps = np.flatnonzero(distances < required - ATOL)
        if overlaps.size:
            right = left + 1 + int(overlaps[0])
            return False, f"Circles {left} and {right} overlap", None

    return True, None, radii


def run_candidate() -> dict[str, Any]:
    """Run the candidate in its own uv project and parse its output file."""
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
            raise ValueError("Candidate output must be a JSON object")
        valid, error, radii = validate_packing(
            payload.get("centers"),
            payload.get("radii"),
        )
        score = float(np.sum(radii)) if valid and radii is not None else 0.0
        result = {"correct": valid, "error": error, "combined_score": score}
    except Exception as exc:
        result = {
            "correct": False,
            "error": f"{type(exc).__name__}: {exc}",
            "combined_score": 0.0,
        }

    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
