"""Evaluator for circle packing (n=26 circles in a unit square)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

N = 26
OUTPUT_FILE = Path("evo/circle_packing.json")


def validate_packing(centers, radii, atol=1e-10):
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    if centers.shape != (N, 2):
        return False, f"Centers shape incorrect. Expected ({N}, 2), got {centers.shape}"
    if radii.shape != (N,):
        return False, f"Radii shape incorrect. Expected ({N},), got {radii.shape}"

    if np.any(np.isnan(centers)) or np.any(np.isnan(radii)):
        return False, "NaN values in output"

    if np.any(radii < 0):
        return False, f"Negative radii found at indices: {np.where(radii < 0)[0]}"

    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        if x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol:
            return False, f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) outside unit square."

    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                return False, (
                    f"Circles {i} & {j} overlap. Dist: {dist:.4f}, "
                    f"Sum Radii: {(radii[i] + radii[j]):.4f}"
                )

    return True, None


if __name__ == "__main__":
    try:
        OUTPUT_FILE.unlink(missing_ok=True)
        subprocess.run(
            ["uv", "run", "--directory", "evo", "python", "main.py"],
            check=True,
        )

        payload = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        is_valid, error_msg = validate_packing(
            payload["centers"], payload["radii"]
        )
        sum_radii = float(np.sum(payload["radii"])) if is_valid else 0.0
        result = {
            "correct": is_valid,
            "error": error_msg,
            "combined_score": sum_radii,
        }
    except Exception as exc:
        result = {
            "correct": False,
            "error": str(exc),
            "combined_score": 0.0,
        }

    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")
