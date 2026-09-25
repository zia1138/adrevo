"""Baseline constructor for packing 32 circles in a unit square.

This entire file is evolvable.  The trusted evaluator only depends on the
output contract: running this file must write ``circle_packing.json`` with
``centers`` and ``radii`` arrays.
"""

import json
from pathlib import Path

import numpy as np


N = 32
OUTPUT_FILE = Path("circle_packing.json")


def construct_circles() -> np.ndarray:
    """Return an ``(N, 3)`` array whose rows are ``(x, y, radius)``."""
    centers = np.zeros((N, 2))
    grid_size = int(np.ceil(np.sqrt(N)))

    index = 0
    for row in range(grid_size):
        for column in range(grid_size):
            if index >= N:
                break
            centers[index] = [
                (row + 0.5) / grid_size,
                (column + 0.5) / grid_size,
            ]
            index += 1
        if index >= N:
            break

    centers = np.clip(centers, 0.01, 0.99)
    radii = compute_max_radii(centers)
    return np.column_stack([centers, radii])


def compute_max_radii(centers: np.ndarray) -> np.ndarray:
    """Compute feasible radii using boundary and pairwise constraints."""
    radii = np.ones(centers.shape[0])

    for index, (x, y) in enumerate(centers):
        radii[index] = min(x, y, 1 - x, 1 - y)

    for left in range(centers.shape[0]):
        for right in range(left + 1, centers.shape[0]):
            distance = np.linalg.norm(centers[left] - centers[right])
            radius_sum = radii[left] + radii[right]
            if radius_sum > distance:
                scale = distance / radius_sum * 0.99
                radii[left] *= scale
                radii[right] *= scale

    return radii


def main() -> None:
    circles = construct_circles()
    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "centers": circles[:, :2].tolist(),
                "radii": circles[:, 2].tolist(),
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
