"""Constructor-based circle packing for n=26 circles."""

import json
import random
from pathlib import Path

import numpy as np


OUTPUT_FILE = Path("circle_packing.json")


def construct_packing():
    """Construct a candidate arrangement of 26 circles in a unit square."""
    n = 26
    centers = np.zeros((n, 2))

    centers[0] = [0.5, 0.5]
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    centers = np.clip(centers, 0.01, 0.99)
    return centers, compute_max_radii(centers)


def compute_max_radii(centers):
    """Compute radii limited by the square boundary and other circles."""
    n = centers.shape[0]
    radii = np.ones(n)

    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


def write_packing(path: Path = OUTPUT_FILE) -> None:
    random.seed(42)
    centers, radii = construct_packing()
    path.write_text(
        json.dumps({"centers": centers.tolist(), "radii": radii.tolist()}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_packing()
