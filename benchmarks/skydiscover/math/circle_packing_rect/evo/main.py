import json
from pathlib import Path

import numpy as np


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4
    in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    circles = np.zeros((n, 3))

    return circles


if __name__ == "__main__":
    Path("circle_packing_rect.json").write_text(
        json.dumps({"circles": circle_packing21().tolist()}), encoding="utf-8"
    )
