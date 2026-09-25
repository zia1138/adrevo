"""Baseline constructor for the third autocorrelation inequality problem.

This entire file is evolvable. The trusted evaluator only depends on the JSON
output contract documented in the task prompt.
"""

import json
from pathlib import Path

import numpy as np


OUTPUT_FILE = Path("autocorrelation.json")


def construct_function() -> np.ndarray:
    """Return the original SimpleTES smooth, oscillatory starting profile."""
    n_points = 400
    x = np.linspace(-0.25, 0.25, n_points)

    heights = np.exp(-(x**2) / (2.0 * 0.1**2))
    heights += 0.3 * np.cos(2.0 * np.pi * 8.0 * x)
    heights += 0.15 * np.sin(2.0 * np.pi * 16.0 * x)
    heights *= 25.0 / np.sum(heights)
    return heights


def main() -> None:
    OUTPUT_FILE.write_text(
        json.dumps({"heights": construct_function().tolist()}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
