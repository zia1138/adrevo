"""Baseline constructor for the Erdős minimum-overlap benchmark.

This entire file is evolvable. The trusted evaluator only depends on the JSON
output contract documented in the task prompt.
"""

import json
from pathlib import Path

import numpy as np


OUTPUT_FILE = Path("erdos.json")


def construct_h() -> np.ndarray:
    """Return a deterministic version of the original SimpleTES construction."""
    rng = np.random.default_rng(0)
    n_points = int(rng.integers(40, 100))
    heights = np.full(n_points, 0.5)
    perturbation = rng.uniform(-0.4, 0.4, n_points)
    heights += perturbation - np.mean(perturbation)
    return heights


def main() -> None:
    OUTPUT_FILE.write_text(
        json.dumps({"heights": construct_h().tolist()}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
