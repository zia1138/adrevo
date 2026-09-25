"""Trusted evaluator for the converted SimpleTES AC2 benchmark."""

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATE_DIR = Path("evo")
OUTPUT_FILE = CANDIDATE_DIR / "autocorrelation.json"
CANDIDATE_TIMEOUT_SECONDS = 1100


def evaluate_sequence(value: Any) -> float:
    """Validate and score a sampled non-negative function as SimpleTES did."""
    if not isinstance(value, list):
        raise ValueError("sequence must be a list")
    if not value:
        raise ValueError("sequence must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("sequence elements must be real numbers")

    sequence = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(sequence)):
        raise ValueError("sequence elements must be finite")

    sequence = np.clip(sequence, 0.0, 1000.0)
    if float(np.sum(sequence)) < 0.01:
        raise ValueError("sum of sequence is too close to zero")

    convolution = np.convolve(sequence, sequence)
    y = np.concatenate(([0.0], convolution, [0.0]))
    interval_width = 1.0 / (len(convolution) + 1)
    l2_squared = (interval_width / 3.0) * np.sum(
        y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2
    )
    l1 = np.sum(np.abs(convolution)) / (len(convolution) + 1)
    linfinity = np.max(np.abs(convolution))
    return float(l2_squared / (l1 * linfinity))


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
        score = evaluate_sequence(payload.get("sequence"))
        result = {
            "correct": True,
            "error": None,
            "c2": score,
            "combined_score": score,
        }
    except Exception as exc:
        result = {
            "correct": False,
            "error": f"{type(exc).__name__}: {exc}",
            "c2": 0.0,
            "combined_score": 0.0,
        }

    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
