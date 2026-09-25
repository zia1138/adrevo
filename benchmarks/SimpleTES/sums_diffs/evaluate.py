"""Trusted evaluator for the converted SimpleTES sums/differences benchmark."""

import json
import math
import subprocess
from pathlib import Path
from typing import Any


CANDIDATE_DIR = Path("evo")
OUTPUT_FILE = CANDIDATE_DIR / "sums_diffs.json"
CANDIDATE_TIMEOUT_SECONDS = 180
MIN_SET_SIZE = 2
MAX_SET_SIZE = 512
MIN_INTEGER = -1_000_000
MAX_INTEGER = 1_000_000
INTEGER_TOLERANCE = 1e-9


def normalize_values(value: Any) -> list[int]:
    """Validate, round within tolerance, deduplicate, and sort A."""
    if not isinstance(value, list):
        raise ValueError("values must be a list")

    integers: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"element {index} is not numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"element {index} is not finite")
        rounded = round(number)
        if abs(number - rounded) > INTEGER_TOLERANCE:
            raise ValueError(f"element {index} is not an integer")
        integer = int(rounded)
        if not MIN_INTEGER <= integer <= MAX_INTEGER:
            raise ValueError(f"element {index} is outside the allowed range")
        integers.append(integer)

    values = sorted(set(integers))
    if not MIN_SET_SIZE <= len(values) <= MAX_SET_SIZE:
        raise ValueError(
            f"set must contain {MIN_SET_SIZE} to {MAX_SET_SIZE} distinct integers"
        )
    return values


def evaluate_values(value: Any) -> dict[str, int | float]:
    """Independently construct A+A and A-A and compute C(A)."""
    values = normalize_values(value)
    set_size = len(values)
    sumset_size = len({left + right for left in values for right in values})
    diffset_size = len({left - right for left in values for right in values})
    sum_ratio = sumset_size / set_size
    diff_ratio = diffset_size / set_size
    if sum_ratio <= 1.0 or diff_ratio <= 1.0:
        raise ValueError("sumset and difference-set ratios must both exceed 1")

    score = math.log(sum_ratio) / math.log(diff_ratio)
    if not math.isfinite(score):
        raise ValueError("computed score is not finite")
    return {
        "set_size": set_size,
        "sumset_size": sumset_size,
        "diffset_size": diffset_size,
        "sum_ratio": sum_ratio,
        "diff_ratio": diff_ratio,
        "c_value": score,
    }


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
        stats = evaluate_values(payload.get("values"))
        result = {
            "correct": True,
            "error": None,
            **stats,
            "combined_score": stats["c_value"],
        }
    except Exception as exc:
        result = {
            "correct": False,
            "error": f"{type(exc).__name__}: {exc}",
            "c_value": 0.0,
            "set_size": 0,
            "sumset_size": 0,
            "diffset_size": 0,
            "combined_score": 0.0,
        }

    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
