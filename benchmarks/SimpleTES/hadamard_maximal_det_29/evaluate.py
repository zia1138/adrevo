"""Trusted evaluator for the converted SimpleTES order-29 determinant task."""

import json
import subprocess
from pathlib import Path
from typing import Any


MATRIX_SIZE = 29
REFERENCE_DETERMINANT = 1_270_698_346_568_170_340_352
CANDIDATE_DIR = Path("evo")
OUTPUT_FILE = CANDIDATE_DIR / "hadamard.json"
CANDIDATE_TIMEOUT_SECONDS = 350


def validate_matrix(value: Any) -> list[list[int]]:
    """Return a validated 29 by 29 matrix containing only integers ±1."""
    if not isinstance(value, list) or len(value) != MATRIX_SIZE:
        raise ValueError(f"matrix must have {MATRIX_SIZE} rows")

    matrix: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != MATRIX_SIZE:
            raise ValueError(f"row {row_index} must contain {MATRIX_SIZE} entries")
        normalized_row: list[int] = []
        for column_index, entry in enumerate(row):
            if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                raise ValueError(f"entry ({row_index}, {column_index}) is not numeric")
            if entry not in (-1, 1):
                raise ValueError(f"entry ({row_index}, {column_index}) is not +1 or -1")
            normalized_row.append(int(entry))
        matrix.append(normalized_row)
    return matrix


def determinant_bareiss(value: list[list[int]]) -> int:
    """Compute an integer determinant exactly using fraction-free elimination."""
    matrix = [row.copy() for row in value]
    size = len(matrix)
    sign = 1
    for pivot_index in range(size - 1):
        if matrix[pivot_index][pivot_index] == 0:
            for row_index in range(pivot_index + 1, size):
                if matrix[row_index][pivot_index] != 0:
                    matrix[pivot_index], matrix[row_index] = (
                        matrix[row_index],
                        matrix[pivot_index],
                    )
                    sign = -sign
                    break
            else:
                return 0

        pivot = matrix[pivot_index][pivot_index]
        denominator = matrix[pivot_index - 1][pivot_index - 1] if pivot_index else 1
        for row_index in range(pivot_index + 1, size):
            for column_index in range(pivot_index + 1, size):
                numerator = (
                    matrix[row_index][column_index] * pivot
                    - matrix[row_index][pivot_index]
                    * matrix[pivot_index][column_index]
                )
                matrix[row_index][column_index] = numerator // denominator
    return sign * matrix[-1][-1]


def evaluate_matrix(value: Any) -> tuple[int, float]:
    matrix = validate_matrix(value)
    absolute_determinant = abs(determinant_bareiss(matrix))
    ratio = absolute_determinant / REFERENCE_DETERMINANT
    return absolute_determinant, ratio


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
        absolute_determinant, ratio = evaluate_matrix(payload.get("matrix"))
        result = {
            "correct": True,
            "error": None,
            "abs_determinant": absolute_determinant,
            "determinant_ratio": ratio,
            "combined_score": ratio,
        }
    except Exception as exc:
        result = {
            "correct": False,
            "error": f"{type(exc).__name__}: {exc}",
            "abs_determinant": 0,
            "determinant_ratio": 0.0,
            "combined_score": 0.0,
        }

    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
