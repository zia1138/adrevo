"""Baseline constructor for the order-29 maximal determinant benchmark.

This entire file is evolvable. The trusted evaluator only depends on the JSON
output contract documented in the task prompt.
"""

import json
import random
from pathlib import Path

import numpy as np


MATRIX_SIZE = 29
OUTPUT_FILE = Path("hadamard.json")


def determinant_bareiss(value: list[list[int]]) -> int:
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


def structured_start() -> list[list[int]]:
    quadratic_residues = {1, 4, 5, 6, 7, 9, 13, 16, 20, 22, 23, 24, 25, 28}
    return [
        [1 if (row - column) % MATRIX_SIZE in quadratic_residues else -1
         for column in range(MATRIX_SIZE)]
        for row in range(MATRIX_SIZE)
    ]


def construct_matrix() -> list[list[int]]:
    """Apply the original deterministic simulated-annealing search."""
    rng = random.Random(42)
    current = structured_start()
    current_determinant = determinant_bareiss(current)
    best = [row.copy() for row in current]
    best_determinant = current_determinant

    for iteration in range(1, 2001):
        row = rng.randrange(MATRIX_SIZE)
        column = rng.randrange(MATRIX_SIZE)
        current[row][column] *= -1
        candidate_determinant = determinant_bareiss(current)

        accept = abs(candidate_determinant) >= abs(current_determinant)
        if not accept:
            temperature = 0.5 / (1.0 + iteration * 0.001)
            relative_change = (
                abs(candidate_determinant) - abs(current_determinant)
            ) / max(1.0, temperature * abs(current_determinant))
            accept = rng.random() < np.exp(relative_change)

        if accept:
            current_determinant = candidate_determinant
            if abs(current_determinant) > abs(best_determinant):
                best_determinant = current_determinant
                best = [matrix_row.copy() for matrix_row in current]
        else:
            current[row][column] *= -1

    return best


def main() -> None:
    OUTPUT_FILE.write_text(
        json.dumps({"matrix": construct_matrix()}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
