"""Implement solve; preserve the NPZ input/output contract."""

import sys
from pathlib import Path

import numpy as np

def solve(problem):
    """Original NumPy baseline: C = A · B."""
    A = np.array(problem["A"])
    B = np.array(problem["B"])
    C = np.dot(A, B)
    return C

def main():
    """Read the NPZ at sys.argv[1] and write C_i products to local output.npz."""
    directory = Path(__file__).resolve().parent
    with np.load(sys.argv[1], allow_pickle=False) as examples:
        products = {
            f"C_{i}": solve({"A": examples[f"A_{i}"], "B": examples[f"B_{i}"]})
            for i in range(len(examples.files) // 2)
        }
    np.savez(directory / "output.npz", **products)


if __name__ == "__main__":
    main()
