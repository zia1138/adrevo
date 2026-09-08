"""Trusted evaluator. Score = baseline wall time / candidate wall time.

Both programs receive a cached input NPZ path and write local output.npz. Keys are
A_0, B_0, A_1, B_1, ...; output keys must be C_0, C_1, ... .
After warmup, timings include uv startup, imports, computation and I/O.
Score uses median wall times from repeated runs; reported times are seconds.
"""

# Adapted from Copyright (c) 2025 Ori Press and the AlgoTune contributors
# https://github.com/oripress/AlgoTune
import fcntl
import json
import random
import statistics
import subprocess
import time
import tempfile
from pathlib import Path

import numpy as np

from baseline.main import solve

EXAMPLE_SIZES = (1, 8, 64, 256, 512)
WARMUP_RUNS = 1
TIMED_RUNS = 5
RUN_TIMEOUT = 60
CACHE_DIR = (
    Path.home() / ".cache"
    / "adrevo" / "datasets" / "matrix-multiplication"
)


def generate_problem(n: int, random_seed: int = 1):
    """Generate compatible random matrices using the original task's distribution."""
    rng = random.Random(random_seed)
    numpy_rng = np.random.RandomState(random_seed)
    m = rng.randint(n, 2 * n)
    p = rng.randint(n, 2 * n)
    return {"A": numpy_rng.randn(n, m), "B": numpy_rng.randn(m, p)}


def ensure_examples():
    """Reuse node-local examples; clear the cache if generation changes."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / ("examples-" + "-".join(map(str, EXAMPLE_SIZES)) + ".npz")
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.is_file():
            arrays = {
                f"{key}_{i}": matrix
                for i, n in enumerate(EXAMPLE_SIZES)
                for key, matrix in generate_problem(n, random_seed=i + 1).items()
            }
            with tempfile.TemporaryDirectory(dir=CACHE_DIR) as staging:
                temporary_path = Path(staging) / "examples.npz"
                np.savez(temporary_path, **arrays)
                temporary_path.replace(path)
    return path


def is_solution(problem, solution) -> bool:
    """Check keys, dimensions, finite values and numerical correctness."""
    try:
        A = np.asarray(problem["A"])
        B = np.asarray(problem["B"])
        C = np.asarray(solution)
        if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
            return False
        if A.shape[1] != B.shape[0] or C.shape != (A.shape[0], B.shape[1]):
            return False
        if not np.isfinite(C).all():
            return False
        return bool(np.allclose(C, solve(problem), rtol=1e-5, atol=1e-8))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def validate_output(path, problems):
    with np.load(path, allow_pickle=False) as output:
        if set(output.files) != {f"C_{i}" for i in range(len(problems))}:
            raise ValueError(f"{path.parent.name}: missing or unexpected output keys")
        for i, problem in enumerate(problems):
            if not is_solution(problem, output[f"C_{i}"]):
                raise ValueError(f"{path.parent.name}: incorrect solution for example {i}")


def main():
    result = {
        "correct": False,
        "error": None,
        "combined_score": 0.0,
        "baseline_time": None,
        "evo_time": None,
        "baseline_times_ns": [],
        "evo_times_ns": [],
    }
    try:
        input_path = ensure_examples()
        with np.load(input_path, allow_pickle=False) as examples:
            problems = [
                {"A": examples[f"A_{i}"], "B": examples[f"B_{i}"]}
                for i in range(len(EXAMPLE_SIZES))
            ]
        for run in range(WARMUP_RUNS + TIMED_RUNS):
            # Alternate order to reduce systematic timing bias.
            names = ("baseline", "evo") if run % 2 == 0 else ("evo", "baseline")
            for name in names:
                output_path = Path(name) / "output.npz"
                output_path.unlink(missing_ok=True)
                start = time.perf_counter_ns()
                try:
                    subprocess.run(
                        ["uv", "run", "-qq", "python", "main.py", str(input_path)],
                        cwd=name,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=RUN_TIMEOUT,
                    )
                finally:
                    elapsed_ns = time.perf_counter_ns() - start
                    if run >= WARMUP_RUNS:
                        samples = result[f"{name}_times_ns"]
                        samples.append(elapsed_ns)
                        result[f"{name}_time"] = statistics.median(samples) / 1e9
                validate_output(output_path, problems)
        result["combined_score"] = result["baseline_time"] / result["evo_time"]
        result["correct"] = True
    except subprocess.CalledProcessError as exc:
        result["error"] = f"{name}: {exc}\n{exc.stderr[-4000:]}"
    except Exception as exc:
        result["error"] = str(exc)
    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
