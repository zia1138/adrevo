"""Score empirical scaling: baseline log-log slope minus candidate slope.

Each NPZ contains the same number of seeded problems at one size n. Input keys
are A_i/B_i and output keys C_i. Timings cover the entire subprocess, including
startup and I/O; slopes describe the measured size range, not asymptotic bounds.
Per-size median batch times are reported in seconds, with raw nanosecond samples.
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

EXAMPLE_SIZES = (128, 256, 512, 1024, 2048)
EXAMPLE_SEEDS = (1, 2, 3)
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


def ensure_examples(n):
    """Reuse node-local examples; clear the cache if generation changes."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (f"examples-n{n}-seeds-" + "-".join(map(str, EXAMPLE_SEEDS)) + ".npz")
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.is_file():
            arrays = {
                f"{key}_{i}": matrix
                for i, seed in enumerate(EXAMPLE_SEEDS)
                for key, matrix in generate_problem(n, random_seed=seed).items()
            }
            with tempfile.TemporaryDirectory(dir=CACHE_DIR) as staging:
                temporary_path = Path(staging) / "examples.npz"
                np.savez(temporary_path, **arrays)
                temporary_path.replace(path)
    return path


def is_solution(problem, solution) -> bool:
    """Validate the candidate output against the trusted product."""
    expected = solve(problem)
    solution = np.asarray(solution)
    return bool(
        solution.shape == expected.shape
        and np.isfinite(solution).all()
        and np.allclose(solution, expected, rtol=1e-5, atol=1e-8)
    )


def validate_output(path, problems):
    with np.load(path, allow_pickle=False) as output:
        if set(output.files) != {f"C_{i}" for i in range(len(problems))}:
            raise ValueError(f"{path.parent.name}: missing or unexpected output keys")
        for i, problem in enumerate(problems):
            if not is_solution(problem, output[f"C_{i}"]):
                raise ValueError(f"{path.parent.name}: incorrect solution for example {i}")


def fit_scaling(sizes, times):
    """Return the slope of log(runtime) against log(problem size)."""
    return float(np.polyfit(np.log(sizes), np.log(times), 1)[0])


def main():
    result = {
        "correct": False,
        "error": None,
        "combined_score": 0.0,
        "baseline_alpha": None,
        "evo_alpha": None,
        "per_size": [],
    }
    try:
        for size_index, n in enumerate(EXAMPLE_SIZES):
            input_path = ensure_examples(n)
            with np.load(input_path, allow_pickle=False) as examples:
                problems = [
                    {"A": examples[f"A_{i}"], "B": examples[f"B_{i}"]}
                    for i in range(len(EXAMPLE_SEEDS))
                ]
            measurement = {
                "n": n,
                "baseline_time": None,
                "evo_time": None,
                "baseline_times_ns": [],
                "evo_times_ns": [],
            }
            result["per_size"].append(measurement)
            for run in range(WARMUP_RUNS + TIMED_RUNS):
                # Alternate order across repetitions and sizes.
                names = ("baseline", "evo") if (run + size_index) % 2 == 0 else ("evo", "baseline")
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
                            measurement[f"{name}_times_ns"].append(elapsed_ns)
                    validate_output(output_path, problems)
            for name in ("baseline", "evo"):
                measurement[f"{name}_time"] = statistics.median(measurement[f"{name}_times_ns"]) / 1e9
        for name in ("baseline", "evo"):
            result[f"{name}_alpha"] = fit_scaling(
                EXAMPLE_SIZES, [row[f"{name}_time"] for row in result["per_size"]]
            )
        # Keep the sign: negative is valid but indicates worse empirical scaling.
        result["combined_score"] = result["baseline_alpha"] - result["evo_alpha"]
        result["correct"] = True
    except subprocess.CalledProcessError as exc:
        result["error"] = f"{name} at n={n}: {exc}\n{exc.stderr[-4000:]}"
    except Exception as exc:
        result["error"] = str(exc)
    Path("results.json").write_text(json.dumps(result, indent=4, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
