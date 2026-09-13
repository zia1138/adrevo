"""Evaluate matrix multiplication with each implementation run in Docker."""

# Adapted from Copyright (c) 2025 Ori Press and the AlgoTune contributors
# https://github.com/oripress/AlgoTune
import fcntl
import json
import random
import signal
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from baseline.main import solve


EXAMPLE_SIZES = (512, 1024, 2048)
EXAMPLE_SEEDS = (1, 2, 3)
WARMUP_RUNS = 1
TIMED_RUNS = 3
CACHE_DIR = Path.home() / ".cache" / "adrevo" / "datasets" / "matrix-multiplication"


RUNNER_IMAGE = "ghcr.io/astral-sh/uv:python3.13-trixie-slim"
CONTAINER_TERMINATION_GRACE_SEC = 10


class EvaluatorTerminated(SystemExit):
    """Raised to unwind the evaluator after Adrevo sends SIGTERM."""


def _handle_sigterm(_signum, _frame) -> None:
    raise EvaluatorTerminated(143)


def generate_problem(n: int, random_seed: int = 1):
    """Generate compatible random matrices using the original task's distribution."""
    rng = random.Random(random_seed)
    numpy_rng = np.random.RandomState(random_seed)
    m = rng.randint(n, 2 * n)
    p = rng.randint(n, 2 * n)
    return {"A": numpy_rng.randn(n, m), "B": numpy_rng.randn(m, p)}


def ensure_examples(n: int) -> Path:
    """Reuse node-local examples; clear the cache if generation changes."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (
        f"examples-n{n}-seeds-" + "-".join(map(str, EXAMPLE_SEEDS)) + ".npz"
    )
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


def validate_output(path: Path, problems: list[dict[str, np.ndarray]]) -> None:
    """Validate the candidate output against the trusted products."""
    with np.load(path, allow_pickle=False) as output:
        expected_keys = {f"C_{i}" for i in range(len(problems))}
        if set(output.files) != expected_keys:
            raise ValueError(f"{path.parent.name}: missing or unexpected output keys")
        for i, problem in enumerate(problems):
            expected = solve(problem)
            solution = np.asarray(output[f"C_{i}"])
            if not (
                solution.shape == expected.shape
                and np.isfinite(solution).all()
                and np.allclose(solution, expected, rtol=1e-5, atol=1e-8)
            ):
                raise ValueError(f"{path.parent.name}: incorrect solution for example {i}")


def fit_scaling(sizes, times) -> float:
    """Return the slope of log(runtime) against log(problem size)."""
    return float(np.polyfit(np.log(sizes), np.log(times), 1)[0])


def run_project(name: str, input_path: Path, output_path: Path) -> int:
    """Run one project in a fresh container and copy out its output artifact."""
    container = None
    output_path.unlink(missing_ok=True)
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        container = subprocess.check_output(
            [
                "docker",
                "create",
                "--cpus=1",
                RUNNER_IMAGE,
                "uv",
                "run",
                "-qq",
                "--directory",
                f"/{name}",
                "python",
                "main.py",
                "/input.npz",
            ],
            text=True,
        ).strip()
        subprocess.run(["docker", "cp", name, f"{container}:/"], check=True)
        subprocess.run(
            ["docker", "cp", input_path, f"{container}:/input.npz"], check=True
        )
        started_at = time.perf_counter_ns()
        subprocess.run(
            ["docker", "start", "-a", container],
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed_ns = time.perf_counter_ns() - started_at
        subprocess.run(
            ["docker", "cp", f"{container}:/{name}/output.npz", output_path],
            check=True,
        )
        return elapsed_ns
    finally:
        if container is not None:
            subprocess.run(
                [
                    "docker",
                    "stop",
                    "--time",
                    str(CONTAINER_TERMINATION_GRACE_SEC),
                    container,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["docker", "rm", "-f", container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


def main() -> None:
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
                names = (
                    ("baseline", "evo")
                    if (run + size_index) % 2 == 0
                    else ("evo", "baseline")
                )
                for name in names:
                    output_path = Path(name) / "output.npz"
                    elapsed_ns = run_project(name, input_path, output_path)
                    if run >= WARMUP_RUNS:
                        measurement[f"{name}_times_ns"].append(elapsed_ns)
                    validate_output(output_path, problems)
            for name in ("baseline", "evo"):
                measurement[f"{name}_time"] = (
                    statistics.median(measurement[f"{name}_times_ns"]) / 1e9
                )
        for name in ("baseline", "evo"):
            result[f"{name}_alpha"] = fit_scaling(
                EXAMPLE_SIZES,
                [row[f"{name}_time"] for row in result["per_size"]],
            )
        result["combined_score"] = result["baseline_alpha"] - result["evo_alpha"]
        result["correct"] = True
    except subprocess.CalledProcessError as exc:
        result["error"] = f"{name} at n={n}: {exc}\n{(exc.stderr or '')[-4000:]}"
    except Exception as exc:
        result["error"] = str(exc)
    Path("results.json").write_text(
        json.dumps(result, indent=4, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
