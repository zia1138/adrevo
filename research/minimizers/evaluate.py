"""Fixed build, validation, and scoring entry point for Adrevo.

Only evo/selection.cpp is candidate-owned. C++ translation units are compiled
separately and linked; no candidate build scripts or compiler flags are used.
This is a research harness, not a sandbox for hostile native code.
"""

import json
import math
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
COMPILER = "c++"
FLAGS = ("-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic")
BUILD_TIMEOUT = 30
RUN_TIMEOUT = 60
SEEDS = (42, 1729, 2026)  # Public development suite, not a hidden holdout.
LENGTH = 10_000
REPLICATES = 20
GC = (0.3, 0.5, 0.7)
K = 15
W = 19
METRIC_WINDOW = 250
MUTATION_RATE = 0.05
MAX_DENSITY = 0.11  # Checked per seed/GC group, separately on both sequences.


def run_command(command, cwd, timeout):
    completed = subprocess.run(
        [str(part) for part in command], cwd=cwd, timeout=timeout,
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{Path(str(command[0])).name} exited {completed.returncode}: "
            f"{completed.stderr[-8000:]}"
        )
    return completed.stdout


def build(project: Path, directory: Path) -> Path:
    """Always rebuild into an empty temporary directory; never reuse binaries."""
    objects = []
    for source, name in (("evaluate.cpp", "evaluate.o"),
                         ("evo/selection.cpp", "selection.o")):
        output = directory / name
        run_command(
            [COMPILER, *FLAGS, "-c", project / source, "-o", output],
            directory, BUILD_TIMEOUT,
        )
        objects.append(output)
    executable = directory / "evaluate"
    run_command([COMPILER, *objects, "-o", executable], directory, BUILD_TIMEOUT)
    return executable


def validate_metrics(metrics, sequences):
    """Validate integer counts before deriving scores in Python."""
    fields = ("sequences", "reference_kmers", "query_kmers",
              "reference_selected", "query_selected", "conserved_anchors",
              "windows", "hit_windows")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    for name in fields:
        value = metrics.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid count: {name}")
    if metrics["sequences"] != sequences:
        raise ValueError("incorrect sequence count")
    for side in ("reference", "query"):
        if metrics[f"{side}_kmers"] != sequences * (LENGTH - K + 1):
            raise ValueError("incorrect k-mer denominator")
        if metrics[f"{side}_selected"] > metrics[f"{side}_kmers"]:
            raise ValueError("selection count exceeds available k-mers")
    if metrics["conserved_anchors"] > min(
        metrics["reference_selected"], metrics["query_selected"]
    ):
        raise ValueError("anchor count exceeds selection count")
    if metrics["windows"] != sequences * (LENGTH - METRIC_WINDOW + 1):
        raise ValueError("incorrect window denominator")
    if metrics["hit_windows"] > metrics["windows"]:
        raise ValueError("hit count exceeds window count")
    return fields


def score_runs(runs):
    if len(runs) != len(SEEDS):
        raise ValueError("incomplete development suite")
    scores = []
    diagnostics = []
    for seed, run in zip(SEEDS, runs, strict=True):
        groups = run["groups"]
        if len(groups) != len(GC):
            raise ValueError("incorrect GC group count")
        fields = validate_metrics(run["overall"], REPLICATES * len(GC))
        for gc, group in zip(GC, groups, strict=True):
            if group["gc"] != gc:
                raise ValueError("unexpected GC group")
            m = group["metrics"]
            validate_metrics(m, REPLICATES)
            reference_density = m["reference_selected"] / m["reference_kmers"]
            query_density = m["query_selected"] / m["query_kmers"]
            if max(reference_density, query_density) > MAX_DENSITY:
                raise ValueError(
                    f"density budget exceeded: seed={seed}, GC={gc}, "
                    f"reference={reference_density:.6f}, query={query_density:.6f}, "
                    f"cap={MAX_DENSITY}"
                )
            p_hit = m["hit_windows"] / m["windows"]
            scores.append(p_hit)
            diagnostics.append({
                "seed": seed, "gc": gc, **m, "P_hit": p_hit,
                "density_reference": reference_density,
                "density_query": query_density,
            })
        for name in fields:
            if run["overall"][name] != sum(g["metrics"][name] for g in groups):
                raise ValueError(f"inconsistent aggregate: {name}")
    score = math.fsum(scores) / len(scores)
    if not math.isfinite(score):
        raise ValueError("non-finite score")
    return {"correct": True, "error": None, "combined_score": score,
            "max_density": MAX_DENSITY, "groups": diagnostics}


def evaluate(project: Path = ROOT):
    with tempfile.TemporaryDirectory(prefix="minimizers-") as temporary:
        directory = Path(temporary)
        executable = build(project, directory)
        runs = []
        for seed in SEEDS:
            output = run_command([
                executable, "--json", "--length", LENGTH,
                "--sequences-per-gc", REPLICATES,
                "--gc", ",".join(map(str, GC)), "--mutation-rate", MUTATION_RATE,
                "--k", K, "--w", W, "--window", METRIC_WINDOW, "--seed", seed,
            ], directory, RUN_TIMEOUT)
            runs.append(json.loads(output))
        return score_runs(runs)


def main():
    # Overwrite inherited results even if compilation or execution fails.
    output = ROOT / "results.json"
    output.unlink(missing_ok=True)
    try:
        result = evaluate()
    except Exception as exc:
        result = {"correct": False, "error": str(exc), "combined_score": 0.0}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
