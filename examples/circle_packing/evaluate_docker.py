"""Run the circle-packing candidate in Docker and validate its output locally."""

import json
import signal
import subprocess
from pathlib import Path

import numpy as np


RUNNER_IMAGE = "ghcr.io/astral-sh/uv:python3.13-trixie-slim"
CONTAINER_TERMINATION_GRACE_SEC = 10


class EvaluatorTerminated(SystemExit):
    """Raised to unwind the evaluator after Adrevo sends SIGTERM."""


def _handle_sigterm(_signum, _frame) -> None:
    raise EvaluatorTerminated(143)


def run_candidate() -> str:
    """Run the candidate in a fresh Docker container."""
    from evaluate import OUTPUT_FILE

    container = None
    OUTPUT_FILE.unlink(missing_ok=True)
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        container = subprocess.check_output(
            [
                "docker",
                "create",
                RUNNER_IMAGE,
                "uv",
                "run",
                "-qq",
                "--directory",
                "/evo",
                "python",
                "main.py",
            ],
            text=True,
        ).strip()
        subprocess.run(["docker", "cp", "evo", f"{container}:/"], check=True)
        subprocess.run(["docker", "start", "-a", container], check=True)
        subprocess.run(
            ["docker", "cp", f"{container}:/evo/circle_packing.json", OUTPUT_FILE],
            check=True,
        )
        return OUTPUT_FILE.read_text(encoding="utf-8")
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
    # This import stays in the trusted evaluator, outside the container.
    from evaluate import validate_packing

    try:
        output_text = run_candidate()
        payload = json.loads(output_text)
        is_valid, error_msg = validate_packing(
            payload["centers"], payload["radii"]
        )
        result = {
            "correct": is_valid,
            "error": error_msg,
            "combined_score": float(np.sum(payload["radii"])) if is_valid else 0.0,
        }
    except Exception as exc:
        result = {"correct": False, "error": str(exc), "combined_score": 0.0}

    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
