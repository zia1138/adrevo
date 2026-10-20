"""Run the circle-packing candidate on Modal and validate its output locally."""

import json
import signal
import subprocess
from pathlib import Path

import modal
import numpy as np
import typer


N = 26

cli = typer.Typer()


@cli.callback()
def main() -> None:
    """Evaluate circle-packing implementations on Modal."""


class EvaluatorTerminated(SystemExit):
    """Raised to cancel the active Modal call after SIGTERM."""


def _handle_sigterm(_signum, _frame) -> None:
    raise EvaluatorTerminated(143)


def validate_packing(centers, radii, atol=1e-10):
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    if centers.shape != (N, 2):
        return False, f"Centers shape incorrect. Expected ({N}, 2), got {centers.shape}"
    if radii.shape != (N,):
        return False, f"Radii shape incorrect. Expected ({N},), got {radii.shape}"

    if np.any(np.isnan(centers)) or np.any(np.isnan(radii)):
        return False, "NaN values in output"

    if np.any(radii < 0):
        return False, f"Negative radii found at indices: {np.where(radii < 0)[0]}"

    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        if x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol:
            return False, f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) outside unit square."

    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                return False, (
                    f"Circles {i} & {j} overlap. Dist: {dist:.4f}, "
                    f"Sum Radii: {(radii[i] + radii[j]):.4f}"
                )

    return True, None


def run_candidate() -> str:
    """Run the evolved program remotely and return its output file."""
    subprocess.run(
        ["uv", "run", "-qq", "--directory", "/evo", "python", "main.py"],
        check=True,
    )
    return Path("/evo/circle_packing.json").read_text(encoding="utf-8")


@cli.command()
def run(
    evo_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing the circle-packing implementation.",
    ),
) -> None:
    """Run and evaluate a circle-packing implementation on Modal."""
    output_file = evo_dir / "circle_packing.json"
    app = modal.App("adrevo-circle-packing")
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .uv_sync(str(evo_dir))
        # copy=True makes /evo writable so main.py can create the output file.
        .add_local_dir(str(evo_dir), remote_path="/evo", copy=True)
    )
    remote_run_candidate = app.function(image=image)(run_candidate)
    call = None
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        with app.run():
            try:
                call = run_candidate.spawn()
                output_text = call.get()
                call = None
            finally:
                if call is not None:
                    call.cancel(terminate_containers=True)

        # The trusted evaluator owns the local result used for validation.
        output_file.write_text(output_text, encoding="utf-8")
        payload = json.loads(output_text)
        is_valid, error_msg = validate_packing(
            payload["centers"], payload["radii"]
        )
        result = {
            "correct": is_valid,
            "error": error_msg,
            "combined_score": float(np.sum(payload["radii"])) if is_valid else 0.0,
        }
    except EvaluatorTerminated:
        raise
    except Exception as exc:
        result = {"correct": False, "error": str(exc), "combined_score": 0.0}
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    Path("results.json").write_text(json.dumps(result, indent=4), encoding="utf-8")


if __name__ == "__main__":
    cli()
