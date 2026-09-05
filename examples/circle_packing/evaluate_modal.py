"""Run the circle-packing candidate on Modal and validate its output locally."""

import json
import subprocess
from pathlib import Path

import modal
import numpy as np


app = modal.App("adrevo-circle-packing")
image = (
    modal.Image.debian_slim(python_version="3.13")
    # Build the candidate environment from evo/pyproject.toml.
    .uv_sync("evo")
    # copy=True makes /evo writable so main.py can create circle_packing.json.
    .add_local_dir("evo", remote_path="/evo", copy=True)
)


@app.function(image=image)
def run_candidate() -> str:
    """Run the evolved program remotely and return its output file."""
    subprocess.run(
        ["uv", "run", "-qq", "--directory", "/evo", "python", "main.py"],
        check=True,
    )
    return Path("/evo/circle_packing.json").read_text(encoding="utf-8")


def main() -> None:
    # This import is intentionally local: Modal imports this module remotely to
    # execute run_candidate(), but validation belongs to the trusted evaluator.
    from evaluate import OUTPUT_FILE, validate_packing

    try:
        with app.run():
            output_text = run_candidate.remote()

        # The trusted evaluator owns the local result used for validation.
        OUTPUT_FILE.write_text(output_text, encoding="utf-8")
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
