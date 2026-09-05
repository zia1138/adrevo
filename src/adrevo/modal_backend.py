"""Modal-based execution backend (DRAFT).

This is a first draft and has NOT been tested against the live Modal API.
"""

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

from adrevo.config import BackendConfig
from adrevo.execution import ExecutionBackend
from adrevo.utils import extract_parent_bytes_to_dir, parse_results_from_zip, zip_dir_to_bytes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default container image — includes uv so evaluate.py can resolve deps.
# Projects needing extra system packages should pass a custom image.
# ---------------------------------------------------------------------------

_DEFAULT_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
)


# ---------------------------------------------------------------------------
# Remote Modal functions
# ---------------------------------------------------------------------------

app = modal.App("adrevo")


@app.function(
    image=_DEFAULT_IMAGE,
    timeout=20 * 60,
)
def modal_evaluator_task(
    parent_zip_bytes: bytes,
    file_replacements: Dict[str, str],
    cmd: List[str],
    timeout_sec: int,
) -> bytes:
    """Run an evaluation in an ephemeral Modal container."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Extract project files.
        extract_parent_bytes_to_dir(parent_zip_bytes, temp_dir)

        # 2. Apply candidate file replacements.
        for file_path, content in file_replacements.items():
            target_file = temp_path / file_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content, encoding="utf-8")

        # 3. Execute.
        try:
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=temp_dir,
                timeout=timeout_sec,
            )
            stdout_text = cp.stdout or ""
            stderr_text = cp.stderr or ""
            returncode = cp.returncode
        except subprocess.TimeoutExpired:
            stdout_text = f"Process timed out after {timeout_sec} seconds."
            stderr_text = ""
            returncode = 255

        # 4. Save logs and returncode.
        (temp_path / "job_log.out").write_text(stdout_text, encoding="utf-8")
        (temp_path / "job_log.err").write_text(stderr_text, encoding="utf-8")
        (temp_path / "returncode.json").write_text(
            json.dumps({"returncode": returncode}), encoding="utf-8"
        )

        # 5. Return zipped results.
        return zip_dir_to_bytes(temp_dir)


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class ModalExecutionBackend(ExecutionBackend):
    """Modal-based execution backend.

    Runs each evaluation in a fresh serverless container.
    """

    def __init__(
        self,
        config: BackendConfig,
        verbose: bool = True,
    ):
        self.config = config
        self.verbose = verbose
        self._app_context = None

    def __enter__(self):
        """Open a persistent Modal session."""
        self._app_context = app.run()
        self._app_context.__enter__()
        return self

    def __exit__(self, *exc):
        """Close the Modal session."""
        if self._app_context is not None:
            self._app_context.__exit__(*exc)
            self._app_context = None

    def _build_command(self) -> List[str]:
        return ["uv", "run", "-qq", "--project", ".", "python", "evaluate.py"]

    def run_job(
        self,
        parent_zip_bytes: bytes,
        file_replacements: Dict[str, str],
        preempt_db: Any | None = None,
        preempt_score: float | None = None,
    ) -> Tuple[Dict[str, Any], float, bytes]:

        cmd = self._build_command()
        t0 = time.time()

        if self.verbose:
            logger.info("Submitting Modal job with replacements for %s", ", ".join(file_replacements))

        result_zip_bytes: bytes = modal_evaluator_task.remote(
            parent_zip_bytes=parent_zip_bytes,
            file_replacements=file_replacements,
            cmd=cmd,
            timeout_sec=self.config.timeout_sec,
        )

        rtime = time.time() - t0
        results = parse_results_from_zip(result_zip_bytes)
        returncode = results.get("returncode")

        if self.verbose:
            logger.info(f"Modal job completed in {rtime:.2f}s with return code: {returncode}")

        if returncode is not None and returncode != 0:
            results["stderr_log"] += f"\nProcess failed with return code {returncode}."

        return results, rtime, result_zip_bytes
