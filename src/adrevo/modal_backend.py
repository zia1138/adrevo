"""Modal-based execution backend (DRAFT).

This is a first draft and has NOT been tested against the live Modal API.
"""

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

from adrevo.config import BackendConfig
from adrevo.execution import ExecutionBackend
from adrevo.utils import (
    extract_bytes_to_dir,
    extract_parent_bytes_to_dir,
    parse_results_from_zip,
    zip_dir_to_bytes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modal app and volume names
# ---------------------------------------------------------------------------

_DATA_VOLUME_NAME = "adrevo-data"
_DATA_MOUNT_PATH = "/data"

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
data_volume = modal.Volume.from_name(_DATA_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=_DEFAULT_IMAGE,
    volumes={_DATA_MOUNT_PATH: data_volume},
    timeout=20 * 60,
)
def modal_evaluator_task(
    parent_zip_bytes: bytes,
    file_replacements: Dict[str, str],
    cmd: List[str],
    timeout_sec: int,
    data_hash: str | None,
    data_dirs: tuple,
) -> bytes:
    """Run an evaluation in an ephemeral Modal container.

    Data directories are available via the mounted volume at
    ``/data/<data_hash>/<data_dir>``.  Symlinks are created in the
    working directory so that evaluate.py sees them at their expected
    relative paths.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Extract code files (data excluded from zip).
        extract_parent_bytes_to_dir(parent_zip_bytes, temp_dir)

        # 2. Symlink data from the volume mount into the workdir.
        if data_hash and data_dirs:
            volume_data_root = Path(_DATA_MOUNT_PATH) / data_hash
            for d in data_dirs:
                src = volume_data_root / d
                dst = temp_path / d
                if dst.exists() or dst.is_symlink():
                    continue
                if not src.exists():
                    logger.warning(f"Data dir not found on volume: {src}")
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src)

        # 3. Apply candidate file replacements.
        for file_path, content in file_replacements.items():
            target_file = temp_path / file_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content, encoding="utf-8")

        # 4. Execute.
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

        # 5. Save logs and returncode.
        (temp_path / "job_log.out").write_text(stdout_text, encoding="utf-8")
        (temp_path / "job_log.err").write_text(stderr_text, encoding="utf-8")
        (temp_path / "returncode.json").write_text(
            json.dumps({"returncode": returncode}), encoding="utf-8"
        )

        # 6. Return zipped results (data dirs excluded).
        return zip_dir_to_bytes(temp_dir, exclude_dirs=data_dirs)


# ---------------------------------------------------------------------------
# Helper: upload data to the volume (runs as a Modal function so it can
# write to the volume from inside Modal's infrastructure).
# ---------------------------------------------------------------------------

@app.function(
    image=_DEFAULT_IMAGE,
    volumes={_DATA_MOUNT_PATH: data_volume},
    timeout=10 * 60,
)
def _upload_data_to_volume(data_zip_bytes: bytes, data_hash: str) -> None:
    """Extract data zip into the volume at /data/<data_hash>/."""
    dest = Path(_DATA_MOUNT_PATH) / data_hash
    marker = dest / ".data_ready"
    if marker.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    extract_bytes_to_dir(data_zip_bytes, str(dest))
    marker.touch()
    data_volume.commit()


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class ModalExecutionBackend(ExecutionBackend):
    """Modal-based execution backend.

    Runs each evaluation in a fresh serverless container. Data files are
    uploaded once to a Modal Volume and mounted into every container.
    """

    def __init__(
        self,
        config: BackendConfig,
        verbose: bool = True,
        data_handle: str | None = None,
    ):
        self.config = config
        self.verbose = verbose
        self.data_handle = data_handle  # data_hash string (or None)
        self.data_dirs = config.data_dirs
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

    def stage_data(self, data_zip_bytes: bytes) -> str:
        """Upload data to a Modal Volume. Returns the data hash key.

        Must be called within an active session (i.e. inside ``with backend:``).
        """
        data_hash = hashlib.sha256(data_zip_bytes).hexdigest()[:16]

        if self.verbose:
            logger.info(
                f"Uploading data to Modal volume '{_DATA_VOLUME_NAME}' "
                f"({len(data_zip_bytes) / 1024 / 1024:.1f} MB, hash={data_hash})"
            )

        _upload_data_to_volume.remote(data_zip_bytes, data_hash)

        self.data_handle = data_hash
        return data_hash

    def link_data(self, workdir: str, data_zip_bytes: bytes, data_dirs: tuple) -> None:
        """No-op — data is mounted via Modal Volume, symlinked inside tasks."""
        pass

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
            data_hash=self.data_handle,
            data_dirs=self.data_dirs,
        )

        rtime = time.time() - t0
        results = parse_results_from_zip(result_zip_bytes)
        returncode = results.get("returncode")

        if self.verbose:
            logger.info(f"Modal job completed in {rtime:.2f}s with return code: {returncode}")

        if returncode is not None and returncode != 0:
            results["stderr_log"] += f"\nProcess failed with return code {returncode}."

        return results, rtime, result_zip_bytes
