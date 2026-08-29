import os
import tempfile
import subprocess
import logging
import time
import signal
from pathlib import Path
from typing import Tuple, Dict, Any, List

import ray

from adrevo.config import BackendConfig
from adrevo.execution import (
    ExecutionBackend,
    stage_data_to_node,
    symlink_data_into_workdir,
)
from adrevo.utils import (
    zip_dir_to_bytes,
    extract_parent_bytes_to_dir,
    parse_results_from_zip,
)

logger = logging.getLogger(__name__)

_PREEMPTED_RETURNCODE = 254
_TIMEOUT_RETURNCODE = 255
_PREEMPT_POLL_INTERVAL_SEC = 1.0


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Terminate a subprocess and any children it spawned."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except ProcessLookupError:
        return
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=5)


def _should_preempt_work(
    preempt_db: Any | None,
    preempt_claim_id: str | None = None,
) -> bool:
    if preempt_db is None:
        return False
    try:
        if preempt_claim_id is None:
            return False
        return ray.get(
            preempt_db.should_preempt_claim.remote(preempt_claim_id)
        )
    except Exception as exc:
        logger.warning("Failed to poll ProgramDatabase for preemption: %s", exc)
        return False


def _run_preemptible_subprocess(
    cmd: List[str],
    env: Dict[str, str],
    cwd: str,
    timeout_sec: int,
    stdout_path: Path,
    stderr_path: Path,
    preempt_db: Any | None = None,
    preempt_claim_id: str | None = None,
) -> tuple[int, bool, bool]:
    deadline = time.monotonic() + timeout_sec
    preempted = False
    timed_out = False

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )

        while proc.poll() is None:
            if time.monotonic() >= deadline:
                timed_out = True
                _terminate_process_group(proc)
                break

            if _should_preempt_work(
                preempt_db=preempt_db,
                preempt_claim_id=preempt_claim_id,
            ):
                preempted = True
                _terminate_process_group(proc)
                break

            time.sleep(_PREEMPT_POLL_INTERVAL_SEC)

        if preempted:
            returncode = _PREEMPTED_RETURNCODE
        elif timed_out:
            returncode = _TIMEOUT_RETURNCODE
        else:
            returncode = proc.returncode

    return returncode, preempted, timed_out

def _run_evaluator_task(
    parent_zip_bytes: bytes,
    generated_code: str,
    exec_fname_rel: str,
    cmd: List[str],
    timeout_sec: int,
    data_zip_bytes: bytes | None = None,
    data_dirs: tuple = (),
    preempt_db: Any | None = None,
    preempt_claim_id: str | None = None,
) -> bytes:
    """
    Runs an evaluation job in an isolated temporary directory.
    Returns the zipped result directory (includes returncode.json, job_log.out, job_log.err).
    """
    import json as _json

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Extract the parent program files (data excluded)
        extract_parent_bytes_to_dir(parent_zip_bytes, temp_dir)

        # 1b. Stage data on this node (once) and symlink into workdir
        if data_zip_bytes is not None and data_dirs:
            cache_dir = stage_data_to_node(data_zip_bytes)
            symlink_data_into_workdir(cache_dir, data_dirs, temp_dir)

        # 2. Write the generated code to the specified relative path (if provided)
        if generated_code and exec_fname_rel:
            target_file = temp_path / exec_fname_rel
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(generated_code, encoding="utf-8")

        # 3. Setup environment variables
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

        # 4. Execute the command, polling for timeout or stale-parent preemption.
        stdout_path = temp_path / "job_log.out"
        stderr_path = temp_path / "job_log.err"
        returncode, preempted, timed_out = _run_preemptible_subprocess(
            cmd=cmd,
            env=env,
            cwd=temp_dir,
            timeout_sec=timeout_sec,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            preempt_db=preempt_db,
            preempt_claim_id=preempt_claim_id,
        )

        stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        if timed_out:
            stderr_text += f"\nProcess timed out after {timeout_sec} seconds."
        if preempted:
            stderr_text += "\nEvaluation preempted because the claim became stale."

        # 5. Save logs and returncode so they are included in the returned zip
        (temp_path / "job_log.out").write_text(stdout_text, encoding="utf-8")
        (temp_path / "job_log.err").write_text(stderr_text, encoding="utf-8")
        (temp_path / "returncode.json").write_text(
            _json.dumps(
                {
                    "returncode": returncode,
                    "preempted": preempted,
                    "timed_out": timed_out,
                }
            ),
            encoding="utf-8",
        )

        # 6. Zip the entire temp directory and return (excluding data dirs)
        return zip_dir_to_bytes(temp_dir, exclude_dirs=data_dirs)

def _run_command_task(
    parent_zip_bytes: bytes,
    cmd: List[str],
    timeout_sec: int = 60,
    data_zip_bytes: bytes | None = None,
    data_dirs: tuple = (),
    preempt_db: Any | None = None,
    preempt_claim_id: str | None = None,
) -> Dict[str, Any]:
    """
    Extracts parent_zip_bytes to a temp directory, runs a command, and returns
    a dict with returncode, stdout, and stderr.
    """
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        extract_parent_bytes_to_dir(parent_zip_bytes, temp_dir)

        if data_zip_bytes is not None and data_dirs:
            cache_dir = stage_data_to_node(data_zip_bytes)
            symlink_data_into_workdir(cache_dir, data_dirs, temp_dir)

        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

        stdout_path = Path(temp_dir) / "job_log.out"
        stderr_path = Path(temp_dir) / "job_log.err"
        returncode, preempted, timed_out = _run_preemptible_subprocess(
            cmd=cmd,
            env=env,
            cwd=temp_dir,
            timeout_sec=timeout_sec,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            preempt_db=preempt_db,
            preempt_claim_id=preempt_claim_id,
        )

        stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        if timed_out:
            stderr_text += f"\nCommand timed out after {timeout_sec} seconds."
        if preempted:
            stderr_text += "\nCommand preempted because the claim became stale."
        return {
            "returncode": returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "preempted": preempted,
            "timed_out": timed_out,
        }

class RayExecutionBackend(ExecutionBackend):
    """
    Ray-based implementation of the execution backend.
    It executes jobs synchronously in the current Ray actor or driver process.
    """
    def __init__(
        self,
        config: BackendConfig,
        evaluator_file: str,
        verbose: bool = True,
        data_handle=None,
    ):
        self.config = config
        self.evaluator_file = evaluator_file
        self.verbose = verbose
        self.data_handle = data_handle  # opaque handle from stage_data() (bytes)
        self.data_dirs = config.data_dirs

    def stage_data(self, data_zip_bytes: bytes) -> bytes:
        """Store data bytes for reuse by this backend and worker actors."""
        self.data_handle = data_zip_bytes
        return self.data_handle

    def _build_command(self) -> List[str]:
        # The evaluator runs from the extracted project directory. Select it
        # explicitly instead of relying on an inherited active environment.
        return ["uv", "-q", "run", "--project", ".", "python", self.evaluator_file]

    def run_job(
        self,
        parent_zip_bytes: bytes,
        generated_code: str,
        exec_fname_rel: str,
        preempt_db: Any | None = None,
        preempt_claim_id: str | None = None,
    ) -> Tuple[Dict[str, Any], float, bytes]:

        cmd = self._build_command()

        t0 = time.time()

        if self.verbose:
            logger.info(f"Running job for {exec_fname_rel}...")

        result_zip_bytes: bytes = _run_evaluator_task(
            parent_zip_bytes=parent_zip_bytes,
            generated_code=generated_code,
            exec_fname_rel=exec_fname_rel,
            cmd=cmd,
            timeout_sec=self.config.timeout_sec,
            data_zip_bytes=self.data_handle,
            data_dirs=self.data_dirs,
            preempt_db=preempt_db,
            preempt_claim_id=preempt_claim_id,
        )
        rtime = time.time() - t0

        # Parse the results directly from the zip bytes
        results = parse_results_from_zip(result_zip_bytes)
        returncode = results.get("returncode")

        if self.verbose:
            logger.info(f"Job completed in {rtime:.2f}s with return code: {returncode}")

        # Communicate this failure back to LLM by appending to stderr_log.
        if returncode is not None and returncode != 0:
            results["stderr_log"] += f"\nProcess failed with return code {returncode}."

        return results, rtime, result_zip_bytes

    def run_command(
        self,
        parent_zip_bytes: bytes,
        cmd: List[str],
        preempt_db: Any | None = None,
        preempt_claim_id: str | None = None,
    ) -> Dict[str, Any]:
        return _run_command_task(
            parent_zip_bytes=parent_zip_bytes,
            cmd=cmd,
            timeout_sec=self.config.timeout_sec,
            data_zip_bytes=self.data_handle,
            data_dirs=self.data_dirs,
            preempt_db=preempt_db,
            preempt_claim_id=preempt_claim_id,
        )

    def run_command_with_zip(
        self,
        parent_zip_bytes: bytes,
        cmd: List[str],
        preempt_db: Any | None = None,
        preempt_claim_id: str | None = None,
    ) -> Tuple[Dict[str, Any], bytes]:
        result_zip_bytes: bytes = _run_evaluator_task(
            parent_zip_bytes=parent_zip_bytes,
            generated_code="",
            exec_fname_rel="",
            cmd=cmd,
            timeout_sec=self.config.timeout_sec,
            data_zip_bytes=self.data_handle,
            data_dirs=self.data_dirs,
            preempt_db=preempt_db,
            preempt_claim_id=preempt_claim_id,
        )
        results = parse_results_from_zip(result_zip_bytes)
        return results, result_zip_bytes
