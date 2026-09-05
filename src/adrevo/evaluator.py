import os
import tempfile
import subprocess
import logging
import time
import signal
from pathlib import Path
from typing import Tuple, Dict, Any, List

import ray

from adrevo.utils import (
    zip_dir_to_bytes,
    extract_parent_bytes_to_dir,
    parse_results_from_zip,
)

logger = logging.getLogger(__name__)

_PREEMPTED_RETURNCODE = 254
_TIMEOUT_RETURNCODE = 255
_PREEMPT_POLL_INTERVAL_SEC = 1.0


def _terminate_process_group(
    proc: subprocess.Popen,
    termination_grace_sec: int,
) -> None:
    """Terminate a subprocess and any children it spawned."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=termination_grace_sec)
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
    cwd: str,
    timeout_sec: int | None,
    termination_grace_sec: int,
    stdout_path: Path,
    stderr_path: Path,
    preempt_db: Any | None = None,
    preempt_claim_id: str | None = None,
) -> tuple[int, bool, bool]:
    deadline = time.monotonic() + timeout_sec if timeout_sec is not None else None
    preempted = False
    timed_out = False

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            cwd=cwd,
            start_new_session=True,
        )

        while proc.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                _terminate_process_group(proc, termination_grace_sec)
                break

            if _should_preempt_work(
                preempt_db=preempt_db,
                preempt_claim_id=preempt_claim_id,
            ):
                preempted = True
                _terminate_process_group(proc, termination_grace_sec)
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
    file_replacements: Dict[str, str],
    cmd: List[str],
    timeout_sec: int | None,
    termination_grace_sec: int,
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

        # 1. Extract the parent program files.
        extract_parent_bytes_to_dir(parent_zip_bytes, temp_dir)

        # 2. Apply the candidate file replacements.
        if not isinstance(file_replacements, dict):
            raise TypeError("file_replacements must be a dictionary")
        for file_path, content in file_replacements.items():
            relative_path = Path(file_path)
            if (
                not isinstance(file_path, str)
                or not file_path
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or not isinstance(content, str)
            ):
                raise ValueError("file_replacements must map safe relative paths to string contents")
            target_file = temp_path / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content, encoding="utf-8")

        # 3. Execute the command, polling for timeout or stale-parent preemption.
        stdout_path = temp_path / "job_log.out"
        stderr_path = temp_path / "job_log.err"
        returncode, preempted, timed_out = _run_preemptible_subprocess(
            cmd=cmd,
            cwd=temp_dir,
            timeout_sec=timeout_sec,
            termination_grace_sec=termination_grace_sec,
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

        # 4. Save logs and returncode so they are included in the returned zip
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

        # 5. Zip the entire temp directory and return.
        return zip_dir_to_bytes(temp_dir)

def run_evaluator(
    parent_zip_bytes: bytes,
    file_replacements: Dict[str, str],
    evaluator_file: str,
    evaluator_timeout_sec: int | None,
    evaluator_termination_grace_sec: int,
    verbose: bool = False,
    preempt_db: Any | None = None,
    preempt_claim_id: str | None = None,
) -> Tuple[Dict[str, Any], float, bytes]:
    """Evaluate candidate file replacements with the trusted evaluator."""
    cmd = ["uv", "run", "-qq", "--project", ".", "python", evaluator_file]
    started_at = time.time()

    if verbose:
        logger.info("Running evaluator with replacements for %s", ", ".join(file_replacements))

    result_zip_bytes = _run_evaluator_task(
        parent_zip_bytes=parent_zip_bytes,
        file_replacements=file_replacements,
        cmd=cmd,
        timeout_sec=evaluator_timeout_sec,
        termination_grace_sec=evaluator_termination_grace_sec,
        preempt_db=preempt_db,
        preempt_claim_id=preempt_claim_id,
    )
    runtime_sec = time.time() - started_at
    results = parse_results_from_zip(result_zip_bytes)
    returncode = results.get("returncode")

    if verbose:
        logger.info(
            "Evaluator completed in %.2fs with return code: %s",
            runtime_sec,
            returncode,
        )

    if returncode is not None and returncode != 0:
        results["stderr_log"] += f"\nProcess failed with return code {returncode}."

    return results, runtime_sec, result_zip_bytes
