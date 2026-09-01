import os
import logging
import fcntl
import hashlib
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, List
from abc import ABC, abstractmethod

from adrevo.utils import extract_bytes_to_dir

logger = logging.getLogger(__name__)

def stage_data_to_node(data_zip_bytes: bytes, project_hash: str | None = None) -> str:
    """Extract data to a node-local cache directory (once).

    Uses a file lock so that only the first task on a node extracts;
    concurrent tasks block on the lock and then see the ready marker.

    Args:
        data_zip_bytes: Zip archive containing the data directories.
        project_hash: Optional cache key.  When *None* a hash of
            ``data_zip_bytes`` is used automatically.

    Returns:
        Absolute path to the cache directory.
    """
    if project_hash is None:
        project_hash = hashlib.sha256(data_zip_bytes).hexdigest()[:16]

    _CACHE_ROOT = Path.cwd() / ".adrevo_data"
    cache_dir = _CACHE_ROOT / project_hash
    marker = cache_dir / ".data_ready"

    # Fast path: already staged.
    if marker.exists():
        return str(cache_dir)

    # Ensure the cache root exists for the lock file.
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = _CACHE_ROOT / f"{project_hash}.lock"

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            # Re-check after acquiring lock — another task may have finished.
            if marker.exists():
                return str(cache_dir)

            staging_dir = cache_dir.with_name(f"{project_hash}.staging.{os.getpid()}")
            try:
                staging_dir.mkdir(parents=True, exist_ok=True)
                extract_bytes_to_dir(data_zip_bytes, str(staging_dir))

                if not cache_dir.exists():
                    staging_dir.rename(cache_dir)
                else:
                    # Shouldn't happen under lock, but be safe.
                    pass

                cache_dir.mkdir(parents=True, exist_ok=True)
                marker.touch()
                logger.info(f"Data staged to {cache_dir}")
            finally:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    return str(cache_dir)

def symlink_data_into_workdir(
    cache_dir: str,
    data_dirs: tuple,
    workdir: str,
) -> None:
    """Create symlinks in *workdir* pointing to cached data directories.

    For each entry in *data_dirs* (e.g. ``"valid_instances"``), creates::

        <workdir>/valid_instances  ->  <cache_dir>/valid_instances

    Args:
        cache_dir: Absolute path returned by :func:`stage_data_to_node`.
        data_dirs: Relative directory names matching those used in the
            project zip (same as ``BackendConfig.data_dirs``).
        workdir: The task's temporary working directory.
    """
    cache_path = Path(cache_dir)
    work_path = Path(workdir)

    for data_dir in data_dirs:
        src = cache_path / data_dir
        dst = work_path / data_dir
        if dst.exists() or dst.is_symlink():
            continue
        if not src.exists():
            logger.warning(f"Cached data directory not found: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


class ExecutionBackend(ABC):
    """Abstract base class for execution backends (Ray, Modal, Local, etc.)."""

    def __enter__(self):
        """Open a backend session. No-op by default (e.g. Ray)."""
        return self

    def __exit__(self, *exc):
        """Close a backend session. No-op by default (e.g. Ray)."""
        pass

    @abstractmethod
    def stage_data(self, data_zip_bytes: bytes) -> Any:
        """Stage immutable data files for distribution to workers.

        Called once on the driver before any jobs are submitted.
        Returns a serializable, opaque handle that can be passed to other
        backend instances (e.g. on workers) so they can access the staged data.

        Backends that don't support data staging should store the bytes or
        return ``None``.
        """
        pass

    def link_data(self, workdir: str, data_zip_bytes: bytes, data_dirs: tuple) -> None:
        """Make staged data available in a task's working directory.

        Default implementation uses node-local caching and symlinks.
        Backends may override for platform-specific mechanisms (e.g. volume
        mounts).

        Args:
            workdir: The task's temporary working directory.
            data_zip_bytes: The raw data zip bytes (retrieved from the handle).
            data_dirs: Relative directory names to symlink (from ``BackendConfig.data_dirs``).
        """
        cache_dir = stage_data_to_node(data_zip_bytes)
        symlink_data_into_workdir(cache_dir, data_dirs, workdir)

    @abstractmethod
    def run_job(
        self,
        parent_zip_bytes: bytes,
        file_replacements: Dict[str, str],
        preempt_db: Any | None = None,
        preempt_claim_id: str | None = None,
    ) -> Tuple[Dict[str, Any], float, bytes]:
        """
        Applies candidate file replacements, executes the trusted evaluator, and
        returns ``(results_dict, runtime_seconds, result_zip_bytes)``.
        """
        pass
