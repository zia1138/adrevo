from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

class ExecutionBackend(ABC):
    """Abstract base class for execution backends (Ray, Modal, Local, etc.)."""

    def __enter__(self):
        """Open a backend session. No-op by default (e.g. Ray)."""
        return self

    def __exit__(self, *exc):
        """Close a backend session. No-op by default (e.g. Ray)."""
        pass

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
