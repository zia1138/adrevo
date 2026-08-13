import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Optional
import math
import ray
import logfire
import uuid

from adrevo.config import AdrevoConfig

logger = logging.getLogger(__name__)
_logging_configured = False

def clean_nan_values(obj: Any) -> Any:
    """
    Recursively clean NaN values from a data structure, replacing them with
    None. This ensures JSON serialization works correctly.
    """
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_nan_values(item) for item in obj)
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.floating):
        if np.isscalar(obj):
            if np.isnan(obj) or np.isinf(obj):
                return None
            else:
                return float(obj)
        else:
            return clean_nan_values(obj.tolist())
    else:
        return obj

@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    code: str
    language: str 
    model_id: str
    timestamp: float = field(default_factory=time.time) 

    # Evolution information
    parent_id: Optional[str] = None
    generation: int = 0
    inference_time: float = 0.0
    compute_time: float = 0.0

    # Performance metrics
    combined_score: float = 0.0
    correct: bool = False
    children_count: int = 0
    children_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict representation, cleaning NaN values for JSON."""
        data = asdict(self)
        return clean_nan_values(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation, ensuring correct types for nested dicts."""

        program_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in program_fields}

        return cls(**filtered_data)


@dataclass
class ParentClaim:
    """Represents one active or completed parent-work reservation."""

    claim_id: str
    worker_id: str
    parent_id: str
    generation: int
    search_epoch: int

    model_id: Optional[str] = None
    model_rank: Optional[int] = None

    finished: bool = False
    outcome: Optional[str] = None
    child_id: Optional[str] = None


@ray.remote(num_cpus=1)
class ProgramDatabase:
    """In-memory program store and search scheduler.

    ``database_state.json`` contains the program records, scheduler cursors, and
    search epoch. Each program's executable project is persisted separately as
    ``{program_id}.zip``. Claims, cancellation, logging, and model lookup tables
    exist only for the current process and are recreated on resume.
    """

    STATE_FILENAME = "database_state.json"

    def __init__(
        self,
        evo_config: AdrevoConfig,
        verbose: bool = False,
        resume_dir: Optional[str] = None,
    ):
        # Runtime configuration. It is supplied again on resume and is not part
        # of the database checkpoint.
        self.evo_config = evo_config

        # ------------------------------------------------------------------
        # Checkpointed database state (returned by get_database_state()).
        # ------------------------------------------------------------------
        # Program records contain code, scores, lineage, and timing metadata.
        # children_count/children_ids are caches rebuilt from parent_id on load.
        self.programs: Dict[str, Program] = {}

        # Persistent scheduler position is driven by four program IDs:
        # - search_focus_id: the parent new claims will expand.
        # - global_best_id: the best program ever found; only this commits a
        #   lineage.
        # - branch_best_id: the best local improvement on the current side
        #   branch; it can move focus without committing lineage.
        # - backtrack_from_id: the cursor on the committed lineage. Max-rank
        #   failures walk this cursor upward.
        #
        # State transitions:
        # - New global best: focus = global_best = backtrack_from, branch clears.
        # - New branch best: focus = branch_best, committed lineage unchanged.
        # - Backtrack: focus = backtrack_from's ancestor, branch clears.
        # - Other stored programs: focus does not move.
        self.global_best_id: Optional[str] = None 
        self.search_focus_id: Optional[str] = None
        self.branch_best_id: Optional[str] = None
        self.backtrack_from_id: Optional[str] = None
        # Saved so a resumed scheduler can advance from the last known epoch.
        self.search_epoch: int = 0

        # ------------------------------------------------------------------
        # Program payloads, persisted separately as {program_id}.zip files.
        # ------------------------------------------------------------------
        # The actor keeps their bytes in memory so workers can evaluate a parent.
        # On resume, _load_checkpoint() reloads these files from the results dir.
        self.pid_zip_bytes: Dict[str, bytes] = {}

        # ------------------------------------------------------------------
        # Transient coordination state. This is deliberately not checkpointed.
        # ------------------------------------------------------------------
        # Old workers no longer exist after restart, so claims start empty and a
        # previous cancellation request does not carry into the resumed run.
        self.active_claims: Dict[str, ParentClaim] = {}
        self.finished_claims: Dict[str, ParentClaim] = {}
        self.cancellation_requested: bool = False
        self.cancellation_reason: Optional[str] = None

        # ------------------------------------------------------------------
        # Runtime-only state derived from the current configuration.
        # ------------------------------------------------------------------
        self.evo_models = self.evo_config.build_evo_models()
        self.model_by_id = {spec.model_id: spec for spec in self.evo_models}
        self.model_to_rank = {
            spec.model_id: rank for rank, spec in enumerate(self.evo_models)
        }
        self.max_model_rank = len(self.evo_models) - 1

        self.verbose = verbose
        global _logging_configured
        if self.verbose:
            if not _logging_configured:
                logfire.configure()
                logger.addHandler(logfire.LogfireLoggingHandler())
                _logging_configured = True
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)

        if resume_dir is None:
            logger.info("Initialized fresh in-memory database.")
        else:
            self._load_checkpoint(Path(resume_dir))
            logger.info(
                "Resumed database from %s with %d programs at epoch %d.",
                resume_dir,
                len(self.programs),
                self.search_epoch,
            )

    def _state_snapshot(self) -> Dict[str, Any]:
        """Return exactly the state persisted in database_state.json."""
        return {
            "programs": [program.to_dict() for program in self.programs.values()],
            "global_best_id": self.global_best_id,
            "search_focus_id": self.search_focus_id,
            "branch_best_id": self.branch_best_id,
            "backtrack_from_id": self.backtrack_from_id,
            "search_epoch": self.search_epoch,
        }

    def _load_checkpoint(self, resume_dir: Path) -> None:
        """Restore completed programs and scheduler state from a run directory."""
        checkpoint_path = resume_dir / self.STATE_FILENAME
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        self.programs = {
            program_data["id"]: Program.from_dict(program_data)
            for program_data in state["programs"]
        }

        # Rebuild the relationship cache from parent IDs.
        for program in self.programs.values():
            program.children_count = 0
            program.children_ids = []
        for program in self.programs.values():
            if program.parent_id is not None:
                parent = self.programs[program.parent_id]
                parent.children_count += 1
                parent.children_ids.append(program.id)

        for cursor_name in (
            "global_best_id",
            "search_focus_id",
            "branch_best_id",
            "backtrack_from_id",
        ):
            setattr(self, cursor_name, state[cursor_name])

        # Claims are intentionally not restored; advance the epoch across the
        # process boundary so any old claim is unambiguously stale.
        self.search_epoch = state["search_epoch"] + 1
        self.pid_zip_bytes = {
            program_id: (resume_dir / f"{program_id}.zip").read_bytes()
            for program_id in self.programs
        }

    def close(self) -> Dict[str, Any]:
        """Compatibility hook for driver cleanup."""
        return {"closed": True}

    def add_initial(
        self,
        program: Program,
        zip_bytes: bytes,
        verbose: bool = False,
    ) -> str:
        """Atomically add the initial program and initialize search cursors."""
        if self.global_best_id is not None:
            raise RuntimeError("Initial program has already been added.")
        if program.parent_id is not None:
            raise ValueError("Initial program must not have a parent_id.")
        if not program.correct or program.combined_score is None:
            raise ValueError("Initial program must be correct and scored.")

        self.programs[program.id] = program
        self.pid_zip_bytes[program.id] = zip_bytes
        self.global_best_id = program.id
        self.search_focus_id = program.id
        self.branch_best_id = None
        self.backtrack_from_id = program.id
        logger.info(
            "SEARCH_INIT best=%s focus=%s epoch=%d.",
            self.global_best_id,
            self.search_focus_id,
            self.search_epoch,
        )

        if verbose:
            logger.info("INITIAL_ADD id=%s score=%s.", program.id, program.combined_score)

        return program.id

    def add(
        self,
        program: Program,
        zip_bytes: bytes,
        verbose: bool = False,
    ) -> str:
        """Atomically add a program and its executable project state."""
        if self.global_best_id is None:
            raise RuntimeError("ProgramDatabase.add_initial() must be called before add().")

        parent_improved = False

        # Update parent's children count and derive whether this child improved
        # its parent. This is search policy state, not persistent Program state.
        if program.parent_id and program.parent_id in self.programs:
            parent = self.programs[program.parent_id]
            parent.children_count += 1
            parent.children_ids.append(program.id)
            parent_improved = (
                parent.combined_score is not None
                and program.combined_score is not None
                and program.combined_score > parent.combined_score
            )

        # Store the program and its project state before publishing any search
        # focus transition. ProgramDatabase is a single Ray actor, so workers
        # cannot observe the new focus without also being able to retrieve its
        # zip bytes.
        self.programs[program.id] = program
        self.pid_zip_bytes[program.id] = zip_bytes

        if program.correct and program.combined_score is not None:
            current_global_best = self.programs.get(self.global_best_id)
            current_global_score = (
                current_global_best.combined_score
                if current_global_best is not None and current_global_best.combined_score is not None
                else -float('inf')
            )

            if program.combined_score > current_global_score:
                # New global best: commit this lineage. Workers focus here, and
                # future backtracking starts from here.
                self.global_best_id = program.id
                self.search_focus_id = program.id
                self.branch_best_id = None
                self.backtrack_from_id = program.id
                self.search_epoch += 1
                logger.info(
                    "GLOBAL_COMMIT id=%s focus=%s epoch=%d.",
                    program.id,
                    self.search_focus_id,
                    self.search_epoch,
                )
            elif parent_improved:
                current_branch_best = (
                    self.programs.get(self.branch_best_id)
                    if self.branch_best_id is not None
                    else None
                )
                current_branch_score = (
                    current_branch_best.combined_score
                    if current_branch_best is not None and current_branch_best.combined_score is not None
                    else -float('inf')
                )

                if program.combined_score > current_branch_score:
                    # New branch best: workers focus here, but the committed
                    # lineage and backtracking cursor stay unchanged.
                    previous_focus_id = self.search_focus_id
                    previous_branch_best_id = self.branch_best_id
                    self.search_focus_id = program.id
                    self.branch_best_id = program.id
                    self.search_epoch += 1
                    logger.info(
                        "BRANCH_FOCUS id=%s prev_branch=%s prev_focus=%s epoch=%d.",
                        program.id,
                        previous_branch_best_id,
                        previous_focus_id,
                        self.search_epoch,
                    )
                # Otherwise: this program is stored, but focus does not move.

        if verbose:
            logger.info("ADD id=%s score=%s.", program.id, program.combined_score)

        return program.id

    def claim_next_parent(self, worker_id: str, generation: int) -> Optional[Dict[str, Any]]:
        """Claim the next parent to work on.

        Workers only claim the current search focus; the database owns when
        that focus moves to a global best, branch best, or backtracked ancestor.
        """
        if self.cancellation_requested:
            return None

        parent = self.programs.get(self.search_focus_id) if self.search_focus_id else None
        if parent is None:
            return None
        if parent.id not in self.pid_zip_bytes:
            return None

        claim_id = str(uuid.uuid4())
        claim = ParentClaim(
            claim_id=claim_id,
            worker_id=worker_id,
            parent_id=parent.id,
            generation=generation,
            search_epoch=self.search_epoch,
        )
        self.active_claims[claim_id] = claim
        return {
            "claim_id": claim.claim_id,
            "parent": parent,
        }

    def finish_parent_claim(
        self,
        claim_id: str,
        outcome: str,
        child_id: Optional[str] = None,
    ) -> None:
        """Mark a parent claim complete and release it from active work."""
        claim = self.active_claims.pop(claim_id, None)
        if claim is None:
            return
        claim.finished = True
        claim.outcome = outcome
        claim.child_id = child_id    
        self.finished_claims[claim_id] = claim

        if (
            claim.outcome != "failed"
            or claim.model_rank is None
            or claim.model_rank != self.max_model_rank
        ):
            return

        logger.info(
            "MAX_RANK_FAILED claim=%s parent=%s model=%s rank=%s epoch=%d/%d.",
            claim.claim_id,
            claim.parent_id,
            claim.model_id,
            claim.model_rank,
            claim.search_epoch,
            self.search_epoch,
        )
        if claim.search_epoch != self.search_epoch:
            logger.info(
                "STALE_IGNORED claim=%s outcome=%s epoch=%d/%d.",
                claim.claim_id,
                claim.outcome,
                claim.search_epoch,
                self.search_epoch,
            )
            return

        if self.backtrack_from_id is None:
            return

        previous_focus_id = self.search_focus_id
        previous_backtrack_from_id = self.backtrack_from_id
        next_focus_id = self.backtrack_from_id
        steps_taken = 0

        for _ in range(self.evo_config.backtrack_steps):
            backtrack_from = self.programs.get(next_focus_id)
            if backtrack_from is None or backtrack_from.parent_id is None:
                break
            next_focus_id = backtrack_from.parent_id
            steps_taken += 1

        if next_focus_id not in self.programs:
            logger.warning("Cannot backtrack from missing program %s.", next_focus_id)
            return

        # Backtrack: workers focus on the selected ancestor. The temporary
        # branch is discarded, and repeated failures continue walking upward
        # from this ancestor.
        self.search_focus_id = next_focus_id
        self.branch_best_id = None
        self.backtrack_from_id = next_focus_id
        self.search_epoch += 1
        logger.info(
            "BACKTRACK from=%s to=%s steps=%d/%d prev_focus=%s epoch=%d.",
            previous_backtrack_from_id,
            next_focus_id,
            steps_taken,
            self.evo_config.backtrack_steps,
            previous_focus_id,
            self.search_epoch,
        )


    def update_claim_model(
        self,
        claim_id: str,
        model_id: str,
    ) -> None:
        """Record the model currently working on an active claim."""
        claim = self.active_claims.get(claim_id)
        if claim is None:
            return

        claim.model_id = model_id
        claim.model_rank = self.model_to_rank.get(model_id)

    def should_preempt_claim(
        self,
        claim_id: str,
    ) -> bool:
        """Return true when a worker's current claim should be abandoned."""
        if self.cancellation_requested:
            return True

        claim = self.active_claims.get(claim_id)
        if claim is None:
            return True

        return self.search_epoch != claim.search_epoch

    def request_cancellation(self, reason: str = "cancelled") -> Dict[str, Any]:
        """Request system-wide cancellation through the database.

        Active claims become stale because their search epoch no longer matches
        the database epoch. Workers and backend jobs poll should_preempt_claim(),
        so this is the central cancellation path for running work.
        """
        if not self.cancellation_requested:
            self.cancellation_requested = True
            self.cancellation_reason = reason
            self.search_epoch += 1
            logger.info(
                "Cancellation requested: %s. Invalidated %d active claims.",
                reason,
                len(self.active_claims),
            )

        return {
            "cancelled": self.cancellation_requested,
            "reason": self.cancellation_reason,
            "active_claims": len(self.active_claims),
            "search_epoch": self.search_epoch,
        }

    def is_cancelled(self) -> bool:
        """Return whether the database has received a cancellation request."""
        return self.cancellation_requested

    def get_best_program_and_score(self) -> tuple[Optional[Program], float]:
        """Return the current best correct program and its score."""
        if self.global_best_id is None:
            return None, -float('inf')

        best_program = self.programs.get(self.global_best_id)
        if best_program is None or best_program.combined_score is None:
            return None, -float('inf')

        return best_program, best_program.combined_score

    def get_best_score(self):
        if self.global_best_id is None:
            return -float('inf')

        best_program = self.programs.get(self.global_best_id)
        if best_program is None or best_program.combined_score is None:
            return -float('inf')

        return best_program.combined_score

    def get(self, program_id: str) -> Optional[Program]:
        return self.programs.get(program_id)

    def get_zip_program_ids(self) -> List[str]:
        """Return the current list of all program IDs that have associated zip bytes."""
        return list(self.pid_zip_bytes.keys())

    def get_zip_bytes(self, program_id: str) -> bytes: 
        """Retrieve zip bytes for a given program ID."""
        return self.pid_zip_bytes[program_id]

    def get_database_state(self) -> Dict[str, Any]:
        """Return the programs and scheduler state as a JSON-serializable dict."""
        return self._state_snapshot()
