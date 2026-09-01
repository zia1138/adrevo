import time
import uuid
import logging
import json
from datetime import datetime
from pathlib import Path
import ray
from adrevo.database import ProgramDatabase, Program
from adrevo.agents import AdrevoModelCoordinator, AdrevoState, AdrevoWorker
from adrevo.config import AdrevoConfig, BackendConfig
from adrevo.ray_backend import RayExecutionBackend
from adrevo.utils import zip_dir_to_bytes, zip_data_dirs_to_bytes
import logfire

# Set up logging
logger = logging.getLogger(__name__)
_logging_configured = False

## NOTE: AdrevoDriver still runs in the ray driver. 
class AdrevoDriver:
    def __init__(
        self,
        evo_config: AdrevoConfig,
        backend_config: BackendConfig,
        project_dir: str | Path,
        results_dir: Path,
        verbose: bool = False,
        resume_from: Path | None = None,
    ):
        self.evo_config = evo_config
        self.backend_config = backend_config
        self.project_dir = Path(project_dir)
        self.results_dir = Path(results_dir)
        self.verbose = verbose
        self.start_gen = 0
        self.resume_from = Path(resume_from) if resume_from is not None else None
        self.state_checkpoint = None

        # Defensive check for programmatic callers. The CLI performs this before
        # starting Ray, and run_ray() atomically creates the directory before use.
        if self.resume_from is None and self.results_dir.exists():
            raise ValueError(
                f"Results directory already exists: {self.results_dir}\n"
                "Use a different results directory or remove the existing one."
            )
        if self.resume_from is not None:
            if self.resume_from.resolve() != self.results_dir.resolve():
                raise ValueError("Resume directory must be the results directory")
            if not self.results_dir.is_dir():
                raise ValueError(f"Resume results directory not found: {self.results_dir}")
            state_path = self.results_dir / "evogen_state.json"
            try:
                self.state_checkpoint = json.loads(state_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise ValueError(f"Resume evolution state not found: {state_path}") from exc
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read resume evolution state: {state_path}") from exc

        if self.verbose:
            global _logging_configured
            if not _logging_configured:
                logfire.configure()
                logger.addHandler(logfire.LogfireLoggingHandler())
                _logging_configured = True
            logger.setLevel(logging.INFO)

            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Evolution run started at {start_time}")
            logger.info(f"Results directory: {self.results_dir}")

        self.db = ProgramDatabase.remote(
            evo_config,
            verbose=self.verbose,
            resume_dir=str(self.resume_from) if self.resume_from is not None else None,
        )

        # TODO: Allow modal, beam.could and other backends here.
        self.backend = RayExecutionBackend(
            config=backend_config,
            evaluator_file=evo_config.evaluate_file,
            verbose=verbose,
        )

        # Stage data files via the backend's transport mechanism.
        if backend_config.data_dirs:
            data_zip_bytes = zip_data_dirs_to_bytes(self.project_dir, backend_config.data_dirs)
            self.data_handle = self.backend.stage_data(data_zip_bytes)
            if self.verbose:
                logger.info(
                    f"Data dirs {backend_config.data_dirs} staged "
                    f"({len(data_zip_bytes) / 1024 / 1024:.1f} MB compressed)"
                )
        else:
            self.data_handle = None


    def run_ray(self):
        """Ray based evolution."""
        all_refs = []
        state = None
        interrupted = False

        try:
            if self.resume_from is None:
                self._create_results_dir()

            try:
                if self.resume_from is None:
                    with self.backend:
                        self._run_generation_0()

                state = AdrevoState.remote(
                    self.evo_config,
                    self.start_gen,
                    checkpoint=self.state_checkpoint,
                )
                cur_gen: int = ray.get(state.get_gen.remote())
                cur_cost: float = ray.get(state.compute_cost.remote())
                if (
                    cur_gen < self.evo_config.max_generations
                    and cur_cost < self.evo_config.max_cost
                ):
                    model_coordinator = AdrevoModelCoordinator.remote(self.evo_config)
                    for agent_id in range(self.evo_config.num_agent_workers):
                        logger.info(f"Starting agent worker {agent_id}.")
                        worker = AdrevoWorker.remote(
                            f"agent_{agent_id}",
                            state,
                            self.evo_config,
                            self.backend_config,
                            self.db,
                            model_coordinator,
                            self.verbose,
                            self.data_handle,
                        )
                        all_refs.append(worker.run.remote())

                    while (
                        cur_gen < self.evo_config.max_generations
                        and cur_cost < self.evo_config.max_cost
                    ):
                        self._download_database_state()
                        self._write_generation_state(
                            ray.get(state.snapshot.remote())
                        )

                        time.sleep(self.evo_config.dl_evostate_freq)
                        cur_gen = ray.get(state.get_gen.remote())
                        cur_cost = ray.get(state.compute_cost.remote())

                    # All in-flight claims finish or observe their stop condition.
                    ray.get(all_refs)
                elif self.verbose:
                    logger.info(
                        "Run state already reached a configured limit "
                        f"(generation={cur_gen}, cost={cur_cost:.4f})."
                    )
            except KeyboardInterrupt:
                interrupted = True
                self._request_database_cancellation("keyboard_interrupt")
                self._wait_for_workers_to_stop(all_refs)

            # Download final database state
            self._download_database_state()

            # Save final generation state
            if state is not None:
                self._write_generation_state(
                    ray.get(state.snapshot.remote())
                )

            if interrupted:
                raise KeyboardInterrupt
        finally:
            self._close_database()


    def _create_results_dir(self):
        """Create the results directory once, refusing to reuse an existing path."""
        try:
            self.results_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError(
                f"Results directory already exists: {self.results_dir}\n"
                "Use a different results directory or remove the existing one."
            ) from exc


    def _request_database_cancellation(self, reason: str) -> None:
        """Cancel the run through the database before doing any other cleanup."""
        result = ray.get(self.db.request_cancellation.remote(reason))
        if self.verbose:
            logger.info(
                "Cancellation requested through ProgramDatabase "
                f"({result['active_claims']} active claims)."
            )

    def _wait_for_workers_to_stop(self, all_refs) -> None:
        """Wait for workers to observe database cancellation and exit."""
        pending = list(all_refs)
        while pending:
            done, pending = ray.wait(
                pending,
                num_returns=len(pending),
                timeout=1.0,
            )
            if done:
                ray.get(done)

    def _download_database_state(self) -> None:
        """Publish a restartable database checkpoint into the results directory."""
        database_state = ray.get(self.db.get_database_state.remote())
        for program in database_state["programs"]:
            program_id = program["id"]
            zip_path = self.results_dir / f"{program_id}.zip"
            if not zip_path.exists():
                program_zip = ray.get(self.db.get_zip_bytes.remote(program_id))
                self._write_bytes_atomically(zip_path, program_zip)

        self._write_bytes_atomically(
            self.results_dir / "database_state.json",
            json.dumps(database_state, indent=2, sort_keys=True).encode("utf-8"),
        )

    def _close_database(self) -> None:
        """Close the database actor."""
        try:
            result = ray.get(self.db.close.remote())
            if self.verbose:
                logger.info(f"ProgramDatabase closed={result.get('closed')}")
        except Exception as exc:
            logger.error(f"Failed to close ProgramDatabase cleanly: {exc}")

    def _write_generation_state(self, state_snapshot: dict) -> None:
        """Write generation and token accounting as one atomic snapshot."""
        state_file = self.results_dir / "evogen_state.json"
        self._write_bytes_atomically(
            state_file,
            json.dumps(state_snapshot, indent=2, sort_keys=True).encode("utf-8"),
        )

    def _write_bytes_atomically(self, path: Path, contents: bytes) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp_path.write_bytes(contents)
        temp_path.replace(path)

    def _run_generation_0(self):
        """Setup and run generation 0 to initialize the database."""
        if self.verbose:
            logger.info("Reading initial evolvable files from %s", self.project_dir)

        try:
            initial_files = {
                spec.file: (self.project_dir / spec.file).read_text(encoding="utf-8")
                for spec in self.evo_config.evolvable_files
            }
        except Exception as e:
            raise ValueError(f"Could not read initial evolvable files: {e}")

        # Zip the project directory to use as parent_zip_bytes for generation 0
        # Data dirs are excluded here; they are staged separately via the object store.
        parent_zip_bytes = zip_dir_to_bytes(self.project_dir, exclude_dirs=self.backend_config.data_dirs)

        # Run the evaluation code using the Ray backend.
        results, rtime, result_zip_bytes = self.backend.run_job(
            parent_zip_bytes=parent_zip_bytes,
            file_replacements={},
        )

        if results.get('correct'):
            combined :float = results.get("combined_score")
            db_program = Program(
                id=str(uuid.uuid4()),
                files=initial_files,
                parent_id=None,
                model_id="initial",
                generation=0,
                correct=True,
                combined_score=combined,
                compute_time=rtime,
            )

            ray.get(
                self.db.add_initial.remote(
                    db_program,
                    result_zip_bytes,
                    verbose=True,
                )
            )

            zip_path = self.results_dir / f"{db_program.id}.zip"
            zip_path.write_bytes(result_zip_bytes)
        else:
            raise ValueError("Initial program is not correct. Please fix the initial program and try again.")
    
