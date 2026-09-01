#!/usr/bin/env python3
import contextlib
import concurrent.futures
import typer
import runpy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import questionary
import ray

from adrevo.config import AdrevoConfig
from adrevo.config import BackendConfig
from adrevo.config import validate_adrevo
from adrevo.config import validate_backend
from adrevo.driver import AdrevoDriver

app = typer.Typer()


@dataclass(frozen=True)
class ProjectRun:
    project_dir: Path
    config_file: Path
    results_dir: Path
    adrevo_cfg: AdrevoConfig
    backend_cfg: BackendConfig


def load_config_file(config_file: Path) -> dict:
    """
    We use a code as config system so runpy runs config.py to populate the namespace.
    We expect config.py to define get_adrevo_config() and get_backend_config().
    The rest is up to the user.
    """
    if not config_file.exists():
        raise typer.BadParameter(f"Config file not found: {config_file}")
    if not config_file.is_file():
        raise typer.BadParameter(f"Config path must be a file: {config_file}")
    return runpy.run_path(str(config_file))


def _validate_project_dir(project: str) -> Path:
    p = Path(project)
    if not p.exists():
        raise typer.BadParameter(f"Project path not found: {project}")
    if not p.is_dir():
        raise typer.BadParameter(f"Project path must be a directory: {project}")
    return p


def _discover_config_files(project_dir: Path) -> list[Path]:
    candidates: list[Path] = []

    default_config = project_dir / "config.py"
    if default_config.is_file():
        candidates.append(default_config)

    variant_configs = [
        path
        for path in project_dir.glob("config_*.py")
        if path.is_file() and not path.name.startswith(".") and path.name != "__init__.py"
    ]
    candidates.extend(variant_configs)
    return sorted(candidates, key=lambda path: (-path.stat().st_mtime, path.name))


def _resolve_explicit_config(
    project_dir: Path,
    config: str,
    allow_absolute_config: bool,
) -> Path:
    config_path = Path(config)
    if config_path.is_absolute() and not allow_absolute_config:
        raise typer.BadParameter(
            "--config must be a filename or relative path in run-folder, not an absolute path."
        )
    if not config_path.is_absolute():
        config_path = project_dir / config_path
    config_path = config_path.resolve()
    if not config_path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise typer.BadParameter(f"Config path must be a file: {config_path}")
    return config_path


def _prompt_for_config(project_dir: Path, config_files: list[Path]) -> Path:
    choices = [
        questionary.Choice(title=config_file.name, value=config_file)
        for config_file in config_files
    ]
    try:
        selection = questionary.select(
            f"Select config for {project_dir}",
            choices=choices,
            use_shortcuts=True,
            use_indicator=True,
        ).ask()
    except KeyboardInterrupt as exc:
        raise typer.Abort() from exc

    if selection is None:
        raise typer.Abort()
    return selection


def _resolve_project_config(
    project_dir: Path,
    config: str | None = None,
    non_interactive: bool = False,
    allow_absolute_config: bool = True,
) -> Path:
    if config is not None:
        return _resolve_explicit_config(
            project_dir,
            config,
            allow_absolute_config=allow_absolute_config,
        )

    candidates = _discover_config_files(project_dir)
    if not candidates:
        raise typer.BadParameter(
            f"No config files found in {project_dir}. Expected config.py or config_*.py."
        )
    if len(candidates) == 1:
        return candidates[0]
    if non_interactive:
        choices = ", ".join(path.name for path in candidates)
        raise typer.BadParameter(
            f"Multiple config files found in {project_dir}: {choices}. "
            "Use --config to select one."
        )
    return _prompt_for_config(project_dir, candidates)


def _load_project_configs(config_file: Path, project_dir: Path) -> tuple[AdrevoConfig, BackendConfig]:
    ns = load_config_file(config_file)

    get_adrevo_config = ns.get("get_adrevo_config")
    get_backend_config = ns.get("get_backend_config")
    if not callable(get_adrevo_config):
        raise typer.BadParameter(
            f"{config_file.name} in {project_dir} must define get_adrevo_config()"
        )
    if not callable(get_backend_config):
        raise typer.BadParameter(
            f"{config_file.name} in {project_dir} must define get_backend_config()"
        )

    try:
        adrevo_cfg: AdrevoConfig = get_adrevo_config()
    except Exception as exc:
        raise typer.BadParameter(
            f"get_adrevo_config() in {config_file.name} failed: {exc}"
        ) from exc

    try:
        backend_cfg: BackendConfig = get_backend_config()
    except Exception as exc:
        raise typer.BadParameter(
            f"get_backend_config() in {config_file.name} failed: {exc}"
        ) from exc
    if not isinstance(adrevo_cfg, AdrevoConfig):
        raise typer.BadParameter(
            f"get_adrevo_config() in {config_file.name} must return AdrevoConfig, "
            f"got {type(adrevo_cfg).__name__}"
        )
    if not isinstance(backend_cfg, BackendConfig):
        raise typer.BadParameter(
            f"get_backend_config() in {config_file.name} must return BackendConfig, "
            f"got {type(backend_cfg).__name__}"
        )
    try:
        validate_adrevo(adrevo_cfg)
        validate_backend(backend_cfg)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"Invalid config in {config_file.name}: {exc}") from exc
    return adrevo_cfg, backend_cfg


def _validate_port(ctx: typer.Context, param: typer.CallbackParam, value: int) -> int:
    if isinstance(value, bool) or not (1 <= value <= 65535):
        raise typer.BadParameter("Port must be in range 1-65535")
    return value


def _validate_max_concurrent(ctx: typer.Context, param: typer.CallbackParam, value: int) -> int:
    if isinstance(value, bool) or value < 1:
        raise typer.BadParameter("max-concurrent must be >= 1")
    return value


def _validate_initial_project_inputs(
    project_dir: Path,
    adrevo_cfg: AdrevoConfig,
    backend_cfg: BackendConfig,
) -> None:
    """Validate files and directories needed before a Ray session starts."""
    for evolvable_file in adrevo_cfg.evolvable_files:
        file_path = project_dir / evolvable_file.file
        if not file_path.is_file():
            raise typer.BadParameter(
                f"Initial evolvable file not found: {file_path}"
            )

    evaluate_path = project_dir / adrevo_cfg.evaluate_file
    if not evaluate_path.is_file():
        raise typer.BadParameter(f"Evaluation file not found: {evaluate_path}")

    for data_dir in backend_cfg.data_dirs:
        data_path = project_dir / data_dir
        if not data_path.is_dir():
            raise typer.BadParameter(f"Data directory not found: {data_path}")


def _normalize_nonempty(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    return s or None


@contextlib.contextmanager
def _ray_session(ray_address: str | None, ray_ip: str | None, ray_port: int, ray_debug: bool):
    """Initialize Ray, yield, then shut down."""
    env_vars: dict[str, str] = {}
    if ray_debug:
        env_vars["RAY_DEBUG"] = "1"
        env_vars["RAY_DEBUG_POST_MORTEM"] = "1"

    runtime_env = {"env_vars": env_vars} if env_vars else None

    ray_address = _normalize_nonempty(ray_address)
    ray_ip = _normalize_nonempty(ray_ip)

    if ray_address and ray_ip:
        raise typer.BadParameter("Use either --ray-address or --ray-ip/--ray-port, not both.")

    try:
        if ray_address:
            if not ray_address.startswith("ray://"):
                raise typer.BadParameter(
                    "Ray address must start with ray://, e.g. ray://127.0.0.1:10001"
                )
            ray.init(address=ray_address, runtime_env=runtime_env)
        elif ray_ip:
            ray.init(address=f"ray://{ray_ip}:{ray_port}", runtime_env=runtime_env)
        else:
            ray.init(runtime_env=runtime_env)
    except typer.BadParameter:
        raise
    except Exception as exc:
        raise typer.BadParameter(f"Could not initialize Ray: {exc}") from exc

    try:
        yield
    finally:
        ray.shutdown()


def _resolve_results_base(results_dir: Path | None) -> Path:
    if results_dir is not None:
        if results_dir.exists() and not results_dir.is_dir():
            raise typer.BadParameter(f"Results base path must be a directory: {results_dir}")
        return results_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"results_{timestamp}").resolve()


def _resolve_results_dir(project_dir: Path, results_base: Path) -> Path:
    """Resolve the results directory for a project.

    Always appends the project name as a subdirectory.
    """
    return (results_base / project_dir.name).resolve()


def _prepare_project_run(
    project_dir: str | Path,
    results_base: Path,
    config: str | None = None,
    non_interactive: bool = False,
    allow_absolute_config: bool = True,
) -> ProjectRun:
    project_path = _validate_project_dir(str(project_dir)).resolve()
    config_file = _resolve_project_config(
        project_path,
        config=config,
        non_interactive=non_interactive,
        allow_absolute_config=allow_absolute_config,
    )
    adrevo_cfg, backend_cfg = _load_project_configs(config_file, project_path)
    _validate_initial_project_inputs(project_path, adrevo_cfg, backend_cfg)
    return ProjectRun(
        project_dir=project_path,
        config_file=config_file,
        results_dir=_resolve_results_dir(project_path, results_base),
        adrevo_cfg=adrevo_cfg,
        backend_cfg=backend_cfg,
    )


def _check_results_dirs_available(project_runs: list[ProjectRun]) -> None:
    """Fail before starting Ray if any run would reuse an existing results dir."""
    for project_run in project_runs:
        if project_run.results_dir.exists():
            raise typer.BadParameter(
                f"Results directory already exists for {project_run.project_dir}: {project_run.results_dir}\n"
                "Please specify a different --results-dir or remove the existing one."
            )


def _check_results_dir_resumable(project_run: ProjectRun) -> None:
    """Fail before starting Ray unless a complete resume checkpoint is present."""
    results_dir = project_run.results_dir
    if not results_dir.is_dir():
        raise typer.BadParameter(
            f"Results directory does not exist for {project_run.project_dir}: {results_dir}"
        )

    required_files = (
        "database_state.json",
        "evogen_state.json",
    )
    missing_files = [
        filename for filename in required_files
        if not (results_dir / filename).is_file()
    ]
    if missing_files:
        raise typer.BadParameter(
            f"Results directory is not resumable: {results_dir}\n"
            "Missing checkpoint files: " + ", ".join(missing_files)
        )


def _run_single_project(
    project_run: ProjectRun,
    verbose: bool = False,
    resume: bool = False,
) -> None:
    """Run evolution for a prepared project."""
    adrevo_driver = AdrevoDriver(
        project_run.adrevo_cfg,
        project_run.backend_cfg,
        str(project_run.project_dir),
        results_dir=project_run.results_dir,
        verbose=verbose,
        resume_from=project_run.results_dir if resume else None,
    )
    adrevo_driver.run_ray()


@app.command()
def run(
    project_dir: str = typer.Argument(
        ...,
        help="Path to project/data folder containing config.py or config_*.py files.",
    ),
    config: str | None = typer.Option(
        None,
        help="Optional config file to use. Relative paths are resolved from the project directory.",
    ),
    non_interactive: bool = typer.Option(
        False,
        help="Disable config selection prompts and fail if multiple config files are present.",
    ),
    results_dir: str | None = typer.Option(
        None,
        help="Base directory for results. Results are saved to <results_dir>/<project_name>/.",
    ),
    verbose: bool = typer.Option(False, help="Enable verbose logging."),
    ray_debug: bool = typer.Option(False, help="Enable Ray debug env vars (RAY_DEBUG, RAY_DEBUG_POST_MORTEM)"),
    ray_address: str | None = typer.Option(
        None,
        help=(
            "Optional Ray Client address (ray://<host>:<port>). "
            "If not provided, a local Ray runtime is started."
        ),
    ),
    ray_ip: str | None = typer.Option(
        None,
        help="Alternative to --ray-address: specify Ray head IP/host.",
    ),
    ray_port: int = typer.Option(
        10001,
        callback=_validate_port,
        help="Ray Client port (default: 10001). Only used with --ray-ip.",
        show_default=True,
    ),
):
    """
    Run adrevo using a selected config file in the specified project directory.
    """
    results_base = _resolve_results_base(Path(results_dir) if results_dir else None)
    project_run = _prepare_project_run(
        project_dir,
        results_base,
        config=config,
        non_interactive=non_interactive,
        allow_absolute_config=True,
    )
    _check_results_dirs_available([project_run])

    with _ray_session(ray_address, ray_ip, ray_port, ray_debug):
        _run_single_project(project_run, verbose=verbose)


@app.command()
def resume(
    project_dir: str = typer.Argument(
        ...,
        help="Path to the original project/data folder.",
    ),
    results_dir: str = typer.Option(
        ...,
        help="Existing results base containing <results_dir>/<project_name>/.",
    ),
    config: str | None = typer.Option(
        None,
        help="Optional config file to use. Relative paths are resolved from the project directory.",
    ),
    non_interactive: bool = typer.Option(
        False,
        help="Disable config selection prompts and fail if multiple config files are present.",
    ),
    verbose: bool = typer.Option(False, help="Enable verbose logging."),
    ray_debug: bool = typer.Option(
        False,
        help="Enable Ray debug env vars (RAY_DEBUG, RAY_DEBUG_POST_MORTEM)",
    ),
    ray_address: str | None = typer.Option(
        None,
        help=(
            "Optional Ray Client address (ray://<host>:<port>). "
            "If not provided, a local Ray runtime is started."
        ),
    ),
    ray_ip: str | None = typer.Option(
        None,
        help="Alternative to --ray-address: specify Ray head IP/host.",
    ),
    ray_port: int = typer.Option(
        10001,
        callback=_validate_port,
        help="Ray Client port (default: 10001). Only used with --ray-ip.",
        show_default=True,
    ),
):
    """Resume a prior run from its last completed checkpoint."""
    results_base = _resolve_results_base(Path(results_dir))
    project_run = _prepare_project_run(
        project_dir,
        results_base,
        config=config,
        non_interactive=non_interactive,
        allow_absolute_config=True,
    )
    _check_results_dir_resumable(project_run)

    with _ray_session(ray_address, ray_ip, ray_port, ray_debug):
        _run_single_project(project_run, verbose=verbose, resume=True)


@app.command()
def run_folder(
    project_dirs: list[str] = typer.Argument(..., help="One or more project directories"),
    config: str | None = typer.Option(
        None,
        help="Optional config filename to use inside every project directory.",
    ),
    non_interactive: bool = typer.Option(
        False,
        help="Disable config selection prompts and fail if multiple config files are present.",
    ),
    results_dir: str | None = typer.Option(
        None,
        help="Base directory for results. Each project saves to <results_dir>/<project_name>/.",
    ),
    max_concurrent: int = typer.Option(
        1,
        callback=_validate_max_concurrent,
        help="Max projects running simultaneously. 1 = sequential.",
    ),
    verbose: bool = typer.Option(False, help="Enable verbose logging."),
    ray_debug: bool = typer.Option(False, help="Enable Ray debug env vars"),
    ray_address: str | None = typer.Option(
        None,
        help="Optional Ray Client address (ray://<host>:<port>).",
    ),
    ray_ip: str | None = typer.Option(
        None,
        help="Alternative to --ray-address: specify Ray head IP/host.",
    ),
    ray_port: int = typer.Option(
        10001,
        callback=_validate_port,
        help="Ray Client port (default: 10001). Only used with --ray-ip.",
        show_default=True,
    ),
):
    """
    Run adrevo as a folder run on multiple projects. Projects run sequentially by default;
    use --max-concurrent to run multiple projects in parallel.
    """
    # Skip file arguments; missing paths fail during project validation below.
    project_paths: list[Path] = []
    for proj in project_dirs:
        project_path = Path(proj)
        if project_path.is_file():
            typer.echo(f"Skipping (not a directory): {proj}")
        else:
            project_paths.append(project_path.resolve())

    if not project_paths:
        typer.echo("No valid project directories to process.")
        raise typer.Exit(code=0)

    # Check for duplicate project names (leaf directory names must be unique
    # since results directories are keyed by project name).
    seen_names: dict[str, str] = {}
    for project_path in project_paths:
        name = project_path.name
        if name in seen_names:
            raise typer.BadParameter(
                f"Duplicate project name '{name}': {seen_names[name]} and {project_path}\n"
                "Project leaf directory names must be unique to avoid results directory collisions."
            )
        seen_names[name] = str(project_path)

    results_base = _resolve_results_base(Path(results_dir) if results_dir else None)
    project_runs: list[ProjectRun] = []
    for project_path in project_paths:
        project_runs.append(
            _prepare_project_run(
                project_path,
                results_base,
                config=config,
                non_interactive=non_interactive,
                allow_absolute_config=False,
            )
        )

    _check_results_dirs_available(project_runs)

    typer.echo(f"Folder run: {len(project_paths)} projects, max_concurrent={max_concurrent}")

    results: dict[str, tuple[bool, str | None]] = {}

    with _ray_session(ray_address, ray_ip, ray_port, ray_debug):
        if max_concurrent <= 1:
            for project_run in project_runs:
                project_label = str(project_run.project_dir)
                typer.echo(f"\n{'='*60}\nRunning: {project_label}\n{'='*60}")
                try:
                    _run_single_project(
                        project_run,
                        verbose=verbose,
                    )
                    results[project_label] = (True, None)
                except Exception as e:
                    typer.echo(f"FAILED: {project_label} -- {e}", err=True)
                    results[project_label] = (False, str(e))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
                futures = {
                    pool.submit(
                        _run_single_project,
                        project_run,
                        verbose,
                    ): project_run
                    for project_run in project_runs
                }
                for future in concurrent.futures.as_completed(futures):
                    project_run = futures[future]
                    project_label = str(project_run.project_dir)
                    try:
                        future.result()
                        results[project_label] = (True, None)
                        typer.echo(f"DONE: {project_label}")
                    except Exception as e:
                        typer.echo(f"FAILED: {project_label} -- {e}", err=True)
                        results[project_label] = (False, str(e))

    # Print summary
    typer.echo("RUN SUMMARY:\n")
    for proj, (ok, err) in results.items():
        status = "OK" if ok else f"FAILED: {err}"
        typer.echo(f"  {proj}: {status}")

    failed = sum(1 for ok, _ in results.values() if not ok)
    if failed:
        raise typer.Exit(code=1)
