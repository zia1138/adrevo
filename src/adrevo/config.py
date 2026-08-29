from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Callable, Dict, Any
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings


@dataclass(frozen=True)
class ModelSpec:
    """
    Specification for a model used in the evolution process.

    Attributes:
        model_id: Unique identifier for the model.
        model: The model instance (e.g., a fully configured Pydantic AI model https://pydantic.dev/docs/ai/models/overview).
        settings: Settings for the model (e.g., reasoning effort, service tier https://pydantic.dev/docs/ai/models/overview).
        input_token_cost: Cost per 1M input tokens for the model.
        output_token_cost: Cost per 1M output tokens for the model.
        max_concurrent_leases: Maximum number of concurrent workers that can use the model.
        max_model_turns: Maximum number of turns the model can take in a multi-turn interaction.
    """
    model_id: str
    model: Model
    settings: ModelSettings
    input_token_cost: float = 0.0  # per 1M tokens
    output_token_cost: float = 0.0  # per 1M tokens
    max_concurrent_leases: int = 1000
    max_model_turns: int = 8

@dataclass(frozen=True)
class AdrevoConfig:
    """
    Configuration for adrevo run.

    Attributes:
        build_evo_models: Callable that returns a list of ModelSpec objects, see config.py for examples.
        task_sys_msg: Optional system message for the task.
        num_agent_workers: Number of agent workers to use.
        max_generations: Maximum number of program generations to evolve.
        lang_identifier: Language used for any LLM code blocks (e.g. ```python ... ```).
        evo_file: Name of the file to use for rewriting for optimization/improvement/discovery. Default is evo/main.py.
        evaluate_file: Name of the file to use for evaluation. Default is evaluate.py.
        use_probe: Whether to run diagnostic probing during the multi-turn loop.
        dl_evostate_freq: Frequency (in seconds) to download evo state from workers and database.
        model_wait_poll_sec: Frequency (in seconds) to poll for model availability (limited by leases).
        code_update_failures_before_diagnostics: Number of consecutive code update failures before running diagnostics.
        pr_package_install: Probability of installing a package in the environment.
        strategies: Tuple of strategy names to use for evolution.
        pr_no_strategy: Probability of not using any strategy.
        pr_strategies: Tuple of probabilities for each strategy in strategies.
        max_cost: Maximum token cost allowed for evolution.
        backtrack_steps: Number of ancestors to move upward on max-rank failure.
    """
    build_evo_models: Callable[[], list[ModelSpec]]
    task_sys_msg: str =  ""
    num_agent_workers: int = 4
    max_generations: int = 500
    lang_identifier: str = "python"  # TODO: Add support for more languages.
    evo_file: str = "evo/main.py"
    evaluate_file: str = "evaluate.py"  # TODO: Remove hard coding of evaluate.py in codebase.
    use_probe: bool = True
    dl_evostate_freq: float = 30
    model_wait_poll_sec: float = 2.0
    code_update_failures_before_diagnostics: int = 2
    pr_package_install: float = 0.01
    strategies: tuple = ()
    pr_no_strategy: float = 1.0
    pr_strategies: tuple = ()
    max_cost: float = float('inf')  # limit token cost in evolution
    backtrack_steps: int = 1

def validate_adrevo(cfg: AdrevoConfig) -> None:
    """Validate the AdrevoConfig object."""
    if not isinstance(cfg, AdrevoConfig):
        raise TypeError(f"Expected AdrevoConfig, got {type(cfg).__name__}")
    if not callable(cfg.build_evo_models):
        raise ValueError("AdrevoConfig.build_evo_models must be callable")
    _validate_model_specs(cfg.build_evo_models())
    if not isinstance(cfg.task_sys_msg, str):
        raise ValueError("AdrevoConfig.task_sys_msg must be a string")
    if not _is_positive_int(cfg.num_agent_workers):
        raise ValueError("AdrevoConfig.num_agent_workers must be an integer >= 1")
    if not _is_positive_int(cfg.max_generations):
        raise ValueError("AdrevoConfig.max_generations must be an integer >= 1")
    if not _is_positive_int(cfg.backtrack_steps):
        raise ValueError("AdrevoConfig.backtrack_steps must be an integer >= 1")
    if not isinstance(cfg.use_probe, bool):
        raise ValueError("AdrevoConfig.use_probe must be a boolean")
    if not isinstance(cfg.lang_identifier, str) or not cfg.lang_identifier.strip():
        raise ValueError("AdrevoConfig.lang_identifier must be a non-empty string")
    _validate_relative_path(cfg.evo_file, "AdrevoConfig.evo_file")
    _validate_relative_path(cfg.evaluate_file, "AdrevoConfig.evaluate_file")
    if not isinstance(cfg.dl_evostate_freq, (int, float)) or cfg.dl_evostate_freq <= 0:
        raise ValueError("AdrevoConfig.dl_evostate_freq must be a positive number")
    if (
        not isinstance(cfg.model_wait_poll_sec, (int, float))
        or isinstance(cfg.model_wait_poll_sec, bool)
        or cfg.model_wait_poll_sec <= 0
    ):
        raise ValueError("AdrevoConfig.model_wait_poll_sec must be a positive number")
    if not isinstance(cfg.strategies, tuple):
        raise ValueError("AdrevoConfig.strategies must be a tuple of non-empty strings")
    for idx, strategy in enumerate(cfg.strategies):
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError(
                f"AdrevoConfig.strategies[{idx}] must be a non-empty string"
            )
    if not isinstance(cfg.pr_strategies, tuple):
        raise ValueError("AdrevoConfig.pr_strategies must be a tuple of probabilities")
    if len(cfg.pr_strategies) != len(cfg.strategies):
        raise ValueError(
            "AdrevoConfig.pr_strategies must have one probability per strategy"
        )
    if (
        not isinstance(cfg.pr_no_strategy, (int, float))
        or isinstance(cfg.pr_no_strategy, bool)
        or cfg.pr_no_strategy < 0
    ):
        raise ValueError("AdrevoConfig.pr_no_strategy must be a non-negative number")
    for idx, probability in enumerate(cfg.pr_strategies):
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or probability < 0
        ):
            raise ValueError(
                f"AdrevoConfig.pr_strategies[{idx}] must be a non-negative number"
            )
    if abs(cfg.pr_no_strategy + sum(cfg.pr_strategies) - 1.0) > 1e-9:
        raise ValueError(
            "AdrevoConfig.pr_no_strategy + sum(pr_strategies) must equal 1.0"
        )

@dataclass(frozen=True)
class BackendConfig:
    """
    Configuration for trusted evaluator execution with uv.

    Attributes:
        timeout_sec: Optional timeout in seconds for script execution. If None, no timeout is applied
        data_dirs: Relative paths within the project directory to treat as immutable data.
            These are excluded from the code zip and staged once per node via symlinks.
            Example: ("valid_instances",)
    """
    timeout_sec: int = 10 * 60
    data_dirs: tuple = ()  # relative paths to exclude from code zip, e.g. ("valid_instances",)


def validate_backend(cfg: BackendConfig) -> None:
    """Validate the BackendConfig object."""
    if not isinstance(cfg, BackendConfig):
        raise TypeError(f"Expected BackendConfig, got {type(cfg).__name__}")
    if cfg.timeout_sec is not None and not _is_positive_int(cfg.timeout_sec):
        raise ValueError("BackendConfig.timeout_sec must be None or an integer >= 1")
    if not isinstance(cfg.data_dirs, (tuple, list)):
        raise ValueError("BackendConfig.data_dirs must be a tuple or list of relative paths")
    for data_dir in cfg.data_dirs:
        _validate_relative_path(data_dir, "BackendConfig.data_dirs")


def _validate_relative_path(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative, got absolute path: {value}")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{field_name} must not contain '..': {value}")


def _validate_model_specs(model_specs: list[ModelSpec]) -> None:
    if not isinstance(model_specs, list):
        raise ValueError("AdrevoConfig.build_evo_models must return a list of ModelSpec")
    if not model_specs:
        raise ValueError("AdrevoConfig.build_evo_models must return at least one ModelSpec")

    model_ids = [spec.model_id for spec in model_specs if isinstance(spec, ModelSpec)]
    duplicate_model_ids = {
        model_id for model_id in model_ids if model_ids.count(model_id) > 1
    }
    if duplicate_model_ids:
        raise ValueError(
            "Duplicate ModelSpec.model_id values: "
            + ", ".join(sorted(duplicate_model_ids))
        )

    for idx, spec in enumerate(model_specs):
        if not isinstance(spec, ModelSpec):
            raise ValueError(
                f"AdrevoConfig.build_evo_models()[{idx}] must be a ModelSpec"
            )
        if not isinstance(spec.model_id, str) or not spec.model_id.strip():
            raise ValueError(f"ModelSpec.model_id at index {idx} must be a non-empty string")
        if (
            not isinstance(spec.input_token_cost, (int, float))
            or isinstance(spec.input_token_cost, bool)
            or not math.isfinite(spec.input_token_cost)
            or spec.input_token_cost < 0
        ):
            raise ValueError(
                f"ModelSpec.input_token_cost for {spec.model_id!r} must be a finite non-negative number"
            )
        if (
            not isinstance(spec.output_token_cost, (int, float))
            or isinstance(spec.output_token_cost, bool)
            or not math.isfinite(spec.output_token_cost)
            or spec.output_token_cost < 0
        ):
            raise ValueError(
                f"ModelSpec.output_token_cost for {spec.model_id!r} must be a finite non-negative number"
            )
        if not _is_positive_int(spec.max_concurrent_leases):
            raise ValueError(
                f"ModelSpec.max_concurrent_leases for {spec.model_id!r} must be an integer >= 1"
            )
        if not _is_positive_int(spec.max_model_turns):
            raise ValueError(
                f"ModelSpec.max_model_turns for {spec.model_id!r} must be an integer >= 1"
            )

def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1
