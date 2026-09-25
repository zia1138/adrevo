import textwrap

from adrevo.config import AdrevoConfig, EvolvableFile, ModelSpec
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.providers.openai import OpenAIProvider


SYSTEM_MSG = textwrap.dedent("""\
    Construct a discrete real-valued function f on [-1/4, 1/4] that minimizes

        C3 = 2 * n * max(abs(convolve(f, f))) / sum(f)^2.

    Running `evo/main.py` must write `autocorrelation.json` in its working
    directory with the JSON shape `{"heights": [...]}`. `heights` must be a
    non-empty one-dimensional list of finite real numbers whose absolute values
    do not exceed 1e10 and whose discretized integral is nonzero. The trusted
    evaluator independently computes C3. Lower `combined_score` is better; aim
    to beat 1.4556427953745406.

    Replace complete candidate files; there are no protected EVOLVE-BLOCK
    regions. The evaluation limit is 70 seconds. Use deterministic random
    seeds. Candidate dependencies belong in `evo/pyproject.toml`.
""")

STRATEGIES = (
    "Optimize the sampled heights directly with a scale-invariant objective and deterministic restarts.",
    "Try a structured analytic family and tune both its parameters and discretization size.",
    "Use a substantially different global-search and local-refinement algorithm from the parent.",
    "Make the smallest evidence-driven change likely to decrease the score.",
)


def build_evo_models() -> list[ModelSpec]:
    cerebras_provider = CerebrasProvider()
    openai_provider = OpenAIProvider()
    return [
        ModelSpec(
            model_id="gpt-oss-120b-medium",
            model=CerebrasModel("gpt-oss-120b", provider=cerebras_provider),
            settings=CerebrasModelSettings(openai_reasoning_effort="medium"),
            input_token_cost=0.35,
            output_token_cost=0.75,
            max_model_turns=10,
        ),
        ModelSpec(
            model_id="gpt-5.6-terra",
            model=OpenAIResponsesModel("gpt-5.6-terra", provider=openai_provider),
            settings=OpenAIResponsesModelSettings(
                openai_reasoning_effort="medium",
                openai_service_tier="flex",
            ),
            input_token_cost=1.0,
            output_token_cost=6.0,
            max_concurrent_leases=1,
            max_model_turns=4,
        ),
    ]


def get_adrevo_config() -> AdrevoConfig:
    return AdrevoConfig(
        num_agent_workers=5,
        task_sys_msg=SYSTEM_MSG,
        build_evo_models=build_evo_models,
        evolvable_files=(
            EvolvableFile("evo/main.py", "python"),
            EvolvableFile("evo/pyproject.toml", "toml"),
        ),
        strategies=STRATEGIES,
        pr_no_strategy=0.2,
        pr_strategies=(0.4, 0.2, 0.1, 0.1),
        max_cost=5.0,
        evaluator_timeout_sec=100,
        maximize_combined_score=False,
    )
