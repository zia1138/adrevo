import textwrap

from adrevo.config import AdrevoConfig, BackendConfig, ModelSpec
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

SYSTEM_MSG = textwrap.dedent("""\
    SETTING:
    You are an expert computational geometer and optimization specialist with deep expertise in circle
    packing problems, geometric optimization algorithms, and constraint satisfaction.
    Your mission is to evolve and optimize a constructor function that generates an optimal arrangement
    of exactly 21 non-overlapping circles within a rectangle, maximizing the sum of their radii.

    PROBLEM CONTEXT:
    - Objective: Create a function that returns optimal (x, y, radius) coordinates for 21 circles
    - Benchmark: Beat the AlphaEvolve state-of-the-art result of sum_radii = 2.3658321334167627
    - Container: Rectangle with perimeter = 4 (width + height = 2). You may choose optimal width/height ratio
    - Constraints:
      * All circles must be fully contained within rectangle boundaries
      * No circle overlaps (distance between centers >= sum of their radii)
      * Exactly 21 circles required
      * All radii must be positive

    PERFORMANCE METRICS:
    1. sum_radii: Total sum of all 21 circle radii (PRIMARY OBJECTIVE - maximize)
    2. combined_score: sum_radii / 2.3658321334167627 (progress toward beating benchmark)
    3. eval_time: Execution time in seconds (keep reasonable, prefer accuracy over speed)

    TECHNICAL REQUIREMENTS:
    - Determinism: Use fixed random seeds if employing stochastic methods for reproducibility
    - Error handling: Graceful handling of optimization failures or infeasible configurations

    NOTE: circle_packing21() is the main entry point of the code.
""")


strategies = ("You must propose an algorithmic approach that combines global stochastic exploration (for example, simulated annealing, evolutionary search, or basin hopping) with local refinement across multiple random restarts. Do not give only a theoretical bound or a purely exact-optimization method. Include: initialization, objective function, move/proposal mechanism, acceptance rule, restart strategy, refinement step, and stopping criteria.",
              "Produce an approach that uses the non-linear optimization methods in the scipy optimize package.",
              "Use a different underlying algorithm from the parent program.",
              "Make your solution verbose adding comments and rationale for each step.",              
              "Make as minimal a change as possible to improve the parent program.")

pr_no_strategy = 0.2
pr_strategies = (0.5, 0.15, 0.05, 0.05, 0.05)


def build_evo_models() -> list[ModelSpec]:
    cerebras_provider = CerebrasProvider()
    gpt_oss_120b_medium = ModelSpec(
        model_id="gpt-oss-120b-medium",
        model=CerebrasModel("gpt-oss-120b", provider=cerebras_provider),
        settings=CerebrasModelSettings(openai_reasoning_effort="medium"),
        input_token_cost=0.35,
        output_token_cost=0.75,            
        max_concurrent_leases=1000,
        max_model_turns=10,        
    )
    openai_provider = OpenAIProvider()
    openai_gpt_5_6_terra = ModelSpec(
        model_id="gpt-5.6-terra",
        model=OpenAIResponsesModel("gpt-5.6-terra", provider=openai_provider),
        settings=OpenAIResponsesModelSettings(openai_reasoning_effort="medium", openai_service_tier="flex"),
        input_token_cost=1, # per 1M tokens
        output_token_cost=6, # per 1M tokens
        max_concurrent_leases=1,
        max_model_turns=4,
    )    
    return [
        gpt_oss_120b_medium,
        openai_gpt_5_6_terra,
    ]



def get_adrevo_config() -> AdrevoConfig:
    return AdrevoConfig(
        num_agent_workers=5,
        task_sys_msg=SYSTEM_MSG,
        build_evo_models=build_evo_models,
        model_wait_poll_sec=2.0,
        strategies=strategies,
        pr_no_strategy=pr_no_strategy,
        pr_strategies=pr_strategies,
        max_cost=5.0,
    )

def get_backend_config() -> BackendConfig:
    return BackendConfig(timeout_sec=120)
