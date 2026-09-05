import textwrap

from adrevo.config import AdrevoConfig, ModelSpec
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

SYSTEM_MSG = textwrap.dedent("""\
    SETTING:
    You are an expert computational geometer and optimization specialist with deep expertise in the
    Heilbronn triangle problem - a fundamental challenge in discrete geometry first posed by Hans Heilbronn in 1957.
    This problem asks for the optimal placement of n points within a convex region of unit area to maximize the area of the smallest
    triangle formed by any three of these points.

    PROBLEM SPECIFICATION:
    Design and implement a constructor function that generates an optimal arrangement of exactly 13 points
    within or on the boundary of a unit-area convex region. The solution must:
    - Place all 13 points within or on a convex boundary
    - Maximize the minimum triangle area among all C(13,3) = 286 possible triangles
    - Return deterministic, reproducible results
    - Execute efficiently within computational constraints
    - Try to reach the benchmark area of 0.030936889034895654, which is the best known solution for 13 points.                             

    PERFORMANCE METRICS:
    1. min_area_normalized: (Area of smallest triangle) / (Area of convex hull) [PRIMARY - MAXIMIZE]
    2. eval_time: Execution time in seconds [EFFICIENCY - secondary priority]

    TECHNICAL REQUIREMENTS:
    - Determinism: Use fixed random seeds if employing stochastic methods for reproducibility
    - Error handling: Graceful handling of optimization failures or infeasible configurations
    - Computational efficiency is also important. The program must run within 2 minutes.                                 

    NOTE: heilbronn_convex13() is the main entry point of the code.
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
        evaluator_timeout_sec=120,
    )
