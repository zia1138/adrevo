import textwrap

from adrevo.config import AdrevoConfig, ModelSpec
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert mathematician specializing in circle packing problems and computational geometry. The best known result for the sum of radii when packing 26 circles in a unit square is 2.635.

    Key insights to explore:
    1. The optimal arrangement likely involves variable-sized circles
    2. A pure hexagonal arrangement may not be optimal due to edge effects
    3. The densest known circle packings often use a hybrid approach
    4. The optimization routine is critically important - simple physics-based models with carefully tuned parameters
    5. Consider strategic placement of circles at square corners and edges
    6. Adjusting the pattern to place larger circles at the center and smaller at the edges
    7. The math literature suggests special arrangements for specific values of n
    8. You can use the scipy optimize package (e.g. LP or SLSQP) to optimize the radii given center locations and constraints
    9. Computational effciency is also important. The program must run within 1 minute.
                             
    Be creative and try to find a new solution better than the best known result.
    
    Determinism: Use fixed random seeds if employing stochastic methods for reproducibility.
  
    NOTE: run_packing() is the main entry point of the code.
""")


strategies = ("Produce an approach that uses the scipy optimize package.",
              "Use a different underlying algorithm from the parent program.",
              "Make your solution verbose adding comments and rationale for each step.",              
              "Make as minimal a change as possible to improve the parent program.")
pr_no_strategy = 0.25
pr_strategies = (0.5, 0.15, 0.05, 0.05)

def test_openai():
    """Code you can put into a python REPL to test of thinking is working."""
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai import Agent
    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import model_request_sync
    provider = OpenAIProvider()
    # model string here can be any of the models here: 
    model=OpenAIResponsesModel("gpt-5.4-mini", provider=provider)
    settings=OpenAIResponsesModelSettings(openai_reasoning_effort="high", openai_reasoning_summary='detailed', openai_service_tier="flex") # <- this controls reasoning effort
    agent = Agent(model=model)    
    response = model_request_sync(model, [ModelRequest.user_text_prompt('5 = 2x - 3, solve for x')], model_settings=settings)
    print(response.parts) # this should have a ThinkingPart
    response = agent.run_sync('5 = 2x - 3, solve for x', model_settings=settings)
    print(response.usage)
    print(response.response.parts) # this should have ThinkingPart


def build_evo_models() -> list[ModelSpec]:
    provider = OpenAIProvider()
    return [
        ModelSpec(
            model_id="gpt-5.4-mini",
            model=OpenAIResponsesModel("gpt-5.4-mini", provider=provider),
            settings=OpenAIResponsesModelSettings(openai_reasoning_effort="high", openai_reasoning_summary='detailed', openai_service_tier="flex"),
            input_token_cost=0.75, # per 1M tokens
            output_token_cost=4.50, # per 1M tokens
        )
    ]


def get_adrevo_config() -> AdrevoConfig:
    return AdrevoConfig(
        task_sys_msg=SYSTEM_MSG,
        build_evo_models=build_evo_models,
        strategies=strategies,
        pr_no_strategy=pr_no_strategy,
        pr_strategies=pr_strategies,
        max_cost=5.0,
        evaluator_timeout_sec=60,
    )
