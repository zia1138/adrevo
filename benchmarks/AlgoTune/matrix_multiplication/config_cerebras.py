from adrevo.config import AdrevoConfig, ModelSpec
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

SYSTEM_MSG = """MatrixMultiplication Task:

Task Description:
Given two matrices A and B, the task is to compute their product C = A · B.
Matrix A is an n x m matrix and matrix B is an m x p matrix, so the resulting product C is an n x p matrix.

Input:
A dictionary with keys:
  - "A": An array representing matrix A.
  - "B": An array representing matrix B.
(The dimensions are such that the number of columns in A equals the number of rows in B.)

Example input:
{
    "A": [
        [1.0, 2.0],
        [3.0, 4.0]
    ],
    "B": [
        [5.0, 6.0],
        [7.0, 8.0]
    ]
}

Output:
A numpy array of shape (n, p) representing the product matrix C, where C = A · B.

Example output:
[
    [19.0, 22.0],
    [43.0, 50.0]
]

Category: matrix_operations"""


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
        max_cost=5.0,
        evaluator_timeout_sec=780,  # 12 runs × 60 seconds, plus evaluation overhead
    )
