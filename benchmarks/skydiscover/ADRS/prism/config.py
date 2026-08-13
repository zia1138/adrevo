import textwrap

from adrevo.config import AdrevoConfig, BackendConfig, ModelSpec
from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert for model placement on GPUs. Your task is to improve a model
    placement algorithm by improving the function named compute_model_placement that
    places models to available GPUs.

    The algorithm must MINIMIZE the maximum KVPR across all GPUs while ensuring models
    can fit into the GPUs' memory. Note that KVPR is KV cache pressure for a GPU. It
    indicates how crowded a GPU is. For a specific GPU, its KVPR is computed as:
        sum(model.req_rate/model.slo for model in models) / (GPU_MEM_SIZE - sum(model.model_size for model in models))
    where models are the models on this GPU.

    The generated program should be as simple as possible and the code should be
    executed correctly without errors.

    The main entry point of the code is:
    `compute_model_placement(gpu_num, models)`
""")



#strategies = ("S1",
#              "S2")
#pr_no_strategy = 0.2
#pr_strategies = (0.5, 0.15, 0.05, 0.05, 0.05)


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
        #strategies=strategies,
        #pr_no_strategy=pr_no_strategy,
        #pr_strategies=pr_strategies,
        max_cost=5.0,
    )


def get_backend_config() -> BackendConfig:
    return BackendConfig(timeout_sec=120)
