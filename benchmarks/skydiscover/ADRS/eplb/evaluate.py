"""Evaluator for EPLB (Expert Parallelism Load Balancer)."""

import json
import functools
import time
import traceback
import os

import torch
from main import rebalance_experts


# ---------- Constants ----------

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKLOAD_PATH = os.path.join(_CURRENT_DIR, "data/expert-load.json")
REBALANCE_INTERVAL = 100

NUM_REPLICAS = 288
NUM_GROUPS = 8
NUM_GPUS = 32
NUM_NODES = 4


@functools.cache
def load_workloads(path: str) -> list[torch.Tensor]:
    with open(path, "r") as f:
        data = json.load(f)

    total_len = len(data['load_history'])
    workloads = []
    for i in range(0, total_len, REBALANCE_INTERVAL):
        start = i
        end = min(start + REBALANCE_INTERVAL, total_len)
        load = torch.tensor([x['logical_expert_load'] for x in data['load_history'][start:end]]).sum(dim=0)
        workloads.append(load)

    return workloads


def simulate_inference(
        log2phy: torch.Tensor,
        logcnt: torch.Tensor,
        workload: torch.Tensor,
    ) -> tuple[float, float]:
    """Simulate a MoE inference and return the balancedness factor."""
    num_layers, num_logical_experts = workload.shape

    num_physical_experts = NUM_REPLICAS
    total_physical_load = torch.zeros(num_layers, num_physical_experts, dtype=torch.float, device=workload.device)

    for layer_id in range(num_layers):
        for logical_id in range(num_logical_experts):
            logical_load = workload[layer_id][logical_id].item()

            if logical_load <= 0:
                continue

            num_replicas = int(logcnt[layer_id][logical_id].item())

            if num_replicas <= 0:
                continue

            physical_ids = log2phy[layer_id][logical_id][:num_replicas]

            replica_load = logical_load / num_replicas

            total_physical_load[layer_id, physical_ids] += replica_load

    total_load = total_physical_load.sum()
    if total_load == 0:
        return 0.0, 0.0

    # Expert-level balancedness
    expert_layer_avg = total_physical_load.mean(dim=1).sum().item()
    expert_layer_max = total_physical_load.max(dim=1).values.sum().item()
    balancedness_expert = expert_layer_avg / expert_layer_max if expert_layer_max > 0 else 0.0

    # GPU-level balancedness
    gpu_load = total_physical_load.view(num_layers, NUM_GPUS, -1).sum(dim=2)

    layer_avg = gpu_load.mean(dim=1)
    layer_max = gpu_load.max(dim=1).values

    avg_load = layer_avg.sum().item()
    max_load = layer_max.sum().item()

    balancedness_gpu = avg_load / max_load if max_load > 0 else 0.0

    return balancedness_gpu, balancedness_expert


if __name__ == "__main__":
    try:
        workloads = load_workloads(WORKLOAD_PATH)

        balancedness_scores_gpu = []
        balancedness_scores_expert = []
        times_algorithm = []
        times_inference = []

        for i in range(len(workloads) - 1):
            start_time = time.perf_counter()
            phy2log, log2phy, logcnt = rebalance_experts(
                workloads[i],
                NUM_REPLICAS,
                NUM_GROUPS,
                NUM_NODES,
                NUM_GPUS,
            )
            end_time_algorithm = time.perf_counter()

            # Validate outputs to prevent reward hacking
            if phy2log.shape[1] != NUM_REPLICAS:
                result = {"correct": False, "error": f"phy2log shape wrong: {tuple(phy2log.shape)}", "combined_score": 0.0}
                with open("results.json", "w") as f:
                    json.dump(result, f, indent=4)
                sys.exit(0)

            if not torch.all(logcnt.sum(dim=1) == NUM_REPLICAS):
                sums = logcnt.sum(dim=1)
                result = {"correct": False, "error": f"logcnt sums != {NUM_REPLICAS}: {sums[:5].tolist()}...", "combined_score": 0.0}
                with open("results.json", "w") as f:
                    json.dump(result, f, indent=4)
                sys.exit(0)

            if (logcnt < 0).any():
                result = {"correct": False, "error": "logcnt contains negative values", "combined_score": 0.0}
                with open("results.json", "w") as f:
                    json.dump(result, f, indent=4)
                sys.exit(0)

            next_workload = workloads[i + 1]
            has_load = next_workload > 0
            has_no_replicas = logcnt == 0
            unhandled = has_load & has_no_replicas
            if unhandled.any():
                unhandled_count = int(unhandled.sum().item())
                result = {"correct": False, "error": f"Unhandled load: {unhandled_count} experts have load but 0 replicas", "combined_score": 0.0}
                with open("results.json", "w") as f:
                    json.dump(result, f, indent=4)
                sys.exit(0)

            balancedness_score_gpu, balancedness_score_expert = simulate_inference(log2phy, logcnt, workloads[i + 1])
            end_time = time.perf_counter()
            balancedness_scores_gpu.append(balancedness_score_gpu)
            balancedness_scores_expert.append(balancedness_score_expert)
            times_algorithm.append(end_time_algorithm - start_time)
            times_inference.append(end_time - start_time)

        avg_balancedness_score_gpu = sum(balancedness_scores_gpu) / len(balancedness_scores_gpu)
        avg_balancedness_score_expert = sum(balancedness_scores_expert) / len(balancedness_scores_expert)
        avg_time_algorithm = sum(times_algorithm) / len(times_algorithm)
        avg_time_inference = sum(times_inference) / len(times_inference)
        speed_score = 0.002 / avg_time_inference
        combined_score = (avg_balancedness_score_expert + speed_score) / 2

        result = {
            "correct": True,
            "error": None,
            "combined_score": float(combined_score),
        }

    except Exception as e:
        traceback.print_exc()
        result = {"correct": False, "error": str(e), "combined_score": 0.0}

    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
