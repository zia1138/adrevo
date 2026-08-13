"""Evaluator for Prism GPU model placement optimization."""

import json
import copy
import sys
import time
import traceback
import concurrent.futures
from dataclasses import dataclass

import numpy as np


GPU_MEM_SIZE = 80  # GB
MIN_INT = float('-inf')

# Preserve the evaluator dependencies that submission code could otherwise
# replace through shared module objects after it is imported.
_trusted_isnan = np.isnan
_trusted_isinf = np.isinf
_trusted_seed = np.random.seed
_trusted_randint = np.random.randint
_trusted_deepcopy = copy.deepcopy


@dataclass
class Model:
    model_name: str
    model_size: int
    req_rate: int
    slo: int
    cur_gpu_id: int


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def safe_float(value):
    try:
        if _trusted_isnan(value) or _trusted_isinf(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def snapshot_models(models: list[Model]) -> list[dict]:
    """Capture model state so solutions cannot improve their score by mutation."""
    return [_trusted_deepcopy(vars(model)) for model in models]


def models_are_unchanged(models: list[Model], snapshots: list[dict]) -> bool:
    """Return whether every input model still has exactly its original state."""
    if len(models) != len(snapshots):
        return False
    return all(vars(model) == snapshot for model, snapshot in zip(models, snapshots))


def verify_gpu_mem_constraint(
    placement_data: dict[int, list[Model]],
    trusted_models: dict[int, dict],
) -> bool:
    """Verify GPU memory using evaluator-owned model snapshots."""
    if placement_data is None:
        return False
    for gpu_id, models in placement_data.items():
        if sum(trusted_models[id(model)]["model_size"] for model in models) > GPU_MEM_SIZE:
            return False
    return True


def verify_gpu_ids(placement_data: dict[int, list[Model]], gpu_num: int) -> bool:
    """Verify that placement only refers to GPUs supplied by the test case."""
    return all(
        type(gpu_id) is int and 0 <= gpu_id < gpu_num
        for gpu_id in placement_data
    )


def calculate_kvcache_pressure(
    placement_data: dict[int, list[Model]],
    trusted_models: dict[int, dict],
) -> float:
    """Calculate pressure using model values captured before submission execution."""
    max_kvpr = MIN_INT
    for gpu_id, models in placement_data.items():
        model_states = [trusted_models[id(model)] for model in models]
        total_model_size = sum(state["model_size"] for state in model_states)
        total_weighted_req_rate = sum(
            state["req_rate"] / state["slo"] for state in model_states
        )
        if GPU_MEM_SIZE - total_model_size > 0:
            kvpr = total_weighted_req_rate / (GPU_MEM_SIZE - total_model_size)
        else:
            kvpr = 1000000
        max_kvpr = max(max_kvpr, kvpr)
    return max_kvpr


def generate_test_gpu_models(num_tests=50):
    test_cases = []
    _trusted_seed(42)

    for i in range(num_tests):
        gpu_num = int(_trusted_randint(5, 10))
        gpu_models = []
        for j in range(gpu_num * 2):
            model_size = int(_trusted_randint(10, 30))
            req_rate = int(_trusted_randint(1, 10))
            slo = int(_trusted_randint(5, 10))
            gpu_models.append(Model(model_name=f"model_{j}", model_size=model_size, req_rate=req_rate, slo=slo, cur_gpu_id=j))

        test_cases.append((gpu_num, gpu_models))

    return test_cases


if __name__ == "__main__":
    try:
        # Build evaluator-owned inputs before importing submission code. Python
        # imports execute module-level code in this process, so import main only
        # after test generation and dependency capture are complete.
        test_gpu_models = generate_test_gpu_models()
        from main import compute_model_placement

        all_kvpr = []
        all_metrics = []
        successful_runs = 0

        for i, (gpu_num, gpu_models) in enumerate(test_gpu_models):
            try:
                start_time = time.time()
                model_snapshots = snapshot_models(gpu_models)
                # Submission code chooses only the placement. Constraints and
                # metrics use evaluator-owned values captured before it runs.
                trusted_models = {
                    id(model): snapshot
                    for model, snapshot in zip(gpu_models, model_snapshots)
                }

                placement = run_with_timeout(
                    compute_model_placement,
                    kwargs={'gpu_num': gpu_num, 'models': gpu_models},
                    timeout_seconds=10
                )

                execution_time = time.time() - start_time

                # Solutions may choose a placement, but must not alter the
                # evaluator-owned model objects to manipulate scoring.
                if not models_are_unchanged(gpu_models, model_snapshots):
                    result = {"correct": False, "error": f"Placement {i}: Input models were modified", "combined_score": 0.0}
                    with open("results.json", "w") as f:
                        json.dump(result, f, indent=4)
                    sys.exit(0)

                # Validate result format
                if type(placement) is not dict:
                    result = {"correct": False, "error": f"Placement {i}: Expected dict, got {type(placement).__name__}", "combined_score": 0.0}
                    with open("results.json", "w") as f:
                        json.dump(result, f, indent=4)
                    sys.exit(0)

                # Validate all models are placed
                placed_models = []
                for gpu_id, assigned_models in placement.items():
                    if type(assigned_models) is not list:
                        result = {"correct": False, "error": f"GPU {gpu_id} value must be list, got {type(assigned_models).__name__}", "combined_score": 0.0}
                        with open("results.json", "w") as f:
                            json.dump(result, f, indent=4)
                        sys.exit(0)
                    placed_models.extend(assigned_models)

                if len(placed_models) != len(gpu_models):
                    result = {"correct": False, "error": f"Not all models placed: {len(placed_models)}/{len(gpu_models)}", "combined_score": 0.0}
                    with open("results.json", "w") as f:
                        json.dump(result, f, indent=4)
                    sys.exit(0)

                # Check for duplicate placements (by object identity)
                placed_ids = [id(m) for m in placed_models]
                if len(set(placed_ids)) != len(placed_ids):
                    result = {"correct": False, "error": "Duplicate models detected", "combined_score": 0.0}
                    with open("results.json", "w") as f:
                        json.dump(result, f, indent=4)
                    sys.exit(0)

                # Check placed models are the exact input objects
                original_ids = {id(m) for m in gpu_models}
                if set(placed_ids) != original_ids:
                    result = {"correct": False, "error": "Placed models don't match input models (missing or foreign models)", "combined_score": 0.0}
                    with open("results.json", "w") as f:
                        json.dump(result, f, indent=4)
                    sys.exit(0)

                # Placements may only use the GPUs provided by the test case.
                if not verify_gpu_ids(placement, gpu_num):
                    result = {"correct": False, "error": f"Placement {i}: Invalid GPU id", "combined_score": 0.0}
                    with open("results.json", "w") as f:
                        json.dump(result, f, indent=4)
                    sys.exit(0)

                # Verify GPU memory constraints
                if not verify_gpu_mem_constraint(placement, trusted_models):
                    result = {"correct": False, "error": "GPU memory constraint violated", "combined_score": 0.0}
                    with open("results.json", "w") as f:
                        json.dump(result, f, indent=4)
                    sys.exit(0)

                max_kvpr = calculate_kvcache_pressure(placement, trusted_models)

                all_kvpr.append(safe_float(max_kvpr))
                all_metrics.append({'execution_time': safe_float(execution_time)})
                successful_runs += 1

            except TimeoutError:
                print(f"Placement {i}: Timeout")
                continue
            except Exception as e:
                print(f"Placement {i}: Error - {str(e)}")
                continue

        if successful_runs == 0:
            result = {"correct": False, "error": "All test cases failed", "combined_score": 0.0}
        else:
            # Do not call mutable third-party module attributes here. main.py is
            # untrusted and is imported into this process, so it can monkey-patch
            # np.mean during import.
            avg_kvpr = sum(all_kvpr) / len(all_kvpr)
            if avg_kvpr != 0:
                avg_kvpr = 1.0 / avg_kvpr
            success_rate = successful_runs / len(test_gpu_models)

            result = {
                "correct": True,
                "error": None,
                "combined_score": safe_float(avg_kvpr) + safe_float(success_rate),
            }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        result = {"correct": False, "error": str(e), "combined_score": 0.0}

    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
