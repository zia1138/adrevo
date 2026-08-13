"""Evaluator for transaction scheduling optimization."""

import sys
import json
import os
import math
import subprocess
import tempfile
import time
import traceback

from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


class TrustedWorkload:
    """Evaluator-owned workload parser and makespan calculator."""

    def __init__(self, workload_json):
        self.workload = list(json.loads(workload_json).values())
        self.txns = []

        for txn in self.workload:
            ops = txn.split()
            txn_len = len(ops)
            txn_ops = []
            for position, op in enumerate(ops, start=1):
                if op != "*":
                    op_type, key = op.split("-")
                    txn_ops.append((op_type, key, position, txn_len))
            self.txns.append(txn_ops)

        self.num_txns = len(self.txns)

    @staticmethod
    def _insert_key_map(key, key_map, op_type, key_start, txn_id):
        index = len(key_map[key]) - 1
        for _ in key_map[key]:
            _, start, end, _ = key_map[key][index]
            if end <= key_start:
                if start <= key_start:
                    index += 1
                    break
            elif start <= key_start:
                index += 1
                break
            index -= 1

        entry = (op_type, key_start, key_start, txn_id)
        if index == -1:
            _, start, end, _ = key_map[key][0]
            if key_start < end and key_start < start:
                key_map[key].insert(0, entry)
            else:
                key_map[key].append(entry)
        else:
            key_map[key].insert(index, entry)

    @staticmethod
    def _find_earliest_read(key, key_map, txn_id):
        if key_map[key][-1][3] == txn_id:
            return key_map[key][-1][1]

        index = len(key_map[key]) - 1
        while index >= 0 and key_map[key][index][0] == "r":
            index -= 1

        if index == -1:
            return 0
        return key_map[key][index][2] + 1

    def get_opt_seq_cost(self, txn_seq):
        key_map = {}
        total_cost = 0

        for txn_id, txn_index in enumerate(txn_seq):
            txn = self.txns[txn_index]
            txn_start = 1
            txn_total_len = 0

            for op_type, key, position, txn_len in txn:
                if key in key_map:
                    if key_map[key][-1][0] == "w" or op_type == "w":
                        key_start = key_map[key][-1][2] + 1
                    else:
                        key_start = self._find_earliest_read(
                            key, key_map, txn_id
                        )
                    txn_start = max(txn_start, key_start - position + 1)
                txn_total_len = txn_len

            txn_end = txn_start + txn_total_len - 1
            total_cost += max(0, txn_end - total_cost)

            for op_type, key, position, _ in txn:
                key_start = txn_start + position - 1
                if key in key_map:
                    self._insert_key_map(
                        key, key_map, op_type, key_start, txn_id
                    )
                else:
                    key_map[key] = [(op_type, key_start, key_start, txn_id)]

        return total_cost


def validate_schedule(txn_seq, expected_size):
    if not isinstance(txn_seq, list) or len(txn_seq) != expected_size:
        return False
    if any(type(txn_id) is not int for txn_id in txn_seq):
        return False
    return set(txn_seq) == set(range(expected_size))


def run_with_timeout(timeout_seconds=600):
    """Run the program in a separate process with timeout."""
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
import sys
import os
import json
import traceback

sys.path.insert(0, r'{eval_dir}')

try:
    from main import get_schedules
    schedules = get_schedules()

    results = {{
        'schedules': schedules,
    }}

    with open({temp_file.name!r} + '.results', 'w') as f:
        json.dump(results, f)

except Exception as e:
    traceback.print_exc()
    with open({temp_file.name!r} + '.results', 'w') as f:
        json.dump({{'error': str(e)}}, f)
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            if os.path.exists(results_path):
                with open(results_path) as f:
                    results = json.load(f)

                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                if set(results) != {"schedules"}:
                    raise RuntimeError("Invalid result format")
                return results["schedules"]
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


if __name__ == "__main__":
    try:
        start_time = time.time()
        schedules = run_with_timeout(timeout_seconds=600)
        eval_time = time.time() - start_time

        workloads = [
            TrustedWorkload(WORKLOAD_1),
            TrustedWorkload(WORKLOAD_2),
            TrustedWorkload(WORKLOAD_3),
        ]

        if not isinstance(schedules, list) or len(schedules) != len(workloads):
            raise ValueError(f"Expected exactly {len(workloads)} schedules")

        for index, (workload, schedule) in enumerate(
            zip(workloads, schedules), start=1
        ):
            if not validate_schedule(schedule, workload.num_txns):
                raise ValueError(f"Invalid schedule for workload {index}")

        makespan = sum(
            workload.get_opt_seq_cost(schedule)
            for workload, schedule in zip(workloads, schedules)
        )
        if not isinstance(makespan, (int, float)):
            raise ValueError("Evaluator computed a non-numeric makespan")
        if not math.isfinite(makespan) or makespan < 0:
            raise ValueError("Evaluator computed an invalid makespan")

        combined_score = 1_000_000.0 / (1.0 + makespan)
        print(
            f"Evaluation: valid=True, makespan={makespan}, "
            f"time={eval_time:.2f}s"
        )
        result = {
            "correct": True,
            "error": None,
            "combined_score": combined_score,
        }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        result = {"correct": False, "error": str(e), "combined_score": 0.0}

    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
