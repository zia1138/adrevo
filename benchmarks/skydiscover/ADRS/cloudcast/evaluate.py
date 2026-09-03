"""Evaluator for CloudCast cloud broadcast optimization."""

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List
from types import SimpleNamespace

import networkx as nx
import pandas as pd


def make_reference_graph(cost_path=None, throughput_path=None, num_vms=1):
    """Build the evaluator-owned graph from the benchmark profiles.

    This deliberately lives in the evaluator rather than ``main.py``: candidate
    code is allowed to change ``main.py`` and therefore must not define the data
    against which its answer is checked.
    """
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    cost = pd.read_csv(cost_path or os.path.join(eval_dir, "profiles/cost.csv"))
    throughput = pd.read_csv(
        throughput_path or os.path.join(eval_dir, "profiles/throughput.csv")
    )

    graph = nx.DiGraph()
    for _, row in throughput.iterrows():
        if row["src_region"] == row["dst_region"]:
            continue
        graph.add_edge(
            row["src_region"],
            row["dst_region"],
            cost=None,
            throughput=num_vms * row["throughput_sent"] / 1e9,
        )

    for _, row in cost.iterrows():
        if row["src"] in graph and row["dest"] in graph[row["src"]]:
            graph[row["src"]][row["dest"]]["cost"] = row["cost"]

    return graph


# ---------- Simulator (from simulator.py) ----------

class BCSimulator:
    data_vol: float = 4.0
    num_partitions: int = 1
    partition_data_vol: int = data_vol / num_partitions
    default_vms_per_region: int = 1
    cost_per_instance_hr: float = 0.54

    def __init__(self, num_vms, reference_graph, output_dir=None):
        self.output_dir = output_dir
        self.default_vms_per_region = num_vms
        self.reference_graph = reference_graph

    def initialization(self, path, config):
        if isinstance(path, str):
            with open(path, "r") as f:
                data = json.loads(f.read())
        else:
            data = {
                "algo": "none",
                "source_node": path.src,
                "terminal_nodes": path.dsts,
                "num_partitions": path.num_partitions,
                "generated_path": path.paths,
            }

        self.src = data["source_node"]
        self.dsts = data["terminal_nodes"]
        self.algo = data["algo"]
        self.paths = data["generated_path"]

        self.num_partitions = config["num_partitions"]
        self.data_vol = config["data_vol"]
        self.partition_data_vol = self.data_vol / self.num_partitions

        providers = ["aws", "gcp", "azure"]
        provider_ingress = [10, 16, 16]
        provider_egress = [5, 7, 16]
        self.ingress_limits = {providers[i]: provider_ingress[i] for i in range(len(providers))}
        self.egress_limits = {providers[i]: provider_egress[i] for i in range(len(providers))}

        if "ingress_limit" in config:
            for p, limit in config["ingress_limit"].items():
                self.ingress_limits[p] = self.default_vms_per_region * limit

        if "egress_limit" in config:
            for p, limit in config["egress_limit"].items():
                self.egress_limits[p] = self.default_vms_per_region * limit

    def evaluate_path(self, path, config, write_to_file=False):
        self.initialization(path, config)

        self.g = self.__construct_g()

        max_t, avg_t, last_dst = self.__transfer_time()
        self.cost = self.__total_cost()

        if write_to_file and self.output_dir:
            open(f"{self.output_dir}/{self.algo}_eval.json", "w").write(
                json.dumps({
                    "path": path,
                    "max_transfer_time": max_t,
                    "avg_transfer_time": avg_t,
                    "last_dst": last_dst,
                    "tot_cost": self.cost,
                })
            )
        return max_t, self.cost

    def __construct_g(self):
        g = nx.DiGraph()
        for dst in self.dsts:
            for partition_id in range(self.num_partitions):
                for edge in self.paths[dst][str(partition_id)]:
                    src, dst_node = edge[0], edge[1]
                    if not g.has_edge(src, dst_node):
                        # Candidate-supplied edge metadata is untrusted. Always
                        # score using the evaluator's canonical graph.
                        edge_data = self.reference_graph[src][dst_node]
                        cost = edge_data["cost"]
                        throughput = edge_data["throughput"]
                        g.add_edge(src, dst_node, throughput=throughput, cost=cost, flow=throughput)
                        g[src][dst_node]["partitions"] = set()
                    g[src][dst_node]["partitions"].add(partition_id)

        for node in g.nodes:
            provider = node.split(":")[0]

            in_edges = g.in_edges(node)
            out_edges = g.out_edges(node)
            in_flow_sum = sum([g[i[0]][i[1]]["flow"] for i in in_edges])
            out_flow_sum = sum([g[o[0]][o[1]]["flow"] for o in out_edges])

            if in_flow_sum > self.ingress_limits.get(provider, 10):
                for edge in in_edges:
                    s, d = edge[0], edge[1]
                    flow_proportion = 1 / len(list(in_edges))
                    g[s][d]["flow"] = min(g[s][d]["flow"], self.ingress_limits.get(provider, 10) * flow_proportion)

            if out_flow_sum > self.egress_limits.get(provider, 5):
                for edge in out_edges:
                    s, d = edge[0], edge[1]
                    flow_proportion = 1 / len(list(out_edges))
                    g[s][d]["flow"] = min(g[s][d]["flow"], self.egress_limits.get(provider, 5) * flow_proportion)

        return g

    def __transfer_time(self, log=True):
        t_dict = dict()
        for dst in self.dsts:
            partition_time = float("-inf")
            for i in range(self.num_partitions):
                path_edges = self.paths[dst][str(i)]
                bottleneck = min(self.g[e[0]][e[1]]['flow'] for e in path_edges)
                t = self.partition_data_vol / bottleneck if bottleneck > 0 else float('inf')
                partition_time = max(partition_time, t)
            t_dict[dst] = partition_time

        max_t = max(t_dict.values())
        last_dst = [k for k, v in t_dict.items() if v == max_t]
        avg_t = sum(t_dict.values()) / len(t_dict.values())
        return max_t, avg_t, last_dst

    def __total_cost(self):
        sum_egress_cost = 0
        for edge in self.g.edges.data():
            edge_data = edge[-1]
            sum_egress_cost += (
                len(edge_data["partitions"]) * self.partition_data_vol * edge_data["cost"]
            )

        runtime_s, _, _ = self.__transfer_time(log=False)
        runtime_s = round(runtime_s, 2)
        sum_instance_cost = 0
        for node in self.g.nodes():
            sum_instance_cost += self.default_vms_per_region * (self.cost_per_instance_hr / 3600) * runtime_s

        sum_cost = sum_egress_cost + sum_instance_cost
        return sum_cost


# ---------- Validation ----------

def validate_broadcast_topology(bc_t, source_node, terminal_nodes, num_partitions, G):
    if set(bc_t.dsts) != set(terminal_nodes):
        missing_dsts = set(terminal_nodes) - set(bc_t.dsts)
        extra_dsts = set(bc_t.dsts) - set(terminal_nodes)
        return False, f"Destination mismatch: missing={missing_dsts}, extra={extra_dsts}"

    if bc_t.src != source_node:
        return False, f"Source mismatch: expected={source_node}, got={bc_t.src}"

    if bc_t.num_partitions != num_partitions:
        return False, (
            f"Partition count mismatch: expected={num_partitions}, "
            f"got={bc_t.num_partitions}"
        )

    missing_partitions = []
    empty_partitions = []
    invalid_paths = []

    for dst in terminal_nodes:
        if dst not in bc_t.paths:
            return False, f"Missing destination '{dst}' in paths"

        for partition_id in range(num_partitions):
            partition_key = str(partition_id)

            if partition_key not in bc_t.paths[dst]:
                missing_partitions.append((dst, partition_id))
                continue

            partition_paths = bc_t.paths[dst][partition_key]

            if partition_paths is None or len(partition_paths) == 0:
                empty_partitions.append((dst, partition_id))
                continue

            path_nodes = [source_node]
            path_valid = True

            for edge in partition_paths:
                if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                    invalid_paths.append((dst, partition_id, "edge format invalid"))
                    path_valid = False
                    break

                edge_src, edge_dst = edge[0], edge[1]

                if not G.has_edge(edge_src, edge_dst):
                    invalid_paths.append((dst, partition_id, f"edge {edge_src}->{edge_dst} not in graph"))
                    path_valid = False
                    break

                if path_nodes[-1] != edge_src:
                    invalid_paths.append((dst, partition_id, f"path discontinuity: expected {path_nodes[-1]}, got {edge_src}"))
                    path_valid = False
                    break

                path_nodes.append(edge_dst)

            if path_valid and path_nodes[-1] != dst:
                invalid_paths.append((dst, partition_id, f"path does not reach destination: ends at {path_nodes[-1]}, expected {dst}"))

    errors = []
    if missing_partitions:
        errors.append(f"Missing partitions: {missing_partitions}")
    if empty_partitions:
        errors.append(f"Empty partitions: {empty_partitions}")
    if invalid_paths:
        errors.append(f"Invalid paths: {invalid_paths}")

    if errors:
        return False, "Validation failed: " + "; ".join(errors)

    expected_total_partitions = len(terminal_nodes) * num_partitions
    actual_partitions = 0
    for dst in terminal_nodes:
        for partition_id in range(num_partitions):
            partition_key = str(partition_id)
            if (partition_key in bc_t.paths[dst] and
                bc_t.paths[dst][partition_key] is not None and
                len(bc_t.paths[dst][partition_key]) > 0):
                actual_partitions += 1

    if actual_partitions != expected_total_partitions:
        return False, f"Data loss detected: expected {expected_total_partitions} partitions, got {actual_partitions}"

    return True, None


def canonicalize_paths(bc_t, terminal_nodes, num_partitions, reference_graph):
    """Replace all candidate metadata with evaluator-owned edge attributes."""
    paths = {}
    for dst in terminal_nodes:
        paths[dst] = {}
        for partition_id in range(num_partitions):
            key = str(partition_id)
            paths[dst][key] = [
                [edge[0], edge[1], dict(reference_graph[edge[0]][edge[1]])]
                for edge in bc_t.paths[dst][key]
            ]
    return paths


# ---------- Main evaluation ----------

if __name__ == "__main__":
    try:
        # Configuration files (relative to evaluate.py location)
        eval_dir = os.path.dirname(os.path.abspath(__file__))
        config_files = [
            os.path.join(eval_dir, "examples/config/intra_aws.json"),
            os.path.join(eval_dir, "examples/config/intra_azure.json"),
            os.path.join(eval_dir, "examples/config/intra_gcp.json"),
            os.path.join(eval_dir, "examples/config/inter_agz.json"),
            os.path.join(eval_dir, "examples/config/inter_gaz2.json")
        ]

        existing_configs = [f for f in config_files if os.path.exists(f)]

        if not existing_configs:
            result = {"correct": False, "error": f"No configuration files found. Checked: {config_files}", "combined_score": 0.0}
            with open("results.json", "w") as f:
                json.dump(result, f, indent=4)
            sys.exit(0)

        num_vms = 2
        total_cost = 0.0
        successful_configs = 0
        failed_configs = 0

        for jsonfile in existing_configs:
            try:
                with open(jsonfile, "r") as f:
                    config_name = os.path.basename(jsonfile).split(".")[0]
                    config = json.loads(f.read())

                reference_graph = make_reference_graph(num_vms=int(num_vms))

                source_node = config["source_node"]
                terminal_nodes = config["dest_nodes"]
                num_partitions = config["num_partitions"]

                input_path = Path("evo/input.json")
                output_path = Path("evo/output.json")
                output_path.unlink(missing_ok=True)
                input_path.write_text(json.dumps({
                    "source_node": source_node,
                    "terminal_nodes": terminal_nodes,
                    "num_partitions": num_partitions,
                    "edges": [[src, dst, dict(data)] for src, dst, data in reference_graph.edges(data=True)],
                }), encoding="utf-8")
                subprocess.run(["uv", "run", "-qq", "--directory", "evo", "python", "main.py"], check=True)
                candidate_output = json.loads(output_path.read_text(encoding="utf-8"))
                bc_t = SimpleNamespace(**candidate_output)

                is_valid, validation_error = validate_broadcast_topology(
                    bc_t, source_node, terminal_nodes, num_partitions, reference_graph
                )

                if not is_valid:
                    result = {"correct": False, "error": f"Invalid broadcast topology: {validation_error}", "combined_score": 0.0}
                    with open("results.json", "w") as f:
                        json.dump(result, f, indent=4)
                    sys.exit(0)

                trusted_paths = canonicalize_paths(
                    bc_t, terminal_nodes, num_partitions, reference_graph
                )

                # Save generated paths with canonical edge data only.
                directory = f"paths/{config_name}"
                Path(directory).mkdir(parents=True, exist_ok=True)
                outf = f"{directory}/search_algorithm.json"
                with open(outf, "w") as outfile:
                    outfile.write(json.dumps({
                        "algo": "search_algorithm",
                        "source_node": bc_t.src,
                        "terminal_nodes": bc_t.dsts,
                        "num_partitions": bc_t.num_partitions,
                        "generated_path": trusted_paths,
                    }))

                # Evaluate
                output_dir = f"evals/{config_name}"
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                simulator = BCSimulator(int(num_vms), reference_graph, output_dir)
                _, cost = simulator.evaluate_path(outf, config)

                total_cost += cost
                successful_configs += 1
                print(f"Config {config_name}: cost={cost:.2f}")

            except Exception as e:
                print(f"Failed to process {os.path.basename(jsonfile)}: {str(e)}")
                failed_configs += 1
                break

        if failed_configs != 0:
            result = {"correct": False, "error": "1 or more configuration files failed to process", "combined_score": 0.0}
        else:
            cost_score = 1.0 / (1.0 + total_cost)
            result = {"correct": True, "error": None, "combined_score": float(cost_score)}

        with open("results.json", "w") as f:
            json.dump(result, f, indent=4)

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        result = {"correct": False, "error": str(e), "combined_score": 0.0}
        with open("results.json", "w") as f:
            json.dump(result, f, indent=4)
