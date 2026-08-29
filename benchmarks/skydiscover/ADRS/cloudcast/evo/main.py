import json
from pathlib import Path

import networkx as nx
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """
    Search for optimal broadcast paths from src to all destinations.

    Args:
        src: Source node
        dsts: List of destination nodes
        G: NetworkX DiGraph with cost/throughput edge attributes
        num_partitions: Number of data partitions

    Returns:
        BroadCastTopology instance containing edge endpoints for every
        destination and partition. Edge cost and throughput are owned by the
        evaluator and must not be included in the returned topology.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t])

    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list([edge source, edge destination])
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        print(f"now the set path is: {self.paths}")
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

# Helper functions
def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance"""
    return BroadCastTopology(src, dsts, num_partitions)


def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)


if __name__ == "__main__":
    request = json.loads(Path("input.json").read_text(encoding="utf-8"))
    graph = nx.DiGraph()
    graph.add_edges_from((edge[0], edge[1], edge[2]) for edge in request["edges"])
    topology = search_algorithm(request["source_node"], request["terminal_nodes"], graph, request["num_partitions"])
    Path("output.json").write_text(json.dumps({"src": topology.src, "dsts": topology.dsts, "num_partitions": topology.num_partitions, "paths": topology.paths}), encoding="utf-8")
