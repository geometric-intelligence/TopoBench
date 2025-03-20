import torch
import networkx as nx
from typing import List, Tuple, Dict


def normalize_cycle(cycle: List[int]) -> Tuple[int, ...]:
    """Normalize a cycle by considering shift and reflection invariance."""
    n = len(cycle)
    cycle_variants = [tuple(cycle[i:] + cycle[:i]) for i in range(n)] + [
        tuple(cycle[i:] + cycle[:i])[::-1] for i in range(n)
    ]
    return min(cycle_variants)


def count_cycles_per_node_optimized(
    edge_index: torch.Tensor, num_nodes: int, cycle_lengths: List[int]
) -> Tuple[int, torch.Tensor]:
    """
    Efficiently counts the number of cycles each node belongs to for multiple cycle lengths.
    """
    G = {i: set() for i in range(num_nodes)}  # Adjacency list
    edges = edge_index.detach().clone().cpu().T.numpy()
    for u, v in edges:
        G[u].add(v)
        G[v].add(u)  # Undirected graph

    max_length = max(cycle_lengths)
    length_indices = {length: i for i, length in enumerate(cycle_lengths)}
    cycle_counts_per_node = torch.zeros(
        (num_nodes, len(cycle_lengths)), dtype=torch.float32
    )
    visited_cycles = set()

    def find_cycles(
        start: int, path: List[int], visited: Dict[int, int]
    ) -> None:
        v = path[-1]
        if len(path) > max_length:
            return
        for neighbor in G[v]:
            if neighbor == start and len(path) in length_indices:
                norm_cycle = normalize_cycle(path)
                if norm_cycle not in visited_cycles:
                    visited_cycles.add(norm_cycle)
                    for node in path:
                        cycle_counts_per_node[
                            node, length_indices[len(path)]
                        ] += 1
            elif neighbor not in visited:
                visited[neighbor] = True
                find_cycles(start, path + [neighbor], visited)
                del visited[neighbor]

    for node in G:
        find_cycles(node, [node], {node: True})

    return num_nodes, cycle_counts_per_node


def test_cycle_counting():
    """Tests the optimized cycle counting function."""
    edge_list = torch.tensor(
        [[0, 1], [1, 2], [2, 0], [1, 4], [3, 4], [4, 1], [0, 3], [1, 3]]
    )  # Triangle + Square
    num_nodes = 5
    cycle_lengths = [3, 4]

    num_nodes_optimized, cycle_counts_optimized = (
        count_cycles_per_node_optimized(edge_list.T, num_nodes, cycle_lengths)
    )

    print("Optimized Cycle Counts Per Node:")
    print(cycle_counts_optimized)


if __name__ == "__main__":
    test_cycle_counting()
