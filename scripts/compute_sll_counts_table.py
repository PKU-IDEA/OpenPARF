#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : compute_sll_counts_table.sh
# Author            : Runzhe Tao <rztao@my.swjtu.edu.cn>
# Date              : 01.31.2024
# Last Modified Date: 01.31.2024
# Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>

# ### Usage Instructions for `compute_sll_counts_table.py` Script
# - **Note**: Considering fabrication and technological constraints, the size of SLR topology do not exceed 5x5. 
#   For problems within this scale, the script employs a straightforward yet effective implementation approach.
# #### Step 1: Generating the SLL Counts Lookup Table
# - Run the script to generate the SLL counts lookup table for the specified SLR topology. 
#   Note that the 1x4 and 2x2 architectures are predefined and do not require additional imports.
# - Example command:
#   ```python
#   python compute_sll_counts_table.py --num_cols <numCols> --num_rows <numRows> --output <filename>
#   ```
#   - Replace `<filename>` with your desired file name. 
#   - Replace `<numCols>` and `<numRows>` with the desired number of columns and rows, respectively.
# - After execution, the script will produce a file named `<filename>.npy`. 
#   You should move this file to the directory:  `<install>/openparf/ops/sll/`.

# #### Step 2: Importing the SLL Lookup Table into Your Project
# - You can import the lookup table into the Python script located at `<install>/openparf/ops/sll/sll.py`:
#   - Use the following code snippet to load the numpy data:
#        ```python
#        self.sll_counts_table = torch.from_numpy(
#            np.load(
#                os.path.join(
#                    os.path.dirname(os.path.abspath(__file__)),
#                    "<filename>.npy"))).to(dtype=torch.int32)
#        ```

# Follow these steps to successfully integrate the SLL counts lookup table into your project.

import os
import argparse
import networkx as nx
import numpy as np
from itertools import combinations
from tqdm import tqdm


def create_grid_graph(m, n):
    """
    Create a grid graph of size m x n.

    Args:
    m (int): Number of rows in the grid.
    n (int): Number of columns in the grid.

    Returns:
    NetworkX graph: A graph representing the SLR grid.
    """
    G = nx.grid_2d_graph(m, n)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping)


def precompute_shortest_paths(G, num_nodes):
    """
    Precompute and cache the shortest path lengths between all pairs of nodes.

    Args:
    G (NetworkX graph): The graph for which to compute shortest paths.
    num_nodes (int): The number of nodes in the graph.

    Returns:
    dict: A dictionary with node pairs as keys and shortest path lengths as values.
    """
    shortest_paths = {}
    for node1, node2 in combinations(range(num_nodes), 2):
        shortest_paths[(node1, node2)] = nx.shortest_path_length(G,
                                                                 source=node1,
                                                                 target=node2)
    return shortest_paths


def compute_sll_counts(selected_nodes, shortest_paths):
    """
    Compute the total weight of the minimum spanning tree for selected nodes.

    Args:
    graph (NetworkX graph): The graph on which calculations are performed.
    selected_nodes (list): A list of selected nodes for which to compute the Steiner tree.
    shortest_paths (dict): Precomputed shortest paths for the graph.

    Returns:
    int: The total weight of the minimum spanning tree.
    """
    new_graph = nx.Graph()
    for node1, node2 in combinations(selected_nodes, 2):
        path_length = shortest_paths.get((min(node1, node2), max(node1, node2)),
                                         0)
        new_graph.add_edge(node1, node2, weight=path_length)

    mst = nx.minimum_spanning_tree(new_graph, algorithm='kruskal')
    total_weight = sum(edge[2]['weight'] for edge in mst.edges(data=True))

    return total_weight


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=
        'Calculate approximate minimum length of Steiner trees in a grid graph.'
    )
    parser.add_argument('--num_cols',
                        type=int,
                        help='Number of columns in the SLR topology.')
    parser.add_argument('--num_rows',
                        type=int,
                        help='Number of rows in the SLR topology.')
    parser.add_argument(
        '--output',
        type=str,
        default='sll_counts_table.npy',
        help='Name of the output file (default: sll_counts_table.npy).')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    num_rows = args.num_rows
    num_cols = args.num_cols

    graph = create_grid_graph(num_rows, num_cols)
    print(f"Edges of {num_cols} x {num_rows} SLR Topology Graph:",
          sorted(graph.edges()))

    shortest_paths = precompute_shortest_paths(graph, num_rows * num_cols)

    sll_counts_table = np.zeros(pow(2, num_cols * num_rows), dtype=np.int32)
    total_combinations = pow(2, num_cols * num_rows)

    # Compute the Steiner tree length for each combination of nodes
    for i in tqdm(range(total_combinations)):
        selected_nodes = [j for j in range(num_cols * num_rows) if (i >> j) & 1]
        sll_counts_table[i] = compute_sll_counts(selected_nodes, shortest_paths)

    # Save and print the results
    dir_path = os.path.dirname(os.path.abspath(__name__))
    output_file = os.path.join(dir_path, args.output)
    np.save(output_file, sll_counts_table)
    print(f'Results saved to "{output_file}.npy".')
    print('sll_counts_table:')
    print(sll_counts_table)
