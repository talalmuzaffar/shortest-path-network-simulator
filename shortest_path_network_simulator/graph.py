"""Random graph generation."""

import networkx as nx
import random

# Function to generate a random graph
def create_initial_graph(num_nodes=10, num_edges=15):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = random.sample(list(nx.non_edges(G)), min(num_edges, num_nodes * (num_nodes - 1) // 2))
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=random.randint(1, 10))
    return G
