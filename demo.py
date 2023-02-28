import networkx as nx
G = nx.Graph()
G.add_node(1)
G.add_edge(1, 2)

import torch
from torch_geometric.nn import GCNConv

convs = torch.nn.ModuleList([GCNConv(input_dim, hidden_dim)]+
                                [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + 
                                [GCNConv(hidden_dim, output_dim)])
bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])
softmax = torch.nn.LogSoftmax()