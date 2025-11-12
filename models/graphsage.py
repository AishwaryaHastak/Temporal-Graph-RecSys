# models/graphsage.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGERec(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
    
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index 
        x = F.relu(self.conv1(x, edge_index ))
        x = self.conv2(x, edge_index)
        return x  # final node embeddings
