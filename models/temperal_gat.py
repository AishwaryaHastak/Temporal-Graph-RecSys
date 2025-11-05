# models/temporal_gat.py
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

class TemporalItemGAT(torch.nn.Module):
    def __init__(self, n_items, in_dim=32, hid=64, heads=2, out_dim=64):
        super().__init__()
        # item features could be learned embeddings + popularity scalar
        self.item_emb = torch.nn.Embedding(n_items+1, in_dim, padding_idx=0)
        self.gat1 = GATConv(in_dim, hid, heads=heads, concat=True)
        self.gat2 = GATConv(hid*heads, out_dim, heads=1, concat=False)

    def forward(self, data: Data):
        # data.x : node features (optional), data.edge_index: edges
        x = self.item_emb(data.x.squeeze().long()) if data.x is not None else self.item_emb.weight
        x = self.gat1(x, data.edge_index)
        x = torch.relu(x)
        x = self.gat2(x, data.edge_index)
        return x  # item embeddings (num_nodes, out_dim)
