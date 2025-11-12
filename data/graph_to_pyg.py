import pickle
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from pathlib import Path

DATA_DIR = Path("data")

print("Loading item graph...")
with open(DATA_DIR / "item_graph.pkl", "rb") as f:
    G = pickle.load(f)

print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# optional sanity check
if not nx.is_directed(G):
    G = G.to_directed()

# for node in G.nodes():
#     G.nodes[node]['x'] = torch.ones(8)  # 8-dim dummy feature; replace later with item embeddings

# Convert to PyG
data = from_networkx(G)
 
# data.x         -> [num_nodes, feature_dim]
# data.edge_index -> [2, num_edges]
print(data)
torch.save(data, DATA_DIR / "item_graph.pt")
print("✅ Saved PyG graph at data/item_graph.pt")
