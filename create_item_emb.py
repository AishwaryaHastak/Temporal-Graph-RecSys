# create_item_emb.py
import torch
from torch_geometric.loader import DataLoader
from models.graphsage import GraphSAGERec
from torch_geometric.utils import negative_sampling

# Load graph
data = torch.load("data/item_graph.pt", weights_only=False)

# Model
in_dim = data.x.size(1)
model = GraphSAGERec(in_dim=in_dim, hidden_dim=64, out_dim=64)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_link_prediction(model, data, epochs=10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(data)  # node embeddings

        # Positive edges
        pos_edge_index = data.edge_index

        # Negative samples
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        # Compute scores
        pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

        # Loss: maximize positive, minimize negative similarities
        loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean() \
               -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

        loss.backward()
        optimizer.step()
        if (epoch)%5==0:
            print(f"Epoch {epoch+1}, Loss={loss.item():.4f}")

train_link_prediction(model, data, epochs=20)

# Get final embeddings
model.eval()
with torch.no_grad():
    embeddings = model(data)

torch.save(embeddings, "data/graphsage_item_embeddings.pt")
print("✅ Trained GraphSAGE embeddings saved.")
