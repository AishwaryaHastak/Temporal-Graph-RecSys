# train.py
import torch
from torch.utils.data import DataLoader
from dataset import SessionDataset
from models.gru4rec import GRU4RecSimple
from evals.metrics import hit_at_k, mrr_at_k
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load datasets and graoh embeddings
train_data = SessionDataset("data/train.parquet")
test_data  = SessionDataset("data/test.parquet")
graph_emb = torch.load("data/graphsage_item_embeddings.pt")  # [num_items, 64]

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=256, shuffle=False) 

# 2. Create initial GRU embeddings 
temp_emb = torch.nn.Embedding(train_data.n_items + 1, 128, padding_idx=0) 
combined_emb = torch.cat([temp_emb.weight[1:], graph_emb], dim=1)
padding_row = torch.zeros(1, combined_emb.size(1))  # padding row
combined_emb = torch.cat([padding_row, combined_emb], dim=0)  # [num_items+1, 192]

# 3. Initialize GRU and optimizer and loss function 
emb_size = combined_emb.size(1)
model = GRU4RecSimple(n_items=train_data.n_items, emb_size=emb_size, n_layers=2).to(device)
model.emb = torch.nn.Embedding.from_pretrained(combined_emb, freeze=False, ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# 4. Initialize evaluate function
def evaluate():
    model.eval()
    hits, mrrs, n = 0, 0, 0
    with torch.no_grad():
        for seq, tgt in test_loader:
            seq, tgt = seq.to(device), tgt.to(device)
            logits, _ = model(seq)
            logits = logits.cpu().numpy()
            tgt = tgt.cpu().numpy()
            for row, t in zip(logits, tgt):
                hits += hit_at_k(row, t, k=10)
                mrrs += mrr_at_k(row, t, k=10)
                n += 1
    print(f"[Eval] Hit@10={hits/n:.4f}, MRR@10={mrrs/n:.4f}")

# 5. Train 
for epoch in range(5):
    model.train()
    total_loss = 0
    for seq, tgt in train_loader:
        seq, tgt = seq.to(device), tgt.to(device)
        logits, _ = model(seq)
        loss = criterion(logits, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch%1==0:
        print(f"Epoch {epoch+1} Loss={total_loss/len(train_loader):.4f}")
        evaluate()
