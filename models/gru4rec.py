# models/gru4rec.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU4RecSimple(nn.Module):
    def __init__(self, n_items, emb_size=128, hid=128, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(n_items+1, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hid, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(hid, n_items+1)

    def forward(self, input_seq, hidden=None):
        # input_seq: (B, L) item indices
        x = self.emb(input_seq)           # (B, L, E)
        out, h = self.gru(x, hidden)      # out: (B, L, H)
        last = out[:, -1, :]              # (B, H)
        logits = self.out(last)           # (B, n_items+1)
        return logits, h
