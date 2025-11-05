# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd

class SessionDataset(Dataset):
    def __init__(self, parquet_path, max_len=50):
        self.df = pd.read_parquet(parquet_path)
        # group by session
        self.sessions = self.df.groupby("session_id")["item_id"].apply(list)
        self.max_len = max_len

        # item vocabulary
        self.items = set(self.df.item_id.unique())
        self.item2idx = {item: idx+1 for idx, item in enumerate(sorted(self.items))}
        self.idx2item = {v:k for k,v in self.item2idx.items()}
        self.n_items = len(self.item2idx)

        # map sessions to indices
        self.sessions = self.sessions.apply(lambda seq: [self.item2idx[i] for i in seq])

        # build sequences + next-item labels
        self.samples = []
        for seq in self.sessions:
            for i in range(1, len(seq)):
                inp = seq[:i]
                tgt = seq[i]
                inp = inp[-max_len:]
                inp = [0]*(max_len-len(inp)) + inp
                self.samples.append((inp, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.tensor(inp), torch.tensor(tgt)
