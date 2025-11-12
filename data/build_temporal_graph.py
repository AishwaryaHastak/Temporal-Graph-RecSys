# build_temporal_graph.py
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import torch
from pathlib import Path
import pickle
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data")

def build_session_graph(df, time_decay=1e-5):
    """
    Build temporal item-item graph from session data.
    Nodes = items
    Edges = co-occurrence transitions (i -> j)
    Edge weight decays with time difference
    """
    # Encode category into integer ids
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["category"])

    G = nx.DiGraph()
    sessions = df.groupby("session_id")

    for _, session in sessions:
        session = session.sort_values("timestamp")
        items = session["item_id"].tolist()
        times = session["timestamp"].tolist() 
        for i in range(len(items) - 1):
            src, tgt = items[i], items[i + 1]
            dt = (times[-1] - times[i]).total_seconds()  # how old relative to session end
            w = np.exp(-time_decay * dt)

            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += w
            else:
                G.add_edge(src, tgt, weight=w)
    # Add category feature as one-hot
    num_cats = len(df["category_id"].unique())
    for item, cat in df.groupby("item_id")["category_id"].first().items():
        cat_onehot = torch.zeros(num_cats)
        cat_onehot[cat] = 1.0
        G.nodes[item]["x"] = cat_onehot
    return G


if __name__ == "__main__":
    print("Loading training data...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")

    print("Building temporal graph...")
    G = build_session_graph(df, time_decay=1e-5)

    print(f"Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges")
    with open(DATA_DIR / "item_graph.pkl", "wb") as f:
        pickle.dump(G, f)
