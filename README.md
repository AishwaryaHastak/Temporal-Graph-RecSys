# Temporal Graph Embeddings Session-Based Recommendation (TGESR)

**TGESR** is a hybrid **graph + sequential recommender system** that integrates temporal and relational patterns in user sessions. It combines GRU4Rec (sequential modeling) and GraphSAGE/TGAT(graph-based modeling) for session-based recommendation.
It learns what items co-occur and in what order they appear.

### Why TGESR?
Traditional session-based models (e.g., GRU4Rec) only capture **temporal order** within sessions.  
However, they ignore **item-item relational structure** across sessions.  
TGESR bridges this by combining:

| Component | Purpose |
|------------|----------|
| **GRU4Rec** | Learns sequential click patterns within a session |
| **Graph Encoder (GraphSAGE / TGAT)** | Learns co-occurrence relationships between items across sessions |
| **Fusion** | Produces temporally- and relationally-aware recommendations |

--- 

## Dataset 
We use the **YooChoose dataset (RecSys 2025 Challenge)** with fields:  
`(session_id, item_id, timestamp, category)`
- Keep only sessions with **≥ 2 interactions**.  
- Split into **train/test** sets.  
- Compute **item popularity** (for optional cold-start handling).

## Build temporal graph for item-item interactions
Each session (e.g., A → B → C → D) creates directed edges:
A→B, B→C, C→D.
Each edge weight represents **recency-weighted co-occurrence strength** which means more recent interactions have higher weight.
We use time decay formula:
![alt text](image.png)

Weights from multiple sessions are aggregated to form a single weighted graph.

Graph built with **NetworkX** is converted to **PyTorch Geometric (PyG)** format to be able to use for downstream GNN models.  
- Node features include **one-hot encoded categories**.  
- Outputs:  
  - `data.x`: node features  
  - `data.edge_index`: edge indices  
  - `data.edge_attr`: edge weights or timestamps (for TGAT)

## Graph Encoders

### 1. **GraphSAGE**
- Learns node embeddings via **message passing and neighborhood aggregation**.  
- Trained using a **link prediction contrastive loss**.  
- Captures static co-occurrence structure but **not temporal evolution**.  

### 2. **TGAT (Temporal Graph Attention Network)**
- Extends GraphSAGE with **time-aware attention**.  
- Each message is weighted based on **temporal proximity**.  
- Learns **dynamic embeddings** that evolve with time.  
- Better suited for **session-based recommendation**, where recency matters.

---

## ⚙️ Sequential Layer (GRU4Rec)

- Once graph embeddings are trained (from GraphSAGE or TGAT),  
  they are used as input to **GRU4Rec**.  
- GRU4Rec models the **intra-session temporal sequence** of clicks.  
- The model predicts the **next likely item** in a session.

**Output:**  
Probability distribution over all items (`batch_size × num_items`).