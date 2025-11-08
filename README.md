# Temporal Graph Embeddings Session-Based Recommendation (TGESR)

The goal of this project is to create a hybrid recommendation systme that uses both:
- sequential patterns: what items users click in order
- graph relationships: how items co-occur in a sequence and are related to each other

We do this by using learned item graph embeddings using GNN models like GAT or GraphSAGE and then passing these embeddings to a seesion based model like GRU4Rec which takes in a sequence of items and tries to predict the next most probable item the user will interact with.

## Dataset 
We are usign the YooChoose dataset form Recsys 2025 challenge which has session data for user's clicks/item interaction on an online platform. A row in the dataset looks like (session_id, item_id, timestamp, category)