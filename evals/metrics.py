# eval/metrics.py
import numpy as np

def hit_at_k(pred_scores, true_items, k=10):
    topk = np.argsort(-pred_scores)[:k]
    return int(true_items in topk)

def mrr_at_k(pred_scores, true_items, k=10):
    ranks = np.argsort(-pred_scores)[:k]
    for i, item in enumerate(ranks, start=1):
        if item == true_items: return 1.0 / i
    return 0.0

def coverage_at_k(all_topk_lists, n_catalog, k=10):
    # all_topk_lists: list of lists of item indices predicted
    unique = set(i for lst in all_topk_lists for i in lst[:k])
    return len(unique) / n_catalog

def gini_coefficient(x):
    # x: list/array of counts (non-negative)
    # Implementation per standard formula
    x = np.asarray(x).astype(float)
    if x.size == 0: return 0.0
    x = x.flatten()
    if np.all(x == 0): return 0.0
    x_sorted = np.sort(x)
    n = x.size
    cumvals = np.cumsum(x_sorted)
    gini = (2.0 * np.sum((np.arange(1, n+1) * x_sorted)))/(n * np.sum(x_sorted)) - (n+1)/n
    return gini

def gini_at_k(all_topk_lists, n_items, k=10):
    # compute how concentrated exposures are across items
    counts = np.zeros(n_items, dtype=int)
    for lst in all_topk_lists:
        for item in lst[:k]:
            counts[item] += 1
    return gini_coefficient(counts)
