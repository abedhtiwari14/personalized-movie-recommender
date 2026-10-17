"""Ranking metrics for Top-K recommendation.

Each function takes a ranked list of item ids (best first), a set of
relevant ids, and a cutoff k.
"""

import numpy as np


def recall_at_k(recommended, relevant, k):
    """Fraction of relevant items appearing in the top k.

    Capped at k / len(relevant), so this is comparable across models on a
    fixed population but not across populations with different target counts.
    """
    relevant = set(relevant)

    if len(relevant) == 0:
        return 0.0

    hits = len(set(recommended[:k]) & relevant)

    return hits / len(relevant)


def precision_at_k(recommended, relevant, k):
    """Fraction of the top-k recommendations that are relevant."""
    if k == 0:
        return 0.0

    hits = len(set(recommended[:k]) & set(relevant))

    return hits / k


def ndcg_at_k(recommended, relevant, k):
    """NDCG with binary relevance; ideal ranking is min(len(relevant), k) hits."""
    relevant = set(relevant)

    if len(relevant) == 0:
        return 0.0

    dcg = 0.0

    for rank, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            dcg += 1 / np.log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def recall_at_r(recommended, relevant):
    """Recall with the list length set to len(relevant).

    Removes the ceiling in recall_at_k. `recommended` must be at least
    len(relevant) items long.
    """
    relevant = set(relevant)

    if len(relevant) == 0:
        return 0.0

    hits = len(set(recommended[:len(relevant)]) & relevant)

    return hits / len(relevant)
