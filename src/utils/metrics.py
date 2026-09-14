"""Ranking metrics for Top-K recommendation.

All functions take a ranked list of recommended item ids (best first), a set
of relevant item ids, and a cutoff K. They are order-sensitive only through
NDCG; recall and precision depend on set overlap within the top K.
"""

import numpy as np


def recall_at_k(recommended, relevant, k):
    """Fraction of relevant items that appear in the top-k recommendations.

    Note the denominator is the user's own target count, so this metric is
    capped at k / len(relevant). It is valid for comparing models on a fixed
    population, not for comparing populations with different target counts.
    See the README, "A segment finding that was a metric artifact".
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
    """Normalized Discounted Cumulative Gain with binary relevance.

    The ideal ranking places min(len(relevant), k) hits in the top positions.
    """
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
    """Recall with the list length set to each user's own target count.

    Removes the denominator ceiling that makes recall_at_k incomparable
    across user segments. `recommended` must already be at least
    len(relevant) items long.
    """
    relevant = set(relevant)

    if len(relevant) == 0:
        return 0.0

    hits = len(set(recommended[:len(relevant)]) & relevant)

    return hits / len(relevant)
