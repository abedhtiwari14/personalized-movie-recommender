"""Assertion-backed tests for the ranking metrics.

These replace the printed sanity checks that previously lived in notebook
cells. Run with `pytest tests/` or `python tests/test_metrics.py`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.metrics import (  # noqa: E402
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recall_at_r,
)


def test_readme_hand_computed_case():
    """The case quoted in the README: hits at ranks 1, 3, 7; 4 relevant items.

    DCG  = 1/log2(2) + 1/log2(4) + 1/log2(8)             = 1.833333
    IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(5) = 2.561606
    NDCG = 0.715701
    """
    recommended = [101, 200, 102, 300, 400, 500, 103, 600, 700, 800]
    relevant = {101, 102, 103, 999}

    assert recall_at_k(recommended, relevant, 10) == 0.75
    assert precision_at_k(recommended, relevant, 10) == 0.3
    assert abs(ndcg_at_k(recommended, relevant, 10) - 0.7157) < 1e-4


def test_small_case():
    """Hits at ranks 2 and 4 out of 3 relevant items, K = 5."""
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 40, 100}

    assert abs(recall_at_k(recommended, relevant, 5) - 2 / 3) < 1e-12
    assert precision_at_k(recommended, relevant, 5) == 0.4
    assert abs(ndcg_at_k(recommended, relevant, 5) - 0.4981892574664128) < 1e-12


def test_perfect_and_empty_rankings():
    relevant = {1, 2, 3}

    assert recall_at_k([1, 2, 3, 4, 5], relevant, 3) == 1.0
    assert precision_at_k([1, 2, 3], relevant, 3) == 1.0
    assert ndcg_at_k([1, 2, 3, 4, 5], relevant, 3) == 1.0

    assert recall_at_k([7, 8, 9], relevant, 3) == 0.0
    assert precision_at_k([7, 8, 9], relevant, 3) == 0.0
    assert ndcg_at_k([7, 8, 9], relevant, 3) == 0.0


def test_ndcg_rewards_earlier_hits():
    relevant = {1}

    early = ndcg_at_k([1, 2, 3, 4, 5], relevant, 5)
    late = ndcg_at_k([2, 3, 4, 5, 1], relevant, 5)

    assert early > late
    assert early == 1.0


def test_truncation_at_k():
    """Items beyond K must not count, for any metric."""
    recommended = [9, 9, 9, 1, 2]
    relevant = {1, 2}

    assert recall_at_k(recommended, relevant, 3) == 0.0
    assert precision_at_k(recommended, relevant, 3) == 0.0
    assert ndcg_at_k(recommended, relevant, 3) == 0.0
    assert recall_at_k(recommended, relevant, 5) == 1.0


def test_empty_relevant_set_is_zero_not_error():
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0
    assert ndcg_at_k([1, 2, 3], set(), 3) == 0.0
    assert recall_at_r([1, 2, 3], set()) == 0.0


def test_recall_at_k_denominator_ceiling():
    """The artifact the README documents, stated as an executable fact.

    A user with 50 relevant items cannot exceed 10/50 at K=10 even with a
    perfect ranking, while a user with 2 relevant items reaches 1.0. That
    gap is the metric's, not the model's.
    """
    heavy_relevant = set(range(50))
    sparse_relevant = {0, 1}

    heavy_perfect = recall_at_k(list(range(50)), heavy_relevant, 10)
    sparse_perfect = recall_at_k([0, 1] + list(range(2, 10)), sparse_relevant, 10)

    assert heavy_perfect == 0.2
    assert sparse_perfect == 1.0

    # Recall@R has no such ceiling: both perfect rankings score 1.0.
    assert recall_at_r(list(range(50)), heavy_relevant) == 1.0
    assert recall_at_r([0, 1] + list(range(2, 10)), sparse_relevant) == 1.0


def test_recall_at_r_partial():
    """4 relevant items, 2 of them inside the first 4 positions."""
    recommended = [1, 99, 2, 98, 3, 4]
    relevant = {1, 2, 3, 4}

    assert recall_at_r(recommended, relevant) == 0.5


if __name__ == "__main__":
    failures = 0

    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")

    print()
    print("All metric tests passed." if failures == 0 else f"{failures} test(s) failed.")
    sys.exit(1 if failures else 0)
