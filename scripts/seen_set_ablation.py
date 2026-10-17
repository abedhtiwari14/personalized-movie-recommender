"""Measure the seen-set asymmetry in Item-KNN.

Popularity and BPR exclude a user's full training history; Item-KNN as
originally published excluded only the positively-rated part. This reports
the size of that difference.

    python scripts/seen_set_ablation.py [n_users]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from src.data import loaders  # noqa: E402
from src.models.item_knn import ItemKNN  # noqa: E402
from src.utils.metrics import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

K = 10
DEFAULT_N_USERS = 20000


def main(n_users=DEFAULT_N_USERS):
    started = time.time()

    train = loaders.load_split("train")
    test = loaders.load_split("test")
    train_positive = loaders.positives(train)

    relevant_by_user = loaders.relevant_by_user(test)
    history = loaders.positive_movies_by_user(train_positive)
    seen_all = train.groupby("userId")["movieId"].apply(set)

    item_knn = ItemKNN.load()

    rng = np.random.default_rng(0)
    all_users = np.asarray(relevant_by_user.index)
    users = rng.choice(all_users, size=min(n_users, len(all_users)), replace=False)

    print(f"Item-KNN seen-set ablation over {len(users):,} randomly sampled users\n")

    empty = set()
    results = {}

    for label, seen_for in (
        ("published (positives only)", lambda u: None),
        ("consistent (full history)", lambda u: seen_all.get(u, empty)),
    ):
        recalls, precisions, ndcgs = [], [], []

        for user_id in users:
            user_history = history.get(user_id, empty)
            recommended = item_knn.recommend(user_history, k=K, seen=seen_for(user_id))
            relevant = relevant_by_user[user_id]

            recalls.append(recall_at_k(recommended, relevant, K))
            precisions.append(precision_at_k(recommended, relevant, K))
            ndcgs.append(ndcg_at_k(recommended, relevant, K))

        results[label] = np.asarray(recalls)

        print(f"  {label:28s}  R@{K}={np.mean(recalls):.4f}  "
              f"P@{K}={np.mean(precisions):.4f}  NDCG@{K}={np.mean(ndcgs):.4f}")

    published, consistent = results.values()
    delta = consistent.mean() - published.mean()

    print(f"\n  difference: {delta:+.5f} absolute "
          f"({delta / published.mean() * 100:+.2f}% relative)")

    if np.allclose(published, consistent):
        print("  identical on every sampled user — the asymmetry has no effect.")
    else:
        _, p_value = wilcoxon(published, consistent)
        changed = int((published != consistent).sum())
        print(f"  users whose recall changed: {changed:,}/{len(users):,}")
        print(f"  Wilcoxon p = {p_value:.3e}")

    print(f"\nDone in {(time.time() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_USERS)
