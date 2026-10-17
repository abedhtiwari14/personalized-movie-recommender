"""Recall@R by user-history bucket, under the consistent seen-set protocol.

Recall@K is capped at K / len(relevant) and so is not comparable across
buckets with different target counts. Recall@R sizes the list to each
user's target count and removes that ceiling.

    python scripts/segment_recall_at_r.py [n_users]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import loaders  # noqa: E402
from src.models.item_knn import ItemKNN  # noqa: E402
from src.utils.metrics import recall_at_r  # noqa: E402

DEFAULT_N_USERS = 20000

# Runtime cap on list length; users with more targets than this are scored
# against their first MAX_R recommendations.
MAX_R = 100

BUCKET_EDGES = [-1, 20, 50, 200, np.inf]
BUCKET_LABELS = ["1-20", "21-50", "51-200", "200+"]


def main(n_users=DEFAULT_N_USERS):
    started = time.time()

    train = loaders.load_split("train")
    test = loaders.load_split("test")
    train_positive = loaders.positives(train)

    relevant_by_user = loaders.relevant_by_user(test)
    history = loaders.positive_movies_by_user(train_positive)
    seen_all = train.groupby("userId")["movieId"].apply(set)
    empty = set()

    item_knn = ItemKNN.load()

    rng = np.random.default_rng(0)
    all_users = np.asarray(relevant_by_user.index)
    users = rng.choice(all_users, size=min(n_users, len(all_users)), replace=False)

    rows = []

    for user_id in users:
        relevant = relevant_by_user[user_id]
        n = min(len(relevant), MAX_R)

        recommended = item_knn.recommend(
            history.get(user_id, empty), k=n, seen=seen_all.get(user_id, empty),
        )

        rows.append({
            "userId": user_id,
            "history": len(history.get(user_id, empty)),
            "n_relevant": len(relevant),
            "recall_at_r": recall_at_r(recommended, relevant),
        })

    frame = pd.DataFrame(rows)
    frame["bucket"] = pd.cut(frame["history"], bins=BUCKET_EDGES, labels=BUCKET_LABELS)

    summary = frame.groupby("bucket", observed=True).agg(
        users=("userId", "size"),
        mean_relevant=("n_relevant", "mean"),
        recall_at_r=("recall_at_r", "mean"),
    )

    print(f"Recall@R by history bucket, consistent seen-set "
          f"(n={len(users):,} random users)\n")
    print(summary.round(4).to_string())
    print(f"\nDone in {(time.time() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_USERS)
