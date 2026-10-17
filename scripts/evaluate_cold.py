"""Cold benchmark for the behavioural models.

Scores Popularity, Item-KNN and BPR over the full catalog for users with at
least one test target that has no training signal. All three score zero: a
cold item has no popularity count, no neighbour row and no embedding row.

    python scripts/evaluate_cold.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from src.data import loaders  # noqa: E402
from src.models.bpr import BPRRecommender  # noqa: E402
from src.models.item_knn import ItemKNN  # noqa: E402
from src.models.popularity import PopularityRecommender  # noqa: E402
from src.utils.evaluation import K, EvaluationContext, evaluate_cold  # noqa: E402

REPORT_COLUMNS = ["model", f"cold_recall@{K}", "cold_items_surfaced",
                  f"overall_recall@{K}"]


def main():
    started = time.time()

    print("Loading evaluation context...", flush=True)
    context = EvaluationContext.load()

    print(f"  cold users: {len(context.cold_users):,}", flush=True)
    print(f"  cold items: {len(context.cold_items):,}", flush=True)
    print(f"  candidates: {len(context.full_universe):,}\n", flush=True)

    train_positive = loaders.positives(loaders.load_split("train"))
    maps = loaders.load_id_maps()

    popularity = PopularityRecommender.fit(train_positive)
    item_knn = ItemKNN.load()
    bpr = BPRRecommender.load(maps["bpr_user_ids"], maps["movie_ids"])

    recommenders = {
        "Popularity": lambda u: popularity.recommend(context.seen_for(u), k=K),
        "Item-KNN": lambda u: item_knn.recommend(
            context.history_for(u), k=K, seen=context.seen_for(u)),
        "BPR (MF)": lambda u: bpr.recommend(u, seen=context.seen_for(u), k=K),
    }

    rows = [
        evaluate_cold(recommend, context, k=K, label=name)
        for name, recommend in recommenders.items()
    ]

    table = pd.DataFrame(rows)[REPORT_COLUMNS]
    table.to_csv(os.path.join(loaders.PROCESSED_DIR, "leaderboard_cold.csv"),
                 index=False)

    print("\n" + table.round(4).to_string(index=False), flush=True)
    print("\nAll three score zero by construction: a cold item has no "
          "popularity count, no neighbour row, no embedding row.", flush=True)
    print(f"\nDone in {(time.time() - started) / 60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
