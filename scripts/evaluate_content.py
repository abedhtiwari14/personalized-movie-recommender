"""Warm and cold evaluation of the content-based models.

Run from the project root:

    python scripts/evaluate_content.py

Two evaluations, reported separately because they answer different questions:

WARM  — the existing benchmark: 193,622 users ranking 43,417 movies that all
        have positive training signal. Content is expected to lose here; the
        behavioural models have strictly more information about every
        candidate. Reported for completeness and to calibrate a hybrid.

COLD  — users with at least one test target that has zero positive training
        interactions, ranking the full 87,585-movie catalog. Popularity,
        Item-KNN and BPR score exactly zero here by construction: a cold item
        has no popularity count, no neighbour row and no embedding row. This
        is the evaluation the content model exists for.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import loaders  # noqa: E402
from src.models.bpr import BPRRecommender  # noqa: E402
from src.models.content import ContentRecommender  # noqa: E402
from src.models.item_knn import ItemKNN  # noqa: E402
from src.models.popularity import PopularityRecommender  # noqa: E402
from src.utils.metrics import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

K = 10
VARIANTS = ("genres", "genres_tags")


def main():
    started = time.time()

    print("Loading...", flush=True)
    train = loaders.load_split("train")
    test = loaders.load_split("test")
    train_positive = loaders.positives(train)

    relevant_by_user = loaders.relevant_by_user(test)
    history = loaders.positive_movies_by_user(train_positive)
    seen_all = train.groupby("userId")["movieId"].apply(set)
    empty = set()

    warm = loaders.warm_universe(train_positive)
    cold = loaders.cold_items(train_positive, test)
    cold_set = set(cold.tolist())

    users = list(relevant_by_user.index)

    print(f"  eval users: {len(users):,}   warm universe: {len(warm):,}   "
          f"cold items: {len(cold):,}", flush=True)

    content = {v: ContentRecommender.load(v) for v in VARIANTS}

    # ---------------- WARM ----------------
    print("\n===== WARM BENCHMARK (43,417 candidates, all models) =====", flush=True)

    rows = []

    for variant in VARIANTS:
        model = content[variant].restrict_to(warm)
        recalls, precisions, ndcgs = [], [], []
        t0 = time.time()

        for n, user_id in enumerate(users):
            rec = model.recommend(history.get(user_id, empty),
                                  seen=seen_all.get(user_id, empty), k=K)
            relevant = relevant_by_user[user_id]

            recalls.append(recall_at_k(rec, relevant, K))
            precisions.append(precision_at_k(rec, relevant, K))
            ndcgs.append(ndcg_at_k(rec, relevant, K))

            if n and n % 50000 == 0:
                print(f"    content-{variant}: {n:,}/{len(users):,} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)

        rows.append({
            "model": f"Content ({variant})",
            f"recall@{K}": np.mean(recalls),
            f"precision@{K}": np.mean(precisions),
            f"ndcg@{K}": np.mean(ndcgs),
        })
        print(f"  Content ({variant}): R@{K}={np.mean(recalls):.4f}  "
              f"P@{K}={np.mean(precisions):.4f}  NDCG@{K}={np.mean(ndcgs):.4f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

        np.save(os.path.join(loaders.PROCESSED_DIR,
                             f"content_{variant}_test_recalls.npy"), np.asarray(recalls))

    warm_table = pd.DataFrame(rows)
    warm_table.to_csv(os.path.join(loaders.PROCESSED_DIR, "leaderboard_content.csv"),
                      index=False)
    print("\n" + warm_table.round(4).to_string(index=False), flush=True)

    # ---------------- COLD ----------------
    cold_users = [u for u in users if relevant_by_user[u] & cold_set]

    print(f"\n===== COLD BENCHMARK (87,585 candidates) =====", flush=True)
    print(f"  users with >=1 cold target: {len(cold_users):,}", flush=True)

    popularity = PopularityRecommender.fit(train_positive)
    item_knn = ItemKNN.load()
    maps = loaders.load_id_maps()
    bpr = BPRRecommender.load(maps["bpr_user_ids"], maps["movie_ids"])

    behavioural = {
        "Popularity": lambda u: popularity.recommend(seen_all.get(u, empty), k=K),
        "Item-KNN": lambda u: item_knn.recommend(history.get(u, empty), k=K,
                                                 seen=seen_all.get(u, empty)),
        "BPR (MF)": lambda u: bpr.recommend(u, seen=seen_all.get(u, empty), k=K),
    }
    content_models = {
        f"Content ({v})": (lambda m: lambda u: m.recommend(
            history.get(u, empty), seen=seen_all.get(u, empty), k=K))(content[v])
        for v in VARIANTS
    }

    cold_rows = []

    for name, recommend in {**behavioural, **content_models}.items():
        cold_recalls, overall_recalls, cold_hits = [], [], 0
        t0 = time.time()

        for user_id in cold_users:
            rec = recommend(user_id)
            relevant = relevant_by_user[user_id]
            cold_targets = relevant & cold_set

            hits = len(set(rec[:K]) & cold_targets)
            cold_hits += hits

            cold_recalls.append(hits / len(cold_targets))
            overall_recalls.append(recall_at_k(rec, relevant, K))

        cold_rows.append({
            "model": name,
            f"cold_recall@{K}": np.mean(cold_recalls),
            "cold_items_surfaced": cold_hits,
            f"overall_recall@{K}": np.mean(overall_recalls),
        })
        print(f"  {name:24s} cold R@{K}={np.mean(cold_recalls):.4f}  "
              f"hits={cold_hits:,}  overall R@{K}={np.mean(overall_recalls):.4f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    cold_table = pd.DataFrame(cold_rows)
    cold_table.to_csv(os.path.join(loaders.PROCESSED_DIR, "leaderboard_cold.csv"),
                      index=False)

    print("\n" + cold_table.round(4).to_string(index=False), flush=True)
    print("\nPopularity, Item-KNN and BPR cannot rank a cold item at all: no "
          "popularity count, no neighbour row, no embedding row.", flush=True)
    print(f"\nTotal: {(time.time() - started) / 60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
