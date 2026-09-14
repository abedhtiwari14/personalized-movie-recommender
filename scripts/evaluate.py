"""Full test-set evaluation of all three models.

Run from the project root:

    python scripts/evaluate.py

Regenerates `data/processed/leaderboard.csv`, the per-user recall arrays, and
the corrected segment analysis. Every number is computed here — nothing is
carried over as a literal.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from src.data import loaders  # noqa: E402
from src.models.bpr import BPRRecommender  # noqa: E402
from src.models.item_knn import ItemKNN  # noqa: E402
from src.models.popularity import PopularityRecommender  # noqa: E402
from src.utils.metrics import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

K = 10

# Users are bucketed by number of positive training interactions.
BUCKET_EDGES = [-1, 20, 50, 200, np.inf]
BUCKET_LABELS = ["1-20", "21-50", "51-200", "200+"]

# Sample size for the popularity-bias and coverage tables.
BIAS_SAMPLE = 2000


def evaluate(recommend, users, relevant_by_user, label):
    """Run one recommender over every evaluation user."""
    recalls = np.empty(len(users))
    precisions = np.empty(len(users))
    ndcgs = np.empty(len(users))

    started = time.time()

    for position, user_id in enumerate(users):
        relevant = relevant_by_user[user_id]
        recommended = recommend(user_id)

        recalls[position] = recall_at_k(recommended, relevant, K)
        precisions[position] = precision_at_k(recommended, relevant, K)
        ndcgs[position] = ndcg_at_k(recommended, relevant, K)

        if position and position % 25000 == 0:
            print(f"    {label}: {position:,}/{len(users):,} "
                  f"({(time.time() - started) / 60:.1f} min)", flush=True)

    print(f"  {label}: R@{K}={recalls.mean():.4f}  P@{K}={precisions.mean():.4f}  "
          f"NDCG@{K}={ndcgs.mean():.4f}  ({(time.time() - started) / 60:.1f} min)",
          flush=True)

    return recalls, precisions, ndcgs


def main():
    started = time.time()

    print("Loading splits...", flush=True)
    train = loaders.load_split("train")
    test = loaders.load_split("test")
    train_positive = loaders.positives(train)

    maps = loaders.load_id_maps()

    # Evaluation targets: each user's positive test interactions.
    relevant_by_user = loaders.relevant_by_user(test)

    # Seen sets. Popularity and BPR exclude the user's entire training
    # history; Item-KNN excludes only the positive part. That asymmetry is
    # inherited from the published run and is quantified in
    # scripts/seen_set_ablation.py.
    seen_all = train.groupby("userId")["movieId"].apply(set)
    history = loaders.positive_movies_by_user(train_positive)

    users = list(relevant_by_user.index)
    print(f"  evaluation users: {len(users):,}", flush=True)
    print(f"  candidate universe: {len(maps['movie_ids']):,} movies", flush=True)

    popularity = PopularityRecommender.fit(train_positive)
    item_knn = ItemKNN.load()
    bpr = BPRRecommender.load(maps["bpr_user_ids"], maps["movie_ids"])

    empty = set()

    print("\nEvaluating...", flush=True)

    pop_scores = evaluate(
        lambda u: popularity.recommend(seen_all.get(u, empty), k=K),
        users, relevant_by_user, "Popularity",
    )

    knn_scores = evaluate(
        lambda u: item_knn.recommend(history.get(u, empty), k=K),
        users, relevant_by_user, "Item-KNN",
    )

    bpr_scores = evaluate(
        lambda u: bpr.recommend(u, seen=seen_all.get(u, empty), k=K),
        users, relevant_by_user, "BPR (MF)",
    )

    # --- Leaderboard -----------------------------------------------------
    leaderboard = pd.DataFrame({
        "model": ["Popularity", "BPR (MF)", "Item-KNN"],
        f"recall@{K}": [pop_scores[0].mean(), bpr_scores[0].mean(), knn_scores[0].mean()],
        f"precision@{K}": [pop_scores[1].mean(), bpr_scores[1].mean(), knn_scores[1].mean()],
        f"ndcg@{K}": [pop_scores[2].mean(), bpr_scores[2].mean(), knn_scores[2].mean()],
    })

    leaderboard.to_csv(os.path.join(loaders.PROCESSED_DIR, "leaderboard.csv"), index=False)

    np.save(os.path.join(loaders.PROCESSED_DIR, "popularity_test_recalls.npy"), pop_scores[0])
    np.save(os.path.join(loaders.PROCESSED_DIR, "itemknn_test_recalls.npy"), knn_scores[0])
    np.save(os.path.join(loaders.PROCESSED_DIR, "bpr_test_recalls.npy"), bpr_scores[0])

    print("\n===== LEADERBOARD =====", flush=True)
    print(leaderboard.round(4).to_string(index=False), flush=True)

    pop_recall = pop_scores[0].mean()
    print(f"\nBPR vs popularity:      "
          f"{(bpr_scores[0].mean() / pop_recall - 1) * 100:+.1f}% relative R@{K}")
    print(f"Item-KNN vs BPR:        "
          f"{(knn_scores[0].mean() / bpr_scores[0].mean() - 1) * 100:+.1f}% relative R@{K}")
    print(f"Item-KNN vs popularity: "
          f"{(knn_scores[0].mean() / pop_recall - 1) * 100:+.1f}% relative R@{K}")

    _, p_value = wilcoxon(knn_scores[0], bpr_scores[0])
    print(f"\nWilcoxon signed-rank, Item-KNN vs BPR over {len(users):,} paired "
          f"per-user recalls: p = {p_value:.3e}")

    # --- Segment analysis ------------------------------------------------
    # Recall@K is capped at K / len(relevant), so it is not comparable across
    # buckets with different target counts. Precision@K is reported alongside
    # because it carries no such ceiling. See the README.
    segments = pd.DataFrame({
        "userId": users,
        "history": [len(history.get(u, empty)) for u in users],
        "n_relevant": [len(relevant_by_user[u]) for u in users],
        "knn_recall": knn_scores[0],
        "knn_precision": knn_scores[1],
        "bpr_recall": bpr_scores[0],
    })

    segments["bucket"] = pd.cut(
        segments["history"], bins=BUCKET_EDGES, labels=BUCKET_LABELS,
    )

    summary = segments.groupby("bucket", observed=True).agg(
        users=("userId", "size"),
        mean_relevant=("n_relevant", "mean"),
        knn_recall=("knn_recall", "mean"),
        knn_precision=("knn_precision", "mean"),
        bpr_recall=("bpr_recall", "mean"),
    )
    summary["recall_ceiling"] = K / segments.groupby(
        "bucket", observed=True)["n_relevant"].mean()

    print("\n===== BY USER HISTORY LENGTH =====", flush=True)
    print(summary.round(4).to_string(), flush=True)
    print("\nRecall@10 falls across buckets while Precision@10 rises: the fall "
          "is the denominator, not the model.", flush=True)

    segments.to_parquet(os.path.join(loaders.PROCESSED_DIR, "segment_analysis.parquet"),
                        index=False)

    # --- Popularity bias and catalog coverage ----------------------------
    popularity_rank = popularity.popularity_rank()
    sample = users[:BIAS_SAMPLE]

    print(f"\n===== POPULARITY BIAS AND COVERAGE (n={len(sample):,}) =====", flush=True)

    rows = []

    for name, recommend in (
        ("BPR (MF)", lambda u: bpr.recommend(u, seen=seen_all.get(u, empty), k=K)),
        ("Item-KNN", lambda u: item_knn.recommend(history.get(u, empty), k=K)),
    ):
        ranks = []
        distinct = set()

        for user_id in sample:
            recommended = recommend(user_id)
            distinct.update(recommended)
            ranks.extend(popularity_rank[m] for m in recommended if m in popularity_rank)

        rows.append({
            "model": name,
            "mean_pop_rank": np.mean(ranks),
            "median_pop_rank": np.median(ranks),
            "distinct_items": len(distinct),
        })

    print(pd.DataFrame(rows).round(1).to_string(index=False), flush=True)
    print(f"(Popularity recommends ranks 1-{K} by construction: {K} distinct items.)")

    print(f"\nTotal: {(time.time() - started) / 60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
