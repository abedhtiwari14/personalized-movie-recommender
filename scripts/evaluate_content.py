"""Warm and cold evaluation of the content-based model.

Run from the project root:

    python scripts/evaluate_content.py

Two evaluations, reported separately because they answer different questions.

WARM  193,622 users ranking the 43,417 movies that have positive training
      signal. Content is expected to lose here: the behavioural models have
      strictly more information about every candidate.

COLD  3,479 users who have at least one test target with zero positive
      training interactions, ranking the full 87,585-movie catalog.
      Popularity, item-KNN and BPR score exactly zero by construction.
      This is the evaluation the content model exists for.

Scoring is batched: one sparse matmul per BATCH users rather than one per
user, which takes the warm run from ~19 min to a few.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from src.data import loaders  # noqa: E402
from src.models.content import build_content_features, user_profile  # noqa: E402
from src.utils.evaluation import EvaluationContext, evaluate_cold, evaluate_warm  # noqa: E402

K = 10
BATCH = 512


def batched_recommendations(X, idx, movie_ids, users, context, k=K, label=""):
    """Top-k for every user, scoring BATCH users per matmul."""
    recs = {}
    started = time.time()

    for start in range(0, len(users), BATCH):
        chunk = users[start:start + BATCH]

        profiles, keep = [], []
        for user_id in chunk:
            v = user_profile(X, idx, context.history_for(user_id))
            if v is not None:
                profiles.append(v)
                keep.append(user_id)

        if not profiles:
            continue

        scores = X @ np.array(profiles).T          # (n_movies, n_users)

        for col, user_id in enumerate(keep):
            s = scores[:, col].copy()

            seen_rows = [idx[m] for m in context.seen_for(user_id) if m in idx]
            s[seen_rows] = -np.inf

            top = np.argpartition(s, -k)[-k:]
            top = top[np.argsort(s[top])[::-1]]
            recs[user_id] = [movie_ids[i] for i in top if np.isfinite(s[i])]

        if start and start % (BATCH * 40) == 0:
            print(f"    {label}: {start:,}/{len(users):,} "
                  f"({(time.time() - started) / 60:.1f} min)", flush=True)

    return recs


def main():
    started = time.time()

    print("Building content features...", flush=True)
    features, movie_ids, vocab = build_content_features(use_tags=True)
    X_full = normalize(features)
    idx_full = {m: i for i, m in enumerate(movie_ids)}
    print(f"  {X_full.shape[0]:,} movies x {X_full.shape[1]:,} features, "
          f"{X_full.nnz:,} nonzero", flush=True)

    print("Loading evaluation context...", flush=True)
    context = EvaluationContext.load()
    print(f"  warm users: {len(context.users):,}   cold users: "
          f"{len(context.cold_users):,}", flush=True)

    # ---------------- WARM ----------------
    # Restrict rows to the warm universe so the candidate set matches the
    # other three models exactly, and so the matmul is half the size.
    warm_ids = np.array([m for m in movie_ids if m in set(
        context.warm_universe.tolist())])
    warm_rows = [idx_full[m] for m in warm_ids]
    X_warm = X_full[warm_rows]
    idx_warm = {m: i for i, m in enumerate(warm_ids)}

    print(f"\n===== WARM ({len(warm_ids):,} candidates) =====", flush=True)
    recs = batched_recommendations(X_warm, idx_warm, warm_ids, context.users,
                                   context, label="warm")
    warm = evaluate_warm(lambda u: recs.get(u, []), context, k=K,
                         label="Content (genres+tags)", verbose=False)
    print(f"  Content: R@{K}={warm[f'recall@{K}']:.4f}  "
          f"P@{K}={warm[f'precision@{K}']:.4f}  "
          f"NDCG@{K}={warm[f'ndcg@{K}']:.4f}", flush=True)

    # ---------------- COLD ----------------
    print(f"\n===== COLD ({len(movie_ids):,} candidates, full catalog) =====",
          flush=True)
    cold_recs = batched_recommendations(X_full, idx_full, movie_ids,
                                        context.cold_users, context, label="cold")
    cold = evaluate_cold(lambda u: cold_recs.get(u, []), context, k=K,
                         label="Content (genres+tags)")

    # ---------------- report ----------------
    pd.DataFrame([{
        "model": "Content (genres+tags)",
        f"recall@{K}": warm[f"recall@{K}"],
        f"precision@{K}": warm[f"precision@{K}"],
        f"ndcg@{K}": warm[f"ndcg@{K}"],
    }]).to_csv(os.path.join(loaders.PROCESSED_DIR, "leaderboard_content.csv"),
               index=False)

    print(f"\nTotal: {(time.time() - started) / 60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
