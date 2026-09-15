"""Benchmark harness: one interface, two benchmarks.

A recommender is anything satisfying `Callable[[int], Sequence[int]]` — given
a userId, return a ranked list of movieIds, best first. Nothing else is
required of it. Both benchmarks below take that callable, so a new model is
evaluated without touching this module.

    WARM  193,622 users ranking the 43,417 movies that have positive training
          signal. Every candidate has behavioural history.

    COLD  3,479 users who have at least one test target with zero positive
          training interactions, ranking the full 87,585-movie catalog.
          Popularity, Item-KNN and BPR score exactly zero here by
          construction — no popularity count, no neighbour row, no embedding
          row.

The two are reported separately because they answer different questions, and
a single blended number would hide both.
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import loaders
from .metrics import ndcg_at_k, precision_at_k, recall_at_k

K = 10

EMPTY = frozenset()


@dataclass
class EvaluationContext:
    """Everything both benchmarks need, loaded once.

    Loading costs ~40s and several GB, almost all of it the per-user seen
    sets. Build one and reuse it across models.
    """

    relevant_by_user: pd.Series   # userId -> set of positive TEST movieIds
    history: pd.Series            # userId -> set of positive TRAIN movieIds
    seen: pd.Series               # userId -> set of ALL TRAIN movieIds
    warm_universe: np.ndarray     # 43,417 movieIds
    full_universe: np.ndarray     # 87,585 movieIds
    cold_items: np.ndarray        # 7,879 movieIds
    users: list                   # 193,622 warm-benchmark users
    cold_users: list              # 3,479 cold-benchmark users

    @classmethod
    def load(cls):
        train = loaders.load_split("train")
        test = loaders.load_split("test")
        train_positive = loaders.positives(train)

        relevant_by_user = loaders.relevant_by_user(test)
        cold = loaders.cold_items(train_positive, test)
        cold_set = set(cold.tolist())

        users = list(relevant_by_user.index)

        return cls(
            relevant_by_user=relevant_by_user,
            history=loaders.positive_movies_by_user(train_positive),
            seen=train.groupby("userId")["movieId"].apply(set),
            warm_universe=loaders.warm_universe(train_positive),
            full_universe=loaders.full_universe(),
            cold_items=cold,
            users=users,
            cold_users=[u for u in users if relevant_by_user[u] & cold_set],
        )

    def history_for(self, user_id):
        """The user's positive training movies — a model's input signal."""
        return self.history.get(user_id, EMPTY)

    def seen_for(self, user_id):
        """The user's full training history — what every model must exclude.

        Excluding only the positive part lets a model spend top-K slots on
        movies the user already rated below 4, and costs it ~16% relative.
        Use this, not history_for, as the exclusion set.
        """
        return self.seen.get(user_id, EMPTY)

    def cold_targets_for(self, user_id):
        """This user's test targets that have no positive training signal."""
        return self.relevant_by_user[user_id] & set(self.cold_items.tolist())


def evaluate_warm(recommend, context, k=K, users=None, label="model", verbose=True):
    """Score a recommender on the warm benchmark.

    `recommend` takes a userId and returns a ranked list of movieIds; it is
    responsible for its own candidate restriction and seen-exclusion. Returns
    aggregate means plus the per-user recall array for significance testing.
    """
    users = context.users if users is None else users

    recalls = np.empty(len(users))
    precisions = np.empty(len(users))
    ndcgs = np.empty(len(users))

    started = time.time()

    for position, user_id in enumerate(users):
        relevant = context.relevant_by_user[user_id]
        recommended = recommend(user_id)

        recalls[position] = recall_at_k(recommended, relevant, k)
        precisions[position] = precision_at_k(recommended, relevant, k)
        ndcgs[position] = ndcg_at_k(recommended, relevant, k)

        if verbose and position and position % 50000 == 0:
            print(f"    {label}: {position:,}/{len(users):,} "
                  f"({(time.time() - started) / 60:.1f} min)", flush=True)

    return {
        "model": label,
        "users": len(users),
        f"recall@{k}": recalls.mean(),
        f"precision@{k}": precisions.mean(),
        f"ndcg@{k}": ndcgs.mean(),
        "per_user_recall": recalls,
        "minutes": (time.time() - started) / 60,
    }


def evaluate_cold(recommend, context, k=K, label="model", verbose=True):
    """Score a recommender on the cold benchmark.

    `cold_recall@k` counts only cold targets: of this user's test positives
    that have no training signal, how many appear in the top k. It is the
    metric no behavioural model can score above zero on.

    `overall_recall@k` counts all of the user's test targets over the same
    ranking, so a model that surfaces cold items at the cost of everything
    else is visible rather than flattered.
    """
    cold_set = set(context.cold_items.tolist())

    cold_recalls = np.empty(len(context.cold_users))
    overall_recalls = np.empty(len(context.cold_users))
    surfaced = 0

    started = time.time()

    for position, user_id in enumerate(context.cold_users):
        relevant = context.relevant_by_user[user_id]
        cold_targets = relevant & cold_set

        recommended = recommend(user_id)
        hits = len(set(recommended[:k]) & cold_targets)
        surfaced += hits

        cold_recalls[position] = hits / len(cold_targets)
        overall_recalls[position] = recall_at_k(recommended, relevant, k)

    if verbose:
        print(f"  {label}: cold R@{k}={cold_recalls.mean():.4f}  "
              f"surfaced={surfaced:,}  overall R@{k}={overall_recalls.mean():.4f}  "
              f"({(time.time() - started) / 60:.1f} min)", flush=True)

    return {
        "model": label,
        "users": len(context.cold_users),
        f"cold_recall@{k}": cold_recalls.mean(),
        "cold_items_surfaced": surfaced,
        f"overall_recall@{k}": overall_recalls.mean(),
        "per_user_cold_recall": cold_recalls,
        "minutes": (time.time() - started) / 60,
    }
