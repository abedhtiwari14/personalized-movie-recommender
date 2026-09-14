"""Loading the processed splits and the canonical id mappings.

Every model in this project indexes matrices and embeddings by *position*,
not by movieId. The position -> movieId mapping is therefore part of the
model, and a saved artifact that does not carry its own mapping is unusable.
An earlier version of this project saved item neighbours without one, against
a mapping that was later rebound — the neighbours silently pointed into a
different id space. Everything here exists to make that unrepresentable.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ml-32m")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SPLITS = ("train", "validation", "test")

# Positive interaction threshold. Ratings below this are not negatives —
# they simply carry no positive signal (see README, "Dataset").
POSITIVE_RATING = 4.0


def split_path(split):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    return os.path.join(PROCESSED_DIR, f"{split}_personalized.parquet")


def load_split(split, columns=("userId", "movieId", "positive")):
    """Load one per-user chronological split.

    Row order is the global chronological order the splits were written in,
    and the id mappings below depend on it — do not sort the result.
    """
    return pd.read_parquet(split_path(split), columns=list(columns))


def load_movies():
    return pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))


def positives(frame):
    """Keep only positive interactions, preserving row order."""
    return frame[frame["positive"] == 1]


def build_id_maps(train_positive):
    """Derive the canonical position <-> id mappings from positive training data.

    Two distinct user orderings exist and both must be preserved exactly, or
    the saved BPR checkpoint becomes meaningless:

    - `movie_ids`: order of first appearance in the training split. Indexes
      the Item-KNN item axis *and* the BPR movie embedding rows.
    - `knn_user_ids`: order of first appearance. Indexes the Item-KNN user
      axis, which is only a feature axis for cosine similarity — its order
      does not affect results.
    - `bpr_user_ids`: sorted ascending (a groupby side effect at training
      time). Indexes the BPR user embedding rows.
    """
    movie_ids = train_positive["movieId"].unique()
    knn_user_ids = train_positive["userId"].unique()
    bpr_user_ids = np.sort(train_positive["userId"].unique())

    return {
        "movie_ids": np.asarray(movie_ids),
        "knn_user_ids": np.asarray(knn_user_ids),
        "bpr_user_ids": np.asarray(bpr_user_ids),
    }


def index_map(ids):
    """id -> position lookup."""
    return {item_id: idx for idx, item_id in enumerate(ids)}


def id_maps_path():
    return os.path.join(PROCESSED_DIR, "id_maps.npz")


def save_id_maps(maps):
    np.savez(id_maps_path(), **maps)

    return id_maps_path()


def load_id_maps():
    path = id_maps_path()

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run scripts/build_item_neighbors.py first."
        )

    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def positive_movies_by_user(train_positive):
    """user id -> set of positively-interacted training movie ids."""
    return (
        train_positive
        .groupby("userId")["movieId"]
        .apply(set)
    )


def relevant_by_user(split_frame):
    """user id -> set of positive movie ids in the given split (eval targets)."""
    return (
        positives(split_frame)
        .groupby("userId")["movieId"]
        .apply(set)
    )
