"""Processed splits and the canonical position <-> id mappings.

Models index matrices and embeddings by row position, so the
position -> movieId mapping is part of the model and must travel with any
artifact that stores positions.
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ml-32m")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SPLITS = ("train", "validation", "test")

# Ratings below this carry no positive signal; they are not negatives.
POSITIVE_RATING = 4.0


def split_path(split):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    return os.path.join(PROCESSED_DIR, f"{split}_personalized.parquet")


def load_split(split, columns=("userId", "movieId", "positive")):
    """Load one per-user chronological split.

    Row order is chronological and build_id_maps depends on it; do not sort.
    """
    return pd.read_parquet(split_path(split), columns=list(columns))


def load_movies():
    return pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))


def positives(frame):
    """Keep only positive interactions, preserving row order."""
    return frame[frame["positive"] == 1]


def build_id_maps(train_positive):
    """Position <-> id mappings derived from positive training data.

    Three mappings, and the orderings are not interchangeable:

    movie_ids    order of first appearance; indexes the Item-KNN item axis
                 and the BPR movie embedding rows
    knn_user_ids order of first appearance; Item-KNN user axis only
    bpr_user_ids sorted ascending; indexes the BPR user embedding rows
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


def full_universe(movies=None):
    """Every movieId in the catalog, sorted. Candidate set for cold evaluation."""
    movies = load_movies() if movies is None else movies

    return np.sort(movies["movieId"].unique())


def warm_universe(train_positive):
    """Movies with at least one positive training interaction."""
    return np.asarray(train_positive["movieId"].unique())


def cold_items(train_positive, test_frame):
    """Test targets with no positive training interactions."""
    warm = set(warm_universe(train_positive).tolist())
    test_positive = positives(test_frame)

    return np.sort(
        test_positive.loc[~test_positive["movieId"].isin(warm), "movieId"].unique()
    )


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
