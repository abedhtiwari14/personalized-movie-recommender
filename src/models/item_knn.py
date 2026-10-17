"""Item-item collaborative filtering.

Cosine similarity between movies over a sparse user x movie matrix of
positive interactions. Neighbours are precomputed offline, so serving is a
lookup and aggregate.

The artifact stores `indices`, `similarities` and `movie_ids` in one .npz:
neighbour rows are positions in `movie_ids` and are meaningless without it.
"""

import os

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from ..data.loaders import PROCESSED_DIR, index_map

N_NEIGHBORS = 51

NEIGHBORS_FILENAME = "item_neighbors.npz"


def neighbors_path():
    return os.path.join(PROCESSED_DIR, NEIGHBORS_FILENAME)


def build_user_movie_matrix(train_positive, user_ids, movie_ids):
    """Sparse CSR matrix of positive interactions (users x movies)."""
    user_to_idx = index_map(user_ids)
    movie_to_idx = index_map(movie_ids)

    rows = train_positive["userId"].map(user_to_idx).to_numpy()
    cols = train_positive["movieId"].map(movie_to_idx).to_numpy()
    data = np.ones(len(train_positive), dtype=np.float32)

    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    )


def build_neighbors(train_positive, user_ids, movie_ids, n_neighbors=N_NEIGHBORS):
    """Cosine nearest neighbours of every movie.

    Returns (indices, similarities), shape (len(movie_ids), n_neighbors).
    Each row's first entry is normally the movie itself at similarity 1.0.
    """
    user_movie = build_user_movie_matrix(train_positive, user_ids, movie_ids)
    movie_user = user_movie.T.tocsr()

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=n_neighbors,
        n_jobs=-1,
    )
    model.fit(movie_user)

    distances, indices = model.kneighbors(movie_user, n_neighbors=n_neighbors)

    return indices, 1 - distances


def save_neighbors(indices, similarities, movie_ids, path=None):
    path = path or neighbors_path()

    if indices.shape != similarities.shape:
        raise ValueError("indices and similarities must have the same shape")

    if indices.shape[0] != len(movie_ids):
        raise ValueError(
            f"neighbour rows ({indices.shape[0]}) must match movie_ids "
            f"({len(movie_ids)}) — these index the same axis"
        )

    np.savez(
        path,
        indices=indices,
        similarities=similarities,
        movie_ids=np.asarray(movie_ids),
    )

    return path


class ItemKNN:
    """Serving-time recommender over precomputed neighbours."""

    def __init__(self, indices, similarities, movie_ids):
        if indices.shape[0] != len(movie_ids):
            raise ValueError(
                "neighbour rows must match movie_ids; this artifact is "
                "inconsistent and would return neighbours from a different "
                "id space"
            )

        self.indices = indices
        self.similarities = similarities
        self.movie_ids = np.asarray(movie_ids)
        self.movie_to_idx = index_map(self.movie_ids)

    @classmethod
    def load(cls, path=None):
        path = path or neighbors_path()

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found — run scripts/build_item_neighbors.py"
            )

        with np.load(path) as data:
            return cls(data["indices"], data["similarities"], data["movie_ids"])

    def recommend(self, history, k=10, seen=None):
        """Rank unseen movies by summed similarity to the user's history.

        `seen` defaults to the history itself; pass the full training
        history to match the exclusion used by the other recommenders.
        """
        seen = set(history) if seen is None else set(seen)

        scores = {}

        for movie_id in history:
            movie_idx = self.movie_to_idx.get(movie_id)

            if movie_idx is None:
                continue

            for neighbor_idx, similarity in zip(
                self.indices[movie_idx],
                self.similarities[movie_idx],
            ):
                candidate = self.movie_ids[neighbor_idx]

                if candidate in seen:
                    continue

                scores[candidate] = scores.get(candidate, 0.0) + similarity

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)

        return [movie_id for movie_id, _ in ranked[:k]]
