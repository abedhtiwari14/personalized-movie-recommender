"""Item-item collaborative filtering over behavioural co-occurrence.

Builds a sparse user x movie matrix from positive training interactions,
transposes into movie space, and takes cosine nearest neighbours between
movies. Neighbours are precomputed offline so that serving a recommendation
is a lookup-and-aggregate rather than a similarity computation.

The neighbour artifact is stored as a single .npz carrying `movie_ids`
alongside `indices` and `similarities`. Neighbour rows are positions in
`movie_ids`, so the two cannot be separated — storing them apart is what
produced the stale-artifact bug this module was extracted to fix.
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
    """Sparse CSR matrix of positive interactions (users x movies).

    Dense would be ~8.7 billion cells; the matrix is ~0.13% occupied.
    """
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
    """Precompute the cosine nearest neighbours of every movie.

    Returns (indices, similarities), both shape (len(movie_ids), n_neighbors).
    Each row's first entry is the movie itself at similarity 1.0.
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

        `history` is the user's positive training movies. `seen` is what to
        exclude; it defaults to the history itself, which is the convention
        the published results were produced under. Passing the user's full
        training history (including negatively-rated items) makes the
        exclusion consistent with the popularity and BPR recommenders.
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
