"""Content-based recommendation from movie metadata.

Scores candidates by cosine similarity between a TF-IDF item representation
and a user profile built from that user's positive history. Unlike Item-KNN,
this does not use precomputed top-N neighbours: every candidate receives a
score, which is the property the cold-start evaluation depends on. A movie
with no interactions has no behavioural neighbours and can never be reached
by an item-neighbour lookup, but it does have genres.

Two feature variants, and the distinction is the point:

- `genres`     20 one-hot dimensions from movies.csv. Static metadata, known
               at item launch, present for 91.5% of cold items. This is the
               variant that genuinely addresses item cold-start.
- `genres+tags` adds 2M user-generated tags. Stronger on warm items, but
               tags are behaviour wearing content clothing: they exist only
               where users already engaged (100% coverage on the top 1,000
               movies by popularity, 70% on the tail, 39.7% on cold items).
               This variant is NOT a cold-start solution and is not presented
               as one.
"""

import os
import re

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..data.loaders import PROCESSED_DIR, index_map

# Tag strings appearing on fewer than this many movies are dropped. There are
# 126,067 distinct tag strings, most of them near-unique free text.
MIN_TAG_DF = 5

NO_GENRES = "(no genres listed)"

WHITESPACE = re.compile(r"\s+")

FEATURES_FILENAME = "content_features_{variant}.npz"


def features_path(variant):
    return os.path.join(PROCESSED_DIR, FEATURES_FILENAME.format(variant=variant))


def normalize_tag(tag):
    return WHITESPACE.sub(" ", str(tag).strip().lower())


def build_documents(movie_ids, movies, tags=None):
    """One token list per movie, in `movie_ids` order.

    Genre and tag tokens are prefixed so that the genre "Drama" and the
    user tag "drama" stay distinct features.
    """
    genre_lookup = dict(zip(movies["movieId"], movies["genres"]))

    tag_lookup = {}

    if tags is not None:
        grouped = (
            tags
            .assign(_tag=tags["tag"].map(normalize_tag))
            .groupby("movieId")["_tag"]
            .apply(list)
        )
        tag_lookup = grouped.to_dict()

    documents = []

    for movie_id in movie_ids:
        tokens = []

        genres = genre_lookup.get(movie_id)

        if isinstance(genres, str) and genres != NO_GENRES:
            tokens.extend(f"genre:{g}" for g in genres.split("|"))

        tokens.extend(f"tag:{t}" for t in tag_lookup.get(movie_id, ()) if t)

        documents.append(tokens)

    return documents


def build_features(movie_ids, movies, tags=None, min_tag_df=MIN_TAG_DF):
    """TF-IDF item matrix, L2-normalised so cosine is a dot product.

    `sublinear_tf` damps tag volume: a movie tagged "classic" 500 times
    should not outweigh one tagged it twice by 250x. Raw counts would
    reconstruct a popularity ranker out of content features — the same
    failure mode the BPR bias term produced.
    """
    documents = build_documents(movie_ids, movies, tags)

    vectorizer = TfidfVectorizer(
        analyzer=lambda tokens: tokens,
        min_df=min_tag_df if tags is not None else 1,
        sublinear_tf=True,
        norm="l2",
    )

    matrix = vectorizer.fit_transform(documents)

    return matrix.tocsr(), vectorizer.get_feature_names_out()


def save_features(matrix, movie_ids, vocabulary, variant):
    """Store the matrix with the ids and vocabulary it is indexed by.

    Same contract as the neighbour artifact: positions are meaningless
    without the vocabulary they refer to, so they travel together.
    """
    if matrix.shape[0] != len(movie_ids):
        raise ValueError(
            f"feature rows ({matrix.shape[0]}) must match movie_ids "
            f"({len(movie_ids)})"
        )

    path = features_path(variant)

    np.savez(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape),
        movie_ids=np.asarray(movie_ids),
        vocabulary=np.asarray(vocabulary, dtype=object),
    )

    return path


class ContentRecommender:

    def __init__(self, matrix, movie_ids):
        if matrix.shape[0] != len(movie_ids):
            raise ValueError("feature rows must match movie_ids")

        self.matrix = matrix
        self.movie_ids = np.asarray(movie_ids)
        self.movie_to_idx = index_map(self.movie_ids)

    @classmethod
    def load(cls, variant):
        path = features_path(variant)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found — run scripts/build_content_features.py"
            )

        with np.load(path, allow_pickle=True) as data:
            matrix = csr_matrix(
                (data["data"], data["indices"], data["indptr"]),
                shape=tuple(data["shape"]),
            )

            return cls(matrix, data["movie_ids"])

    def restrict_to(self, movie_ids):
        """A recommender over a subset of the catalog.

        Used to score the warm benchmark (43,417 candidates) with features
        built over the full catalog, so both evaluations share one artifact.
        """
        keep = [
            self.movie_to_idx[movie_id]
            for movie_id in movie_ids
            if movie_id in self.movie_to_idx
        ]

        return ContentRecommender(self.matrix[keep], self.movie_ids[keep])

    def profile(self, history):
        """User profile: L2-normalised mean of the history's item vectors.

        Mean rather than sum so the profile does not grow with history
        length. Item rows are already L2-normalised, so each watched movie
        contributes equally regardless of how many tags it carries.
        """
        rows = [
            self.movie_to_idx[movie_id]
            for movie_id in history
            if movie_id in self.movie_to_idx
        ]

        if not rows:
            return None

        vector = np.asarray(self.matrix[rows].mean(axis=0)).ravel()
        norm = np.linalg.norm(vector)

        return vector / norm if norm > 0 else None

    def score(self, history):
        """Cosine similarity of every candidate to the user profile."""
        vector = self.profile(history)

        if vector is None:
            return None

        return self.matrix @ vector

    def recommend(self, history, seen=None, k=10):
        scores = self.score(history)

        if scores is None:
            return []

        seen = set(history) if seen is None else set(seen)

        seen_rows = [
            self.movie_to_idx[movie_id]
            for movie_id in seen
            if movie_id in self.movie_to_idx
        ]
        scores = scores.copy()
        scores[seen_rows] = -np.inf

        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]

        return [self.movie_ids[idx] for idx in top if np.isfinite(scores[idx])]
