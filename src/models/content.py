"""Content-based recommendation from movie genres and user tags."""

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import loaders


def _identity_analyzer(tokens):
    """Pass pre-tokenized lists through TfidfVectorizer unchanged."""
    return tokens


def user_profile(X, idx, history):
    """Mean of the user's liked-movie vectors, L2-normalised.

    X       : L2-normalised TF-IDF matrix, one row per movie
    idx     : {movieId: row number}
    history : movieIds the user liked in training

    Returns a unit vector, or None if no history row is present in X.
    """
    rows = [idx[m] for m in history if m in idx]

    if not rows:
        return None

    v = np.asarray(X[rows].mean(axis=0)).ravel()
    norm = np.linalg.norm(v)

    return v / norm if norm > 0 else None


def recommend_content(X, idx, movie_ids, history, seen, k=10, candidate_mask=None):
    """Rank movies by cosine similarity to the user's profile.

    Every candidate is scored, not only the neighbours of items the user
    already liked, so items with no interaction history remain rankable.

    seen           : movieIds to exclude; pass the full training history,
                     not only the positively-rated part
    candidate_mask : boolean array over movie_ids restricting the candidate
                     set, or None for the full catalog
    """
    profile = user_profile(X, idx, history)

    if profile is None:
        return []

    scores = X @ profile

    if candidate_mask is not None:
        scores = np.where(candidate_mask, scores, -np.inf)

    seen_rows = [idx[m] for m in seen if m in idx]
    scores[seen_rows] = -np.inf

    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]

    return [movie_ids[i] for i in top if np.isfinite(scores[i])]


def build_content_features(min_df=5, use_tags=True):
    """TF-IDF matrix over the full catalog.

    Returns (features, movie_ids, vocabulary). Rows follow full_universe()
    order; tags are kept as whole tokens rather than split into words.
    """
    movies = loaders.load_movies()
    movie_ids = loaders.full_universe(movies)

    genre_lookup = dict(zip(movies["movieId"], movies["genres"]))

    tag_lookup = {}

    if use_tags:
        tags = pd.read_csv(os.path.join(loaders.RAW_DIR, "tags.csv"))
        tags = tags.dropna(subset=["tag"])
        tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
        tags = tags[tags["tag"] != ""]
        tag_lookup = tags.groupby("movieId")["tag"].apply(list).to_dict()

    # Prefixes keep the genre "Drama" distinct from the user tag "drama".
    documents = []

    for movie_id in movie_ids:
        tokens = []

        genres = genre_lookup.get(movie_id)

        if isinstance(genres, str) and genres != "(no genres listed)":
            tokens.extend(f"genre:{g}" for g in genres.split("|"))

        tokens.extend(f"tag:{t}" for t in tag_lookup.get(movie_id, ()))

        documents.append(tokens)

    # sublinear_tf damps tag volume, which otherwise tracks popularity.
    vectorizer = TfidfVectorizer(
        analyzer=_identity_analyzer,
        min_df=min_df,
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(documents)

    return features, movie_ids, vectorizer.get_feature_names_out()
