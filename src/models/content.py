import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import loaders

def _identity_analyzer(tokens):
    """Pass pre-tokenized lists through TfidfVectorizer unchanged.

    A module-level function rather than a lambda so the vectorizer stays
    picklable.
    """
    return tokens


def user_profile(X, idx, history):
    """Average a user's liked-movie vectors into one taste vector.

    X       : L2-normalized TF-IDF matrix, one row per movie
    idx     : {movieId: row number}
    history : movieIds the user liked in training

    Returns a unit-length vector, or None if none of the history is in X.
    """
    rows = [idx[m] for m in history if m in idx]
    if not rows:
        return None

    v = np.asarray(X[rows].mean(axis=0)).ravel()
    norm = np.linalg.norm(v)

    return v / norm if norm > 0 else None

def recommend_content(X, idx, movie_ids, history, seen, k=10, candidate_mask=None):
    """Rank movies by cosine similarity to the user's taste profile.

    Unlike item-KNN, every candidate gets a score rather than only the
    neighbours of movies the user already liked. That is the property the
    cold-start evaluation needs: a movie nobody has watched has no
    neighbours, but it does have genres.

    seen           : movieIds to exclude. Pass the user's FULL training
                     history, not just the liked part, or the model wastes
                     top-k slots on films they already rated and disliked.
    candidate_mask : optional boolean array over movie_ids. Use it to
                     restrict to the 43,417-movie warm universe so results
                     are comparable with the other three models; leave it
                     None to rank the full catalog for cold-start.
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
    """
    Build a TF-IDF content matrix over the full 87,585-movie catalog.

    Returns
    -------
    features : scipy.sparse.csr_matrix, shape (87585, V)
    movie_ids : np.ndarray, the row ordering (sorted, from full_universe())
    """
    movies = loaders.load_movies()
    movie_ids = loaders.full_universe(movies)

    # Genres stay whole: "Action|Sci-Fi|Thriller" -> three tokens, and
    # "Sci-Fi" stays one feature instead of splitting into "sci" and "fi".
    genre_lookup = dict(zip(movies["movieId"], movies["genres"]))

    tag_lookup = {}
    if use_tags:
        tags = pd.read_csv(os.path.join(loaders.RAW_DIR, "tags.csv"))
        tags = tags.dropna(subset=["tag"])
        tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
        tags = tags[tags["tag"] != ""]
        tag_lookup = tags.groupby("movieId")["tag"].apply(list).to_dict()

    # One token list per movie, in movie_ids order. Prefixes keep the genre
    # "Drama" distinct from the user tag "drama".
    documents = []
    for movie_id in movie_ids:
        tokens = []

        genres = genre_lookup.get(movie_id)
        if isinstance(genres, str) and genres != "(no genres listed)":
            tokens.extend(f"genre:{g}" for g in genres.split("|"))

        tokens.extend(f"tag:{t}" for t in tag_lookup.get(movie_id, ()))

        documents.append(tokens)

    # sublinear_tf damps tag volume: a movie tagged "classic" 500 times
    # should not outweigh one tagged it twice by 250x, or raw counts just
    # rebuild a popularity ranker out of content features.
    vectorizer = TfidfVectorizer(
        analyzer=_identity_analyzer,
        min_df=min_df,
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(documents)

    return features, movie_ids, vectorizer.get_feature_names_out()