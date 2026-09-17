import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import loaders


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

    # Genres: "Action|Sci-Fi|Thriller" -> "Action Sci-Fi Thriller"
    genre_text = (
        movies["genres"]
        .replace("(no genres listed)", "")
        .str.replace("|", " ", regex=False)
    )
    genre_by_id = pd.Series(genre_text.values, index=movies["movieId"])
    aligned_genres = genre_by_id.reindex(movie_ids).fillna("")

    if use_tags:
        tags = pd.read_csv("../data/raw/ml-32m/tags.csv")
        tags = tags.dropna(subset=["tag"])
        tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
        tag_by_id = tags.groupby("movieId")["tag"].apply(lambda s: " ".join(s))
        aligned_tags = tag_by_id.reindex(movie_ids).fillna("")
        combined = aligned_genres + " " + aligned_tags
    else:
        combined = aligned_genres

    vectorizer = TfidfVectorizer(min_df=min_df)
    features = vectorizer.fit_transform(combined)

    return features, movie_ids