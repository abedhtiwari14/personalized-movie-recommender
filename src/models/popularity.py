"""Non-personalized popularity baseline.

Ranks movies by the count of positive training interactions, not by average
rating: a 5.0 average from 2 users should not outrank a 4.4 from 50,000.
"""

import numpy as np


class PopularityRecommender:

    def __init__(self, movie_ids_by_popularity):
        self.ranking = np.asarray(movie_ids_by_popularity)

    @classmethod
    def fit(cls, train_positive):
        counts = (
            train_positive
            .groupby("movieId")
            .size()
            .sort_values(ascending=False)
        )

        return cls(counts.index.to_numpy())

    def popularity_rank(self):
        """movieId -> 1-based position in the popularity ranking."""
        return {mid: rank for rank, mid in enumerate(self.ranking, start=1)}

    def recommend(self, seen, k=10):
        """Top-k most popular movies the user has not already interacted with."""
        seen = set(seen)
        out = []

        for movie_id in self.ranking:
            if movie_id in seen:
                continue

            out.append(movie_id)

            if len(out) == k:
                break

        return out
