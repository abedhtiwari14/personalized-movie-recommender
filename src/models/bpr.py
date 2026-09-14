"""BPR matrix factorization.

Score is `u . i + b_i`: 64-dim user and movie embeddings plus a per-item
bias, trained on `(user, positive item, sampled unobserved item)` triples
with the pairwise loss `-log sigmoid(score_pos - score_neg)`. Implemented in
PyTorch, but it is not a neural network — the scoring function has no hidden
layers.

Initialization matters here and is not a detail: at `std=0.01` the embedding
dot product and the gradients into it both start near zero while the bias
receives a full-sized gradient, so training collapses into a popularity
ranker with embedding norms of 0.0000. See the README.
"""

import os

import numpy as np
import torch
import torch.nn as nn

from ..data.loaders import MODELS_DIR, index_map

EMBEDDING_DIM = 64

CHECKPOINT_FILENAME = "bpr_model.pt"


def checkpoint_path():
    return os.path.join(MODELS_DIR, CHECKPOINT_FILENAME)


class BPRMatrixFactorization(nn.Module):

    def __init__(self, n_users, n_movies, embedding_dim=EMBEDDING_DIM, init_std=0.1):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.movie_bias = nn.Embedding(n_movies, 1)

        nn.init.normal_(self.user_embedding.weight, std=init_std)
        nn.init.normal_(self.movie_embedding.weight, std=init_std)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, users, movies):
        user_vectors = self.user_embedding(users)
        movie_vectors = self.movie_embedding(movies)

        scores = (user_vectors * movie_vectors).sum(dim=1)
        scores = scores + self.movie_bias(movies).squeeze()

        return scores


def bpr_loss(positive_scores, negative_scores):
    """Pairwise ranking loss: push positives above sampled negatives."""
    return -torch.log(torch.sigmoid(positive_scores - negative_scores) + 1e-10).mean()


def embedding_health(model):
    """Norms that distinguish a trained model from a collapsed one.

    A collapsed run shows embedding norms at 0.0000 against a bias std near
    0.54 — the check that caught the silent failure.
    """
    return {
        "user_norm": model.user_embedding.weight.norm(dim=1).mean().item(),
        "movie_norm": model.movie_embedding.weight.norm(dim=1).mean().item(),
        "bias_std": model.movie_bias.weight.std().item(),
    }


class BPRRecommender:
    """Serving-time recommender over a trained checkpoint."""

    def __init__(self, user_embedding, movie_embedding, movie_bias,
                 user_ids, movie_ids):
        self.user_embedding = np.asarray(user_embedding, dtype=np.float32)
        self.movie_embedding = np.asarray(movie_embedding, dtype=np.float32)
        self.movie_bias = np.asarray(movie_bias, dtype=np.float32).reshape(-1)

        self.user_ids = np.asarray(user_ids)
        self.movie_ids = np.asarray(movie_ids)

        if self.user_embedding.shape[0] != len(self.user_ids):
            raise ValueError("user embedding rows must match user_ids")

        if self.movie_embedding.shape[0] != len(self.movie_ids):
            raise ValueError("movie embedding rows must match movie_ids")

        self.user_to_idx = index_map(self.user_ids)
        self.movie_to_idx = index_map(self.movie_ids)

    @classmethod
    def load(cls, user_ids, movie_ids, path=None):
        path = path or checkpoint_path()

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]

        return cls(
            state["user_embedding.weight"].numpy(),
            state["movie_embedding.weight"].numpy(),
            state["movie_bias.weight"].numpy(),
            user_ids,
            movie_ids,
        )

    def scores_for(self, user_id):
        user_idx = self.user_to_idx.get(user_id)

        if user_idx is None:
            return None

        return self.movie_embedding @ self.user_embedding[user_idx] + self.movie_bias

    def recommend(self, user_id, seen=(), k=10):
        scores = self.scores_for(user_id)

        if scores is None:
            return []

        scores = scores.copy()

        seen_indices = [
            self.movie_to_idx[movie_id]
            for movie_id in seen
            if movie_id in self.movie_to_idx
        ]
        scores[seen_indices] = -np.inf

        # argpartition is safe here: exact ties between real scores are
        # measure-zero, and masked entries never reach the top k.
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]

        return [self.movie_ids[idx] for idx in top]
