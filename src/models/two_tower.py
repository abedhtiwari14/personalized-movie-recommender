"""Two-tower retrieval model.

Score is the dot product of a user embedding and an item embedding. The item
tower is a function of content features only, with no per-item learned id, so
items absent from training still receive an embedding.
"""

import numpy as np
import torch
import torch.nn as nn


def content_bag(features, rows):
    """Convert rows of a CSR feature matrix to EmbeddingBag inputs.

    CSR stores the same three arrays EmbeddingBag expects, so this is a
    rename: indices -> ids, indptr[:-1] -> offsets, data -> weights.
    Rows with no features give an empty bag, which pools to a zero vector.

    Returns (ids, offsets, weights) as tensors.
    """
    block = features[rows]

    return (
        torch.from_numpy(block.indices.astype(np.int64)),
        torch.from_numpy(block.indptr[:-1].astype(np.int64)),
        torch.from_numpy(block.data.astype(np.float32)),
    )


class ItemTower(nn.Module):
    """Content features -> embedding."""

    def __init__(self, n_features, bag_dim=128, out_dim=64, init_std=0.1):
        super().__init__()

        # "sum" not "mean": rows are L2-normalised, so the weighted sum has
        # near-constant scale regardless of feature count.
        self.bag = nn.EmbeddingBag(n_features, bag_dim, mode="sum")

        self.mlp = nn.Sequential(
            nn.Linear(bag_dim, bag_dim),
            nn.ReLU(),
            # No output bias: it is identical across items, so it cancels in
            # the pairwise score difference and its gradient is always zero.
            nn.Linear(bag_dim, out_dim, bias=False),
        )

        nn.init.normal_(self.bag.weight, std=init_std)

    def forward(self, input_ids, offsets, weights):
        return self.mlp(self.bag(input_ids, offsets, per_sample_weights=weights))


class UserTower(nn.Module):
    """User id -> embedding. No cold-user support."""

    def __init__(self, n_users, embed_dim=64, out_dim=64, init_std=0.1):
        super().__init__()

        self.embedding = nn.Embedding(n_users, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim),
        )

        nn.init.normal_(self.embedding.weight, std=init_std)

    def forward(self, user_ids):
        return self.mlp(self.embedding(user_ids))


class TwoTower(nn.Module):

    def __init__(self, n_users, n_features, bag_dim=128, embed_dim=64,
                 out_dim=64, init_std=0.1):
        super().__init__()

        self.user_tower = UserTower(n_users, embed_dim, out_dim, init_std)
        self.item_tower = ItemTower(n_features, bag_dim, out_dim, init_std)

    def score(self, user_ids, item_bag):
        """Score (user, item) pairs. item_bag is (ids, offsets, weights)."""
        return (self.user_tower(user_ids) * self.item_tower(*item_bag)).sum(dim=1)

    def forward(self, user_ids, positive_bag, negative_bag):
        """Return (positive_scores, negative_scores) for a batch of triples."""
        users = self.user_tower(user_ids)

        positive = (users * self.item_tower(*positive_bag)).sum(dim=1)
        negative = (users * self.item_tower(*negative_bag)).sum(dim=1)

        return positive, negative


def tower_health(model):
    """Mean embedding norms, logged during training to detect collapse."""
    return {
        "user_embed": model.user_tower.embedding.weight.norm(dim=1).mean().item(),
        "item_feature": model.item_tower.bag.weight.norm(dim=1).mean().item(),
    }
