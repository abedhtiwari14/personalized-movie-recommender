"""Train the two-tower retrieval model.

Reuses BPR's pairwise loss and triple structure; each batch converts its
positive and negative item rows into EmbeddingBag inputs.

    python scripts/train_two_tower.py [--negatives uniform|popularity]

weight_decay is 0: Adam would decay every feature embedding each step while
only the ones present in the batch receive a gradient. Early stopping is on
validation Recall@10 rather than training loss, over a random user sample.

Negatives are drawn from the warm universe. Cold items have no training
positives, so sampling them as negatives would only ever penalise them.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from src.data import loaders  # noqa: E402
from src.models.bpr import bpr_loss  # noqa: E402
from src.models.content import build_content_features  # noqa: E402
from src.models.two_tower import TwoTower, content_bag, tower_health  # noqa: E402
from src.utils.metrics import recall_at_k  # noqa: E402

K = 10
BATCH = 4096
EPOCHS = 12
PATIENCE = 3          # validation is noisy; 2 risked stopping on a dip
LR = 0.005
VAL_USERS = 5000
SEED = 42

# Popularity exponent for negative sampling (word2vec convention).
NEG_ALPHA = 0.75


def checkpoint_path(mode):
    name = "two_tower.pt" if mode == "uniform" else f"two_tower_{mode}.pt"
    return os.path.join(loaders.MODELS_DIR, name)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def negative_distribution(mode, warm_ids, train_positive):
    """Sampling weights over the warm universe, or None for uniform.

    Uniform sampling makes a popular movie appear constantly as a positive
    and rarely as a negative, so the model correctly infers "content like
    this is good" and learns popularity through the content features. Drawing
    negatives in proportion to popularity puts popular items on both sides of
    the comparison, which largely cancels that signal.
    """
    if mode == "uniform":
        return None

    counts = train_positive.groupby("movieId").size()
    weights = np.array([counts.get(m, 0) for m in warm_ids], dtype=np.float64)
    weights = weights ** NEG_ALPHA

    return weights / weights.sum()


def sample_negatives(n, warm_rows, rng, probs=None):
    """Negatives from the warm universe, as row indices into X."""
    if probs is None:
        return warm_rows[rng.integers(0, len(warm_rows), size=n)]

    return warm_rows[rng.choice(len(warm_rows), size=n, p=probs)]


def validate(model, X, device, users, user_to_idx, relevant, seen, warm_rows, warm_ids):
    """Recall@10 over the warm universe for a sample of validation users."""
    model.eval()

    # movieId -> column position, so masking a user's seen items costs
    # O(len(seen)) instead of a scan over all 43,417 warm ids per user.
    warm_index = {m: i for i, m in enumerate(warm_ids)}

    with torch.no_grad():
        # Embed every warm item once, then score each user against all of them.
        item_vecs = []
        for start in range(0, len(warm_rows), 8192):
            block = warm_rows[start:start + 8192]
            ids, offs, wts = content_bag(X, list(block))
            item_vecs.append(
                model.item_tower(ids.to(device), offs.to(device), wts.to(device)))
        item_vecs = torch.cat(item_vecs)

        recalls = []
        for start in range(0, len(users), 1024):
            chunk = users[start:start + 1024]
            idxs = torch.tensor([user_to_idx[u] for u in chunk], device=device)
            scores = model.user_tower(idxs) @ item_vecs.T      # (users, items)

            for row, user_id in enumerate(chunk):
                s = scores[row].clone()
                blocked = [warm_index[m] for m in seen.get(user_id, ())
                           if m in warm_index]
                s[blocked] = float("-inf")
                top = torch.topk(s, K).indices.cpu().numpy()
                recalls.append(recall_at_k([warm_ids[i] for i in top],
                                           relevant[user_id], K))

    model.train()
    return float(np.mean(recalls))


def main(negatives="uniform"):
    started = time.time()
    print(f"negative sampling: {negatives}", flush=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"device: {device}", flush=True)

    print("building content features...", flush=True)
    features, movie_ids, _ = build_content_features(use_tags=True)
    X = normalize(features)
    row_of = {m: i for i, m in enumerate(movie_ids)}
    print(f"  {X.shape[0]:,} movies x {X.shape[1]:,} features", flush=True)

    print("loading splits...", flush=True)
    train = loaders.load_split("train")
    train_positive = loaders.positives(train)
    maps = loaders.load_id_maps()

    user_ids = maps["bpr_user_ids"]
    user_to_idx = {u: i for i, u in enumerate(user_ids)}

    warm_ids = list(maps["movie_ids"])
    warm_rows = np.array([row_of[m] for m in warm_ids])

    # Triples are (user index, positive item row); negatives resampled per epoch.
    pos_users = train_positive["userId"].map(user_to_idx).to_numpy()
    pos_rows = train_positive["movieId"].map(row_of).to_numpy()
    print(f"  {len(pos_users):,} positive interactions", flush=True)

    validation = loaders.load_split("validation")
    val_relevant = loaders.relevant_by_user(validation)
    seen = train.groupby("userId")["movieId"].apply(set)
    eligible = [u for u in val_relevant.index if u in user_to_idx]
    val_users = list(rng.choice(eligible, size=min(VAL_USERS, len(eligible)),
                                replace=False))
    print(f"  validating on {len(val_users):,} random users", flush=True)

    model = TwoTower(n_users=len(user_ids), n_features=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters\n", flush=True)

    neg_probs = negative_distribution(negatives, warm_ids, train_positive)
    checkpoint = checkpoint_path(negatives)

    best_recall, best_epoch = -1.0, -1

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        order = rng.permutation(len(pos_users))
        neg_rows = sample_negatives(len(pos_users), warm_rows, rng, neg_probs)
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]

            users = torch.from_numpy(pos_users[batch].astype(np.int64)).to(device)
            pos = [t.to(device) for t in content_bag(X, pos_rows[batch])]
            neg = [t.to(device) for t in content_bag(X, neg_rows[batch])]

            positive, negative = model(users, pos, neg)
            loss = bpr_loss(positive, negative)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        health = tower_health(model)
        recall = validate(model, X, device, val_users, user_to_idx,
                          val_relevant, seen, warm_rows, warm_ids)

        print(f"epoch {epoch} | loss {total_loss / n_batches:.4f} | "
              f"val R@{K} {recall:.4f} | user_embed {health['user_embed']:.4f} | "
              f"item_feature {health['item_feature']:.4f} | "
              f"{(time.time() - t0) / 60:.1f}m", flush=True)

        if recall > best_recall:
            best_recall, best_epoch = recall, epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_recall": recall,
                        "negatives": negatives}, checkpoint)
            print(f"  new best, checkpoint saved", flush=True)
        elif epoch - best_epoch >= PATIENCE:
            print(f"  early stopping (best epoch {best_epoch})", flush=True)
            break

    print(f"\nbest epoch {best_epoch}, val R@{K} {best_recall:.4f}", flush=True)
    print(f"total {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negatives", choices=["uniform", "popularity"],
                        default="uniform",
                        help="how to draw negatives (default: uniform)")
    main(parser.parse_args().negatives)