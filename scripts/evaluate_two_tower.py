"""Warm and cold evaluation of the two-tower model.

Item embeddings are independent of the user, so the item tower runs once
over the catalog and each user is a single matmul against the cache. Warm
and cold differ only in which rows of that cache are scored.

    python scripts/evaluate_two_tower.py [--negatives uniform|popularity]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from src.data import loaders  # noqa: E402
from src.models.content import build_content_features  # noqa: E402
from src.models.two_tower import TwoTower, content_bag  # noqa: E402
from src.utils.evaluation import EvaluationContext, evaluate_cold, evaluate_warm  # noqa: E402

K = 10
USER_BATCH = 256
ITEM_BATCH = 8192

def checkpoint_path(mode):
    name = "two_tower.pt" if mode == "uniform" else f"two_tower_{mode}.pt"
    return os.path.join(loaders.MODELS_DIR, name)


def embed_all_items(model, X, device):
    """Run the item tower once over the whole catalog."""
    vectors = []

    with torch.no_grad():
        for start in range(0, X.shape[0], ITEM_BATCH):
            rows = list(range(start, min(start + ITEM_BATCH, X.shape[0])))
            ids, offs, wts = content_bag(X, rows)
            vectors.append(
                model.item_tower(ids.to(device), offs.to(device), wts.to(device)).cpu())

    return torch.cat(vectors)


def recommend_all(model, item_vecs, candidate_rows, candidate_ids,
                  users, context, user_to_idx, row_of, device, label):
    """Top-k for every user, scored against a fixed slice of the item cache."""
    subset = item_vecs[candidate_rows].to(device)
    position = {m: i for i, m in enumerate(candidate_ids)}

    recs = {}
    started = time.time()

    with torch.no_grad():
        for start in range(0, len(users), USER_BATCH):
            chunk = [u for u in users[start:start + USER_BATCH] if u in user_to_idx]

            if not chunk:
                continue

            idxs = torch.tensor([user_to_idx[u] for u in chunk], device=device)
            scores = model.user_tower(idxs) @ subset.T

            for row, user_id in enumerate(chunk):
                s = scores[row].clone()

                blocked = [position[m] for m in context.seen_for(user_id)
                           if m in position]
                s[blocked] = float("-inf")

                top = torch.topk(s, K).indices.cpu().numpy()
                recs[user_id] = [candidate_ids[i] for i in top]

            if start and start % (USER_BATCH * 100) == 0:
                print(f"    {label}: {start:,}/{len(users):,} "
                      f"({(time.time() - started) / 60:.1f} min)", flush=True)

    return recs


def main(negatives="uniform"):
    started = time.time()
    print(f"negatives: {negatives}", flush=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}", flush=True)

    features, movie_ids, _ = build_content_features(use_tags=True)
    X = normalize(features)
    row_of = {m: i for i, m in enumerate(movie_ids)}

    maps = loaders.load_id_maps()
    user_ids = maps["bpr_user_ids"]
    user_to_idx = {u: i for i, u in enumerate(user_ids)}

    checkpoint = torch.load(checkpoint_path(negatives), map_location="cpu", weights_only=False)
    model = TwoTower(n_users=len(user_ids), n_features=X.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"loaded epoch {checkpoint['epoch']}, val R@{K} {checkpoint['val_recall']:.4f}",
          flush=True)

    print("embedding the full catalog once...", flush=True)
    item_vecs = embed_all_items(model, X, device)
    print(f"  item cache: {tuple(item_vecs.shape)}", flush=True)

    context = EvaluationContext.load()
    print(f"  warm users {len(context.users):,}  cold users {len(context.cold_users):,}",
          flush=True)

    # ---------------- WARM ----------------
    warm_ids = list(maps["movie_ids"])
    warm_rows = [row_of[m] for m in warm_ids]

    print(f"\n===== WARM ({len(warm_ids):,} candidates) =====", flush=True)
    recs = recommend_all(model, item_vecs, warm_rows, warm_ids,
                         context.users, context, user_to_idx, row_of, device, "warm")
    warm = evaluate_warm(lambda u: recs.get(u, []), context, k=K,
                         label=f"Two-tower ({negatives} neg)", verbose=False)
    print(f"  Two-tower: R@{K}={warm[f'recall@{K}']:.4f}  "
          f"P@{K}={warm[f'precision@{K}']:.4f}  NDCG@{K}={warm[f'ndcg@{K}']:.4f}",
          flush=True)

    # ---------------- COLD ----------------
    all_ids = list(movie_ids)
    all_rows = list(range(len(movie_ids)))

    print(f"\n===== COLD ({len(all_ids):,} candidates) =====", flush=True)
    cold_recs = recommend_all(model, item_vecs, all_rows, all_ids,
                              context.cold_users, context, user_to_idx, row_of,
                              device, "cold")
    evaluate_cold(lambda u: cold_recs.get(u, []), context, k=K, label=f"Two-tower ({negatives} neg)")

    pd.DataFrame([{
        "model": f"Two-tower ({negatives} neg)",
        f"recall@{K}": warm[f"recall@{K}"],
        f"precision@{K}": warm[f"precision@{K}"],
        f"ndcg@{K}": warm[f"ndcg@{K}"],
    }]).to_csv(os.path.join(loaders.PROCESSED_DIR, f"leaderboard_two_tower_{negatives}.csv"),
               index=False)

    print(f"\nTotal: {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negatives", choices=["uniform", "popularity"], default="uniform")
    main(parser.parse_args().negatives)