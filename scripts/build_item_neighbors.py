"""Rebuild the Item-KNN neighbour artifact and the id mappings.

Writes data/processed/item_neighbors.npz and id_maps.npz.

    python scripts/build_item_neighbors.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from src.data import loaders  # noqa: E402
from src.models.item_knn import (  # noqa: E402
    N_NEIGHBORS,
    build_neighbors,
    neighbors_path,
    save_neighbors,
)

STALE_FILES = (
    "item_neighbor_indices.npy",
    "item_neighbor_similarities.npy",
)


def main():
    started = time.time()

    print("Loading training split...")
    train_positive = loaders.positives(loaders.load_split("train"))
    print(f"  positive training interactions: {len(train_positive):,}")

    maps = loaders.build_id_maps(train_positive)
    print(f"  users:  {len(maps['knn_user_ids']):,}")
    print(f"  movies: {len(maps['movie_ids']):,}")

    loaders.save_id_maps(maps)
    print(f"  wrote {loaders.id_maps_path()}")

    print(f"\nComputing {N_NEIGHBORS} nearest neighbours per movie...")
    indices, similarities = build_neighbors(
        train_positive,
        maps["knn_user_ids"],
        maps["movie_ids"],
    )
    print(f"  shape: {indices.shape}  ({(time.time() - started) / 60:.1f} min)")

    self_is_first = (indices[:, 0] == np.arange(indices.shape[0])).mean()
    print(f"  rows whose nearest neighbour is themselves: {self_is_first:.4f}")
    print(f"  self-similarity mean: {similarities[:, 0].mean():.6f}")

    path = save_neighbors(indices, similarities, maps["movie_ids"])
    print(f"  wrote {path}")

    for name in STALE_FILES:
        stale = os.path.join(loaders.PROCESSED_DIR, name)

        if os.path.exists(stale):
            shape = np.load(stale, mmap_mode="r").shape
            os.remove(stale)
            print(f"  removed stale {name} {shape} — built against a "
                  f"different id space")

    print(f"\nDone in {(time.time() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
