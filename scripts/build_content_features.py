"""Build the TF-IDF content feature matrices.

Run from the project root:

    python scripts/build_content_features.py

Writes `data/processed/content_features_genres.npz` and
`content_features_genres_tags.npz`, each over the FULL catalog so that cold
items are representable. Every matrix carries its own movie_ids and
vocabulary.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import loaders  # noqa: E402
from src.models.content import build_features, save_features  # noqa: E402

VARIANTS = ("genres", "genres_tags")


def main():
    started = time.time()

    movies = loaders.load_movies()
    universe = loaders.full_universe(movies)
    print(f"Full catalog: {len(universe):,} movies")

    train_positive = loaders.positives(loaders.load_split("train"))
    warm = set(loaders.warm_universe(train_positive).tolist())
    cold = loaders.cold_items(train_positive, loaders.load_split("test"))

    print(f"  warm (positive train signal): {len(warm):,}")
    print(f"  cold (test target, no train signal): {len(cold):,}")
    print(f"  never a target, never trained on: "
          f"{len(universe) - len(warm) - len(cold):,}\n")

    tags = pd.read_csv(os.path.join(loaders.RAW_DIR, "tags.csv"))
    print(f"Tags: {len(tags):,} rows from {tags['userId'].nunique():,} users\n")

    for variant in VARIANTS:
        use_tags = tags if variant == "genres_tags" else None

        matrix, vocabulary = build_features(universe, movies, use_tags)

        covered = np.diff(matrix.indptr) > 0
        cold_rows = np.isin(universe, cold)

        print(f"{variant}:")
        print(f"  matrix: {matrix.shape[0]:,} x {matrix.shape[1]:,}  "
              f"nnz={matrix.nnz:,}  density={matrix.nnz / np.prod(matrix.shape):.5f}")
        print(f"  movies with at least one feature: {covered.mean() * 100:.1f}%")
        print(f"  COLD movies with at least one feature: "
              f"{covered[cold_rows].mean() * 100:.1f}%")

        path = save_features(matrix, universe, vocabulary, variant)
        print(f"  wrote {os.path.basename(path)}\n")

    print(f"Done in {(time.time() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
