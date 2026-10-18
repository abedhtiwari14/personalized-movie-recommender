"""Assemble the Hugging Face Space folder.

Copies the model code and writes the artifacts the demo needs. Only app.py,
README.md and requirements.txt are tracked in this repo; everything else
here is generated.

    python scripts/build_space.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from src.data import loaders  # noqa: E402
from src.models.content import build_content_features  # noqa: E402

SPACE = os.path.join(loaders.PROJECT_ROOT, "space")

MODULES = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/loaders.py",
    "src/models/__init__.py",
    "src/models/item_knn.py",
    "src/models/content.py",
]

PICKER_SIZE = 2000


def main():
    for relative in MODULES:
        target = os.path.join(SPACE, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy(os.path.join(loaders.PROJECT_ROOT, relative), target)
    print(f"copied {len(MODULES)} modules")

    shutil.copy(
        os.path.join(loaders.PROCESSED_DIR, "item_neighbors.npz"),
        os.path.join(SPACE, "item_neighbors.npz"),
    )

    maps = loaders.load_id_maps()
    rankable = set(maps["movie_ids"].tolist())

    movies = loaders.load_movies()
    titles = movies[movies["movieId"].isin(rankable)][["movieId", "title", "genres"]]
    titles.to_csv(os.path.join(SPACE, "titles.csv"), index=False)

    train_positive = loaders.positives(loaders.load_split("train"))
    popular = (
        train_positive.groupby("movieId").size()
        .sort_values(ascending=False).head(PICKER_SIZE).index
    )
    titles[titles["movieId"].isin(set(popular))].to_csv(
        os.path.join(SPACE, "picker.csv"), index=False)

    features, movie_ids, _ = build_content_features(use_tags=True)
    X = normalize(features).astype(np.float32)
    keep = np.array([i for i, m in enumerate(movie_ids) if m in rankable])
    X = csr_matrix(X[keep])

    np.savez_compressed(
        os.path.join(SPACE, "content_features.npz"),
        data=X.data, indices=X.indices, indptr=X.indptr,
        shape=np.asarray(X.shape), movie_ids=np.asarray(movie_ids)[keep],
    )

    print(f"titles {len(titles):,} | picker {PICKER_SIZE:,} | content {X.shape}")

    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, names in os.walk(SPACE) for f in names
    )
    print(f"space/ total: {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
