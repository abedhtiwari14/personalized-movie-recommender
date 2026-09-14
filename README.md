# Personalized Movie Recommendation System

An end-to-end Top-K recommendation system built on MovieLens 32M, benchmarking classical and learned approaches under a leakage-audited temporal evaluation protocol.

**Task:** given a user's historical interactions, rank previously-unseen movies so that the ones they will positively interact with next appear near the top. This is a ranking problem, not rating prediction.

---

## Results

Full test-set evaluation on **193,622 users**, ranking within a candidate universe of **43,417 movies**.

| Model | Recall@10 | Precision@10 | NDCG@10 |
|---|---|---|---|
| Popularity (baseline) | 0.0480 | 0.0382 | 0.0530 |
| BPR (Matrix Factorization) | 0.0543 | 0.0421 | 0.0584 |
| **Item-KNN (item-item CF)** | **0.0644** | **0.0456** | **0.0644** |

- BPR beats the non-personalized baseline by **+13.1%** relative on Recall@10.
- Item-KNN beats BPR by **+18.6%**. Wilcoxon signed-rank test on 193,622 paired per-user recalls: **p < 1e-6**.

The simplest personalized model wins. Added model complexity did not pay off on this benchmark.

---

## Dataset

[MovieLens 32M](https://grouplens.org/datasets/movielens/) — 32,000,204 ratings from 200,948 users across 87,585 movies, spanning 1995–2023.

| File | Signal | Used by |
|---|---|---|
| `ratings.csv` | Behavioral (userId, movieId, rating, timestamp) | All current models |
| `movies.csv` | Content (title, genres) | Planned — content-based model |
| `tags.csv` | Semantic (2M user-generated tags, 140K unique) | Planned — content-based model |

**Positive interaction** is defined as `rating >= 4`, yielding 15,938,231 positives (49.8% of all ratings). Ratings below 4 are *not* treated as explicit negatives — they are simply not positive signals. This matters for negative sampling.

---

## Evaluation protocol

**Per-user chronological 70/15/15 split.** Each user's interactions are sorted by timestamp; the earliest 70% go to train, the next 15% to validation, the final 15% to test. Verified: zero train→validation and validation→test ordering violations.

**Why not a global time-based split?** That was the first attempt. An audit found **86.6% of test users and 77.3% of validation users had no training history at all**. MovieLens users tend to rate in bursts and then stop, so cutting globally by calendar date sorts users into eras rather than splitting their histories — producing an accidental cold-start benchmark instead of a personalization one.

*Known tradeoff:* per-user splitting means one user's training interaction can be chronologically later than another user's test interaction. This is standard in recsys benchmarking and appropriate for evaluating personalization, but it is less production-realistic than a global cut. The global split is retained as a potential future cold-start benchmark.

**Metrics:** Recall@10, Precision@10, NDCG@10. Accuracy is unsuitable — with ~3 relevant items among 43,417 candidates, predicting "not relevant" for everything scores 99.99%.

Metric implementations live in [`src/utils/metrics.py`](src/utils/metrics.py) and are unit-tested in [`tests/test_metrics.py`](tests/test_metrics.py) against a hand-computed case (hits at ranks 1, 3, 7 with 4 relevant items → Recall@10 = 0.75, NDCG@10 ≈ 0.7157), plus truncation, tie, and empty-set cases. Run them with `pytest tests/`.

One test encodes the denominator ceiling documented below as an executable fact: a perfect top-10 for a user with 50 relevant items scores 0.20, while a perfect top-10 for a user with 2 scores 1.00.

---

## Models

**Popularity** — ranks movies by count of positive interactions in training data (not average rating: a 5.0 average from 2 users should not outrank a 4.4 from 50,000). Non-personalized; serves as the minimum bar and a cold-start fallback.

**Item-KNN** — builds a sparse 200,501 × 43,417 user-item matrix (CSR; a dense version would be ~8.7 billion cells), transposes to movie-space, and computes cosine similarity between movies based on shared user behavior. Item neighbors are precomputed offline so online recommendation is a lookup-and-aggregate rather than a similarity computation.

Note this model has no access to genres or plots — Shawshank's nearest neighbors (Pulp Fiction 0.554, Silence of the Lambs 0.547, Forrest Gump 0.536, Schindler's List 0.535) emerge purely from behavioral co-occurrence.

**BPR** — matrix factorization with 64-dim user and movie embeddings plus a per-item bias, trained with a pairwise ranking loss: `-log σ(score_pos − score_neg)`. Training triples are `(user, positive item, sampled unobserved item)`. Not a neural network — implemented in PyTorch, but the scoring function is a dot product with no hidden layers.

---

## Two debugging findings

### 1. Silent model collapse in BPR

The first BPR run scored **0.0475** Recall@10 — marginally *below* the popularity baseline. Training loss had fallen 0.693 → 0.154 throughout, appearing healthy.

Diagnosis: the score is `u·i + b_i`. With embeddings initialized at `std=0.01`, their dot product and the gradients flowing into it start near zero, while `movie_bias` is a free parameter with a full-sized gradient. Gradient descent took the cheap path and learned popularity via the bias alone. Measured post-training:

```
mean ‖user embedding‖:  0.0000
mean ‖movie embedding‖: 0.0000
movie bias std:         0.5370
```

The bias outweighed the personalized term by ~4 orders of magnitude — the model had degenerated into a popularity ranker with noise on top, which is why it scored slightly *worse* than clean popularity.

Fix: larger initialization (`std=0.1`), higher learning rate (0.005), no weight decay, and early stopping on validation Recall@10. Result: 0.0543.

*A failed intermediate fix worth recording:* adding `weight_decay=1e-4` re-collapsed the embeddings within one epoch. Adam decays every embedding row on every step, but a given row only receives a gradient when its item appears in a batch — so rare items get decayed constantly and updated rarely. Early stopping, not regularization, was the correct tool.

**Takeaway: training loss measured the training objective, not recommendation quality.**

### 2. A segment finding that was a metric artifact

Recall@10 bucketed by user history length appeared to show Item-KNN collapsing on heavy users (0.0962 → 0.0173, a 5.5× drop), with BPR overtaking it at 200+ interactions (p < 1e-6).

Hypothesis: unnormalized similarity summation dilutes signal for long histories. Two fixes were built — capping history to the 50 most recent items, and to the 50 with strongest neighbor signal. **Both made the heavy bucket slightly worse** (0.0173 → 0.0165, 0.0166), while leaving sparse buckets byte-identical (correct, since those users have fewer than 50 items). Hypothesis rejected.

The actual cause was Recall@K's denominator:

| History | Mean relevant test items | Recall@10 | Precision@10 | Recall@R |
|---|---|---|---|---|
| 1–20 | 2.90 | 0.0962 | 0.0273 | 0.0343 |
| 21–50 | 6.66 | 0.0619 | 0.0411 | 0.0442 |
| 51–200 | 18.30 | 0.0372 | 0.0645 | 0.0571 |
| 200+ | 59.34 | 0.0173 | **0.0909** | **0.0702** |

A 200+ user averages 59.3 relevant test items, so a *perfect* top-10 caps at 10/59 ≈ 0.17. Two metrics without that ceiling — Precision@10, and Recall@R (list sized to each user's own target count) — both trend the **opposite** direction.

**Corrected conclusion: Item-KNN performs better on users with more history, not worse.** The apparent collapse was an artifact of comparing Recall@K across groups with very different target counts.

Consequences:
- A proposed history-length routing rule (Item-KNN for sparse users, BPR for heavy) was **withdrawn** — it rested on the artifact.
- The aggregate three-model comparison is **unaffected**: all models were evaluated on identical users with identical targets, so the artifact cancels entirely.
- Recall@K is appropriate for comparing models on a fixed population, not for comparing populations to each other.

---

## Popularity bias and catalog coverage

Over a 2,000-user sample across 43,417 ranked items:

| Model | Mean popularity rank | Median | Distinct items recommended |
|---|---|---|---|
| Popularity | 1–10 by construction | — | 10 |
| BPR | 59.9 | 20.0 | 410 |
| Item-KNN | 133.2 | 28.0 | 698 |

Item-KNN reaches further into the long tail *and* ranks better. BPR's stronger popularity skew reflects residual influence from its bias term. Both distributions are heavily right-skewed (mean ≫ median).

---

## Known limitations

1. **Candidate universe is 43,417 movies**, not the full 87,585-movie catalog — evaluation ranks only within movies that have positive training interactions. Full-catalog and item cold-start are out of scope for the current benchmark.
2. **No hyperparameter search.** `n_neighbors=51` and `embedding_dim=64` were chosen, not tuned.
3. **Single uniform negative per positive** in BPR. Popularity-aware or hard-negative sampling is untested and could plausibly close part of the gap to Item-KNN.
4. **An earlier validation subsample was not random** (`index[:5000]` skews toward low userIds and thus earlier, more active accounts), which inflated validation estimates relative to full test.
5. **Ratings below 4 carry no signal** — a movie rated 2 is indistinguishable from one never seen.
6. **The three models exclude different "already seen" sets.** Popularity and BPR exclude a user's entire training history; Item-KNN excludes only the positively-rated part, so it can spend top-K slots on movies the user already rated 2.0. Those items are essentially never test targets, so the asymmetry most likely costs Item-KNN rather than flatters it — but the direction is reasoned, not measured. `scripts/seen_set_ablation.py` quantifies it; it has not been run yet.

---

## Roadmap

- [x] EDA, leakage-audited evaluation design, metric unit tests
- [x] Popularity baseline
- [x] Item-item collaborative filtering
- [x] BPR matrix factorization
- [ ] Content-based model using `movies.csv` + `tags.csv` — the first model to use non-behavioral signal, and the only approach able to handle item cold-start
- [ ] Hybrid: combine collaborative and content scores, test whether the combination beats either alone
- [ ] Hyperparameter tuning on the validation split
- [ ] Cold-start evaluation slice

---

## Repository structure

```
personalized-content-ranking/
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA, split design, leakage audit
│   └── 02_baseline_models.ipynb    # exploratory model development
├── src/
│   ├── data/loaders.py             # splits, positive filtering, canonical id maps
│   ├── models/
│   │   ├── popularity.py           # popularity baseline
│   │   ├── item_knn.py             # neighbour building + serving
│   │   └── bpr.py                  # model definition, loss, checkpoint serving
│   └── utils/metrics.py            # Recall@K, Precision@K, NDCG@K, Recall@R
├── scripts/
│   ├── build_item_neighbors.py     # rebuild neighbours + id maps
│   ├── evaluate.py                 # full test-set evaluation → leaderboard
│   └── seen_set_ablation.py        # quantifies the seen-set asymmetry
├── tests/test_metrics.py           # assertion-backed metric tests
├── data/
│   ├── raw/ml-32m/                 # MovieLens 32M (not tracked)
│   └── processed/                  # parquet splits, item_neighbors.npz,
│                                   #   id_maps.npz, leaderboard, per-user metrics
├── models/bpr_model.pt             # best checkpoint by validation recall (not tracked)
├── requirements.txt
└── README.md
```

The notebooks are the exploratory record, including the dead ends described
above. The reproducible pipeline is `src/` plus `scripts/`.

---

## Reproducing the results

```bash
pip install -r requirements.txt
pytest tests/                         # metric tests, no data required
```

Download [MovieLens 32M](https://grouplens.org/datasets/movielens/) into
`data/raw/ml-32m/`, then run `notebooks/01_data_exploration.ipynb` to write the
per-user chronological splits to `data/processed/`. After that:

```bash
python scripts/build_item_neighbors.py   # item_neighbors.npz + id_maps.npz
python scripts/evaluate.py               # leaderboard + segment analysis
```

**On artifact integrity.** Every model here indexes matrices and embeddings by
position, so the position → movieId mapping is part of the model. An earlier
version stored item neighbours in bare `.npy` files against a mapping that was
later rebound, leaving on-disk neighbours pointing into a different id space.
Neighbours are now stored in a single `.npz` alongside the `movie_ids` they
index, and the loaders reject any artifact whose row count disagrees with its
mapping.

## Stack

Python, pandas, NumPy, SciPy (sparse), scikit-learn, PyTorch, Matplotlib
