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
| **Item-KNN (item-item CF)** | **0.0749** | **0.0563** | **0.0793** |

- BPR beats the non-personalized baseline by **+13.2%** relative on Recall@10.
- Item-KNN beats BPR by **+37.9%** and the baseline by **+56.1%**. Wilcoxon signed-rank test on 193,622 paired per-user recalls: **p < 1e-300**.

The simplest personalized model wins, and wins in every user segment. Added model complexity did not pay off on this benchmark.

<sup>**Note on a corrected evaluation.** Earlier versions of these results reported Item-KNN at 0.0644 / 0.0456 / 0.0644. That run had a harness inconsistency: Popularity and BPR excluded a user's *entire* training history from their candidates, while Item-KNN excluded only the positively-rated part — so Item-KNN could spend top-10 slots on movies the user had already rated below 4, which are essentially never test targets. The three models were not ranking the same candidate set. Under a consistent seen-set the numbers above are the fair comparison; Popularity and BPR are unchanged, since they were already correct. `scripts/evaluate.py --seen-set legacy` reproduces the old figures exactly, and `scripts/seen_set_ablation.py` measures the gap directly (+15.9% relative on a 20,000-user random sample, Wilcoxon p ≈ 3.6e-280, with 8.5% of users affected).</sup>

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

### 2. A segment finding with two independent causes of death

Recall@10 bucketed by user history length appeared to show Item-KNN collapsing on heavy users (0.0962 → 0.0173, a 5.5× drop), with BPR overtaking it at 200+ interactions (p < 1e-6). That reading motivated a proposed history-length routing rule: Item-KNN for sparse users, BPR for heavy ones.

The rule was wrong twice over, for unrelated reasons. Both had to be found separately, and either one alone would have been enough to withdraw it.

**Cause of death 1 — the metric.** The first hypothesis was that unnormalized similarity summation dilutes signal over long histories. Two fixes were built: capping history to the 50 most recent items, and to the 50 with strongest neighbor signal. **Both made the heavy bucket slightly worse**, while leaving sparse buckets byte-identical (correct — those users have fewer than 50 items). Hypothesis rejected, which is what prompted questioning the metric instead of the model.

Recall@K divides by the user's own target count, so it is capped at `K / len(relevant)`:

| History | Mean relevant test items | Recall@10 ceiling | Recall@10 | Precision@10 | Recall@R |
|---|---|---|---|---|---|
| 1–20 | 2.90 | 3.45 | 0.1080 | 0.0307 | 0.0397 |
| 21–50 | 6.66 | 1.50 | 0.0715 | 0.0477 | 0.0525 |
| 51–200 | 18.30 | 0.55 | 0.0473 | 0.0828 | 0.0716 |
| 200+ | 59.34 | **0.17** | 0.0259 | **0.1362** | **0.0942** |

A 200+ user averages 59.3 relevant test items, so a *perfect* top-10 caps at 10/59 ≈ 0.17. Two metrics without that ceiling — Precision@10, and Recall@R (list sized to each user's own target count) — both trend the **opposite** direction.

**Cause of death 2 — the harness.** The seen-set inconsistency described above was not uniform across users. Item-KNN's penalty was proportional to how many movies a user had rated below 4, which scales with total history length — so the bug was concentrated almost entirely in the heavy-user bucket, the one place BPR appeared to win. Correcting it flips that bucket outright:

| History | Item-KNN R@10 | BPR R@10 | Winner |
|---|---|---|---|
| 1–20 | 0.1080 | 0.0767 | Item-KNN |
| 21–50 | 0.0715 | 0.0524 | Item-KNN |
| 51–200 | 0.0473 | 0.0354 | Item-KNN |
| 200+ | **0.0259** | 0.0202 | **Item-KNN** |

**Corrected conclusion: Item-KNN performs better on users with more history, not worse, and beats BPR in every segment.** There is no crossover and no population for which BPR is the better choice on this benchmark.

Consequences:
- The history-length routing rule is **withdrawn**. It rested on an artifact *and* on a bug, each sufficient on its own to invalidate it.
- The aggregate three-model comparison was **unaffected by the metric artifact** — all models were evaluated on identical users with identical targets, so it cancels entirely. It *was* affected by the harness bug, which is why the headline number changed.
- Recall@K is appropriate for comparing models on a fixed population, not for comparing populations to each other.
- Two unrelated errors produced the same false conclusion. A single plausible mechanism for a surprising result is not evidence that you have found the cause.

---

## Popularity bias and catalog coverage

Over a 2,000-user sample across 43,417 ranked items:

| Model | Mean popularity rank | Median | Distinct items recommended |
|---|---|---|---|
| Popularity | 1–10 by construction | — | 10 |
| BPR | 59.9 | 20.0 | 410 |
| Item-KNN | 139.7 | 30.0 | 729 |

Item-KNN reaches further into the long tail *and* ranks better. BPR's stronger popularity skew reflects residual influence from its bias term. Both distributions are heavily right-skewed (mean ≫ median).

---

## Two benchmarks

**Warm** — 193,622 users ranking the 43,417 movies with positive training signal. This is the benchmark all results above are reported on. Every candidate has behavioural history, which is the regime collaborative filtering is built for.

**Cold** — 3,479 users who have at least one test target with *zero* positive training interactions, ranking the full 87,585-movie catalog. There are 7,879 such items carrying 10,411 test positives (0.47% of all test positives); they are genuine recent releases — *The Creator* (2023), *A Haunting in Venice* (2023). The remaining 36,289 catalog entries are neither trained on nor ever a target, and serve as distractors so the task is not artificially easy.

Popularity, Item-KNN and BPR score **exactly zero** on the cold benchmark, by construction rather than by weakness: a cold item has no popularity count, no neighbour row, and no embedding row. No amount of tuning changes this. It is the one gap in the current system that only non-behavioural signal can close.

---

## Known limitations

1. **The warm benchmark's candidate universe is 43,417 movies**, not the full 87,585-movie catalog — it ranks only within movies that have positive training interactions, and therefore contains no cold items by construction. A separate cold benchmark over the full catalog now exists (see below); the two are reported separately because they answer different questions.
2. **No hyperparameter search.** `n_neighbors=51` and `embedding_dim=64` were chosen, not tuned.
3. **Single uniform negative per positive** in BPR. Popularity-aware or hard-negative sampling is untested and could plausibly close part of the gap to Item-KNN.
4. **An earlier validation subsample was not random** (`index[:5000]` skews toward low userIds and thus earlier, more active accounts), which inflated validation estimates relative to full test.
5. **Ratings below 4 carry no signal** — a movie rated 2 is indistinguishable from one never seen.
6. **Item-KNN neighbour ties are not bit-reproducible.** `NearestNeighbors` with `n_jobs=-1` breaks cosine ties non-deterministically, and rare movies sharing an identical user set tie at similarity 1.0. Recall@10 moves in the 7th decimal between rebuilds; it rounds identically, but the artifact is not byte-stable.

---

## Roadmap

- [x] EDA, leakage-audited evaluation design, metric unit tests
- [x] Popularity baseline
- [x] Item-item collaborative filtering
- [x] BPR matrix factorization
- [x] Reproducible pipeline extracted from notebooks; assertion-backed metric tests
- [x] Seen-set consistency fix across all three models
- [x] Cold-start evaluation slice — full-catalog candidate universe and cold-item targets
- [ ] Content-based model using `movies.csv` + `tags.csv` — the first model to use non-behavioral signal, and the only approach able to score a cold item at all
- [ ] Hybrid: combine collaborative and content scores, test whether the combination beats either alone
- [ ] Hyperparameter tuning on the validation split

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
│   ├── evaluate.py                 # warm evaluation → leaderboard (--seen-set)
│   ├── evaluate_cold.py            # cold benchmark, full-catalog candidates
│   ├── seen_set_ablation.py        # measures the seen-set asymmetry
│   └── segment_recall_at_r.py      # Recall@R by history bucket
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
python scripts/evaluate.py               # warm leaderboard + segment analysis
python scripts/evaluate_cold.py          # cold benchmark over the full catalog
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
