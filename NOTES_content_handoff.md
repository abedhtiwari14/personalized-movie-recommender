# Handoff — evaluation interfaces

Everything you need to score a new recommender against both benchmarks. No
modelling guidance here by design.

**Repo state note:** the working tree has no content model — `src/models/`
holds only `popularity.py`, `item_knn.py` and `bpr.py`, and `data/processed/`
has no content artifacts. An earlier attempt (`content.py`,
`build_content_features.py`, `evaluate_content.py`, plus its feature matrices
and leaderboards) was committed in `69df5ca` and then removed, so it is
recoverable from history if you ever want to compare. Nothing in the working
tree imports or references it.

---

## 1. `full_universe()` vs the 43,417 universe

```python
loaders.full_universe(movies: pd.DataFrame | None = None) -> np.ndarray
```

Returns **every `movieId` in `movies.csv`, sorted ascending: 87,585 ids.**
Pass a preloaded `movies` frame to avoid re-reading the CSV; omit it and the
function loads it.

The 43,417 universe is a different thing with a different definition:

```python
loaders.warm_universe(train_positive: pd.DataFrame) -> np.ndarray
```

Returns movieIds with **at least one positive training interaction**, in
order of first appearance in the training split — *not* sorted. That ordering
is load-bearing: it indexes the Item-KNN item axis and the BPR movie
embedding rows, so do not sort it or rebuild it from a different frame.

| | `warm_universe()` | `full_universe()` |
|---|---|---|
| Size | 43,417 | 87,585 |
| Definition | ≥1 positive train interaction | every catalog entry |
| Order | first appearance in train | sorted ascending |
| Contains cold items? | **No, by construction** | Yes — 7,879 of them |
| Used by | warm benchmark | cold benchmark |

The consequence that matters: the warm universe *cannot* contain a cold item,
because "has positive training signal" is the definition of warm. Evaluating
cold-start against it is not a hard test, it is an impossible one.

Breakdown of the 87,585:

| Group | Count | Meaning |
|---|---|---|
| Warm | 43,417 | positive training signal |
| Cold | 7,879 | a test target, zero positive training signal |
| Neither | 36,289 | never trained on, never a target — distractors |

The 36,289 matter. Without them, every cold candidate would be relevant to
somebody, and "rank all unwatched items first" would score well.

---

## 2. The cold-start evaluation split

Not a file on disk — derived at load time from the existing parquet splits.

```python
loaders.cold_items(train_positive: pd.DataFrame,
                   test_frame: pd.DataFrame) -> np.ndarray
```

**Items.** 7,879 movieIds, sorted ascending. An item is cold iff it appears
as a positive test interaction **and** has zero positive *training*
interactions. Note both halves are defined on `rating >= 4`: a movie rated
only 2.0 in training still counts as cold, because the models never saw a
positive signal for it. These are real recent releases — *The Creator*
(2023), *A Haunting in Venice* (2023), *The Duke* (2021).

**Users.** 3,479, available as `context.cold_users`. A user qualifies iff at
least one of their positive test targets is a cold item. That is 1.8% of the
193,622 warm-benchmark users. They are a strict subset, so per-user results
are directly comparable between benchmarks.

**Targets.** Two per user, and they are different:

- **Cold targets** — `relevant_by_user[u] & set(cold_items)`. Only the cold
  subset. Drives `cold_recall@10`, whose denominator is the count of cold
  targets, not all targets. Total across all users: 10,411 positives, which
  is 0.47% of all test positives.
- **All targets** — `relevant_by_user[u]`, the user's full positive test set.
  Drives `overall_recall@10` over the same single ranking, so a model that
  chases cold items at the expense of everything else shows up as a drop
  here rather than as a win.

**Candidates.** The full 87,585. A model must return a ranking over that,
excluding `context.seen_for(u)`.

**Shape of the numbers.** `cold_recall@10` is a mean over 3,479 per-user
values, each `hits / len(cold_targets)`. `cold_items_surfaced` is the raw
integer count of cold hits across all users — report it alongside, because
at these rates the mean alone is hard to read.

---

## 3. Functions to call

### Load the shared context once

```python
from src.utils.evaluation import EvaluationContext, evaluate_warm, evaluate_cold

EvaluationContext.load() -> EvaluationContext
```

~10s and several GB, almost all of it the per-user seen sets. Build one,
reuse it across models. Fields:

| Field | Type | Contents |
|---|---|---|
| `relevant_by_user` | `pd.Series` | userId → set of positive **test** movieIds |
| `history` | `pd.Series` | userId → set of positive **train** movieIds |
| `seen` | `pd.Series` | userId → set of **all** train movieIds |
| `warm_universe` | `np.ndarray` | 43,417 |
| `full_universe` | `np.ndarray` | 87,585 |
| `cold_items` | `np.ndarray` | 7,879 |
| `users` | `list` | 193,622 warm-benchmark users |
| `cold_users` | `list` | 3,479 cold-benchmark users |

Accessors — use these rather than indexing the Series directly, since they
handle missing users:

```python
context.history_for(user_id)      -> set   # model INPUT: positive train history
context.seen_for(user_id)         -> set   # model EXCLUSION: full train history
context.cold_targets_for(user_id) -> set
```

**`history_for` and `seen_for` are not interchangeable.** Feed the model
`history_for`; exclude `seen_for`. Using `history_for` as the exclusion set
lets the model spend top-K slots on movies the user already rated below 4 and
costs ~16% relative — that was a real bug in this repo, documented in the
README footnote.

### The recommender interface

One callable, `Callable[[int], Sequence[int]]` — userId in, ranked movieIds
out, best first. It owns its own candidate restriction and seen-exclusion.
That is the entire contract.

```python
def my_recommender(user_id):
    ...
    return [movie_id, ...]
```

### Score it

```python
evaluate_warm(recommend, context, k=10, users=None,
              label="model", verbose=True) -> dict
```
Returns `{"model", "users", "recall@10", "precision@10", "ndcg@10",
"per_user_recall", "minutes"}`. `per_user_recall` is an `np.ndarray` aligned
to `users` — feed it to `scipy.stats.wilcoxon` against another model's array
for paired significance testing. Pass `users=context.users[:20000]` while
iterating; the full population takes minutes.

```python
evaluate_cold(recommend, context, k=10,
              label="model", verbose=True) -> dict
```
Returns `{"model", "users", "cold_recall@10", "cold_items_surfaced",
"overall_recall@10", "per_user_cold_recall", "minutes"}`. Always runs over
all 3,479 cold users.

### Metrics, if you want them directly

```python
from src.utils.metrics import recall_at_k, precision_at_k, ndcg_at_k, recall_at_r

recall_at_k(recommended: Sequence, relevant: set, k: int) -> float
precision_at_k(recommended: Sequence, relevant: set, k: int) -> float
ndcg_at_k(recommended: Sequence, relevant: set, k: int) -> float
recall_at_r(recommended: Sequence, relevant: set) -> float
```

`recall_at_k` is capped at `k / len(relevant)`, so it is not comparable
across user groups with different target counts. Use `recall_at_r` or
`precision_at_k` for that. `tests/test_metrics.py` encodes this.

### Loaders, if you need the raw frames

```python
loaders.load_split(split: str,
                   columns=("userId", "movieId", "positive")) -> pd.DataFrame
loaders.positives(frame: pd.DataFrame) -> pd.DataFrame
loaders.load_movies() -> pd.DataFrame
loaders.relevant_by_user(split_frame: pd.DataFrame) -> pd.Series
loaders.positive_movies_by_user(train_positive: pd.DataFrame) -> pd.Series
loaders.index_map(ids) -> dict           # id -> position
loaders.load_id_maps() -> dict           # movie_ids, knn_user_ids, bpr_user_ids
```

`split` is `"train" | "validation" | "test"`. **Do not sort the returned
frame** — row order is the chronological order the splits were written in,
and `warm_universe` ordering depends on it.

`validation` exists and has never been used for tuning. It is the correct
place to fit any blend weight or hyperparameter.

---

## 4. Reference numbers

Sanity-check a new model against these. Both benchmarks, k=10.

| Model | Warm R@10 | Warm P@10 | Cold R@10 | Cold items surfaced |
|---|---|---|---|---|
| Popularity | 0.0480 | 0.0382 | 0.0000 | 0 |
| BPR (MF) | 0.0543 | 0.0421 | 0.0000 | 0 |
| Item-KNN | 0.0749 | 0.0563 | 0.0000 | 0 |

Warm figures are the full 193,622 users under the consistent seen-set. The
three zeroes are structural, not weakness: a cold item has no popularity
count, no neighbour row and no embedding row.

Regenerate them with `python scripts/evaluate.py` (warm, ~5 min) and
`python scripts/evaluate_cold.py` (cold, ~20 s). The cold script is also the
shortest worked example of the harness — three recommenders defined as
lambdas, passed straight to `evaluate_cold`. Read it before wiring up your
own.

On a 20,000-user prefix, Item-KNN warm R@10 reads 0.0758 rather than 0.0749 —
that gap is sampling, not a bug, so compare like-for-like.

**A caveat on the cold benchmark.** 10,411 cold positives spread over 3,479
users against 87,585 candidates is a hard, sparse target. Expect small
absolute numbers, and report `cold_items_surfaced` alongside the mean.
Anything above zero is qualitatively different from the behavioural models,
which cannot score here at all — but treat single-digit hit counts as
directional, not as a result.

---

## Known duplication

`scripts/evaluate.py` still carries its own inline evaluation loop rather
than importing `src/utils/evaluation.py`. The two agree (verified: Item-KNN
cold `0.0000 / 0.0224` from both), but it should be migrated to one
implementation. Small, mechanical, and worth doing before the file grows.
