# Personalized Movie Recommendation System

An end-to-end Top-K recommendation system built on MovieLens 32M, benchmarking four
approaches under a leakage-audited temporal evaluation protocol.

**Result:** Item-KNN reaches Recall@10 of 0.0749 across 193,622 users, beating a popularity
baseline by 56% and BPR matrix factorization by 38%. The simplest personalized model won. Two
findings were retracted along the way, after their supporting evidence turned out to be
measurement artifacts rather than model behaviour.

## The problem

Given what a user has watched and liked, put the movies they will like next at the top of a
list of about 43,000 candidates.

This is a ranking problem, not a rating-prediction one. It does not matter whether the model
predicts 4.2 stars or 4.5 — what matters is whether the ten movies on screen are good ones.
So models are scored on where the right answers land in the list, not on prediction error.

Accuracy would be useless here. A user likes about 3 movies out of 43,417 candidates, so
guessing "not relevant" for everything scores 99.99%.

## The data

MovieLens 32M: 32,000,204 ratings from 200,948 users on 87,585 movies, covering 1995 to 2023.

A rating of 4 or higher counts as a "like" — 15,938,231 of them, about half the dataset. Lower
ratings are not treated as dislikes; they simply do not count as likes. A movie rated 2 and a
movie never seen look identical to these models.

| File | What it holds | Used by |
|---|---|---|
| `ratings.csv` | who rated what, and when | popularity, item-KNN, BPR |
| `movies.csv` | titles and genres | content model |
| `tags.csv` | 2M user-written tags | content model |

## How it is evaluated

**The split.** Each user's ratings are sorted by date: the first 70% are training, the next 15%
validation, the last 15% test. Every model therefore predicts forward in time for every user,
and the split was audited for leakage — zero ordering violations across all 200,948 users.

The first attempt split by calendar date instead: everything before a cutoff was training. That
failed badly — **86.6% of test users had no training history at all**. MovieLens users tend to
rate a burst of movies and then stop, so a single date cut sorts users into eras rather than
splitting each person's history. It measured cold-start, not personalization.

One trade-off worth naming: with per-user splits, one user's training rating can fall later in
time than another user's test rating. That is standard for benchmarking personalization, but it
is less production-realistic than a single cutoff.

**The metrics**, all measured on 10 recommendations:

- **Recall@10** — of the movies a user liked in the test period, how many made the top 10
- **Precision@10** — of the 10 recommended, how many were liked
- **NDCG@10** — like recall, but gives more credit for ranking good movies higher

All three are covered by assertion-backed tests in `tests/test_metrics.py`.

## The four approaches

Everything below runs from `src/` and `scripts/`; the notebooks are the exploratory record.
Metrics are unit-tested, and the full four-model evaluation reproduces from two commands.

**Popularity.** Recommend the most-liked movies overall, skipping anything the user has already
rated. Not personalized. It is the bar any real model has to clear, and a sensible fallback for
brand-new users. Movies are ranked by number of likes, not average rating — a 5.0 from two
people should not outrank a 4.4 from fifty thousand.

**Item-KNN (item-to-item).** Two movies are similar if the same people liked both. Build a
sparse 200,501 × 43,417 user-movie matrix (dense would be ~8.7 billion cells), then measure
cosine similarity between movies. To recommend, take the movies a user liked, look up each
one's 51 nearest neighbours, and add up the similarity scores. Neighbours are precomputed
offline, so serving a recommendation is a lookup rather than a similarity computation.

**BPR (matrix factorization).** Learn 64 numbers for each user and each movie so that a user's
vector points toward the movies they like. Training uses triples of (user, a movie they liked, a
random movie they have not seen) with a pairwise loss, `-log σ(score_pos − score_neg)`, that
simply says "score the liked one higher." Written in PyTorch, but not a neural network — the
score is a dot product plus a per-movie bias, with no hidden layers.

**Content-based.** The only model here that uses no behavioural signal at all. Each movie
becomes a bag of tokens (genres plus user tags), TF-IDF weighted into a 26,285-dimensional
sparse vector. A user is represented as the L2-normalised **mean of the vectors of the movies
they liked**, which places the person in the same space as the films — so one cosine similarity
compares the two. Scoring is a single sparse matrix multiply against the whole catalog.

Tags are kept as whole tokens rather than split into words. A word-level tokenizer turns
`sci-fi` into `sci` + `fi` and `chick flick` into two meaningless halves. `sublinear_tf` damps
tag volume, so a movie tagged `classic` 500 times does not outweigh one tagged it twice by
250×; without it, raw counts quietly rebuild a popularity ranker out of content features.

### The same movie, seen two ways

Item-KNN and the content model disagree about what "similar to *The Shawshank Redemption*"
means, and both are right:

| Item-KNN (who watched what) | Content (what the film is about) |
|---|---|
| Pulp Fiction · 0.554 | Se7en · 0.498 |
| Silence of the Lambs · 0.547 | Lucky Number Slevin · 0.422 |
| Forrest Gump · 0.536 | The Sixth Sense · 0.397 |
| Schindler's List · 0.535 | *Escape from San Quentin*, *Within These Walls* |

Item-KNN found "people who liked Shawshank also liked…". The content model found **prison
movies** — including obscure ones from the 1930s–50s, which rank high precisely because they
have few tags, so a single token like `prison` dominates their vector.

## Results

### Warm benchmark — 193,622 users, 43,417 candidates

| Model | Recall@10 | Precision@10 | NDCG@10 |
|---|---|---|---|
| Popularity | 0.0480 | 0.0382 | 0.0530 |
| Content (genres + tags) | 0.0522 | 0.0447 | 0.0610 |
| BPR (Matrix Factorization) | 0.0543 | 0.0421 | 0.0584 |
| **Item-KNN (item-item CF)** | **0.0749** | **0.0563** | **0.0793** |

Personalization works: BPR beats the baseline by 13.2%. But the simple neighbour-based method
beats the learned one by 37.9%, and the baseline by 56.1%. A Wilcoxon signed-rank test on
193,622 paired per-user scores confirms the gaps are not chance.

The headline is that the more complicated model lost. Squeezing a user's taste into 64 numbers
throws away detail that item-to-item similarity keeps.

The content model is the quiet surprise. With no behavioural signal whatsoever it beats the
baseline on all three metrics, and **outranks BPR on Precision@10 and NDCG@10** despite lower
Recall@10 — it retrieves slightly fewer correct items but orders them better.

Item-KNN also reaches further into the catalog: across a 2,000-user sample it surfaced 729
distinct titles against BPR's 410 and popularity's 10.

### Cold benchmark — 3,479 users, 87,585 candidates

The warm benchmark ranks only movies that have training history, so by construction it contains
no new releases. The cold benchmark ranks the full catalog instead. 7,879 movies appear as test
targets with zero training likes — genuine recent releases like *The Creator* (2023) — carrying
10,411 positives. A further 36,289 catalog entries are neither trained on nor ever a target, and
act as distractors so the task is not artificially easy.

| Model | cold Recall@10 | cold items surfaced | overall Recall@10 |
|---|---|---|---|
| Popularity | 0.0000 | 0 | 0.0154 |
| Item-KNN | 0.0000 | 0 | 0.0224 |
| BPR (MF) | 0.0000 | 0 | 0.0166 |
| **Content (genres + tags)** | **0.0017** | **13** | 0.0183 |

The three behavioural models score **exactly zero**, by construction rather than by weakness: a
cold movie has no popularity count, no neighbour row and no embedding row. No amount of tuning
changes that.

Stated honestly, 13 hits out of 10,411 is a rounding error, and content does not *solve*
cold-start. What it does is move the number off zero, which nothing else here can. The
defensible claim is coverage, not accuracy.

One caveat on the tag signal: tags are user-generated, so they exist mainly where people have
already engaged — full coverage on the top 1,000 movies by popularity, about 70% on the tail.
Tags are closer to behaviour wearing content clothing than to true metadata. Genres are the part
genuinely available the day a film is released.

## Three things that went wrong

These were the most useful part of the project.

### 1. BPR quietly trained itself into a popularity ranker

The first BPR run scored 0.0475 — slightly *worse* than popularity. Training loss fell from
0.693 to 0.154 the whole way, so nothing looked wrong.

The score is `user · movie + movie_bias`. With the vectors starting at tiny random values, their
dot product and its gradients both start near zero, while `movie_bias` is a single free number
per movie with a full-sized gradient. The model took the easy route and learned popularity
through the bias alone. Measured afterwards:

```
mean user vector length:   0.0000
mean movie vector length:  0.0000
movie bias spread:         0.5370
```

Fixed by starting the vectors larger, raising the learning rate, dropping weight decay, and
stopping early on validation score. Result: 0.0543.

One failed fix is worth recording. Adding weight decay made the collapse worse, not better. Adam
shrinks every movie's vector on every step, but a movie only receives a gradient when it appears
in a batch — so rare movies were shrunk constantly and updated rarely. Early stopping, not
regularization, was the right tool.

**Lesson: the loss measured the training objective, not recommendation quality.**

### 2. A segment finding with two independent causes

Splitting users by how much history they had appeared to show Item-KNN collapsing on heavy
users, with BPR overtaking it. That suggested a routing rule: simple model for light users,
learned model for heavy ones. The conclusion was wrong twice, for unrelated reasons, and either
one alone would have been enough to withdraw it.

**Cause one — the metric.** The first hypothesis was that unnormalized similarity summation
dilutes signal over long histories. Two fixes were built, capping history to the 50 most recent
items and to the 50 with strongest neighbour signal. Both made the heavy bucket slightly *worse*
while leaving sparse buckets byte-identical. Hypothesis rejected — which is what prompted
questioning the metric instead of the model.

Recall@10 divides by how many movies the user liked in the test period, so it is capped at
`10 / len(relevant)`. Precision@10 and Recall@R have no such ceiling, and both run the opposite
way:

| History | Movies liked in test | Recall@10 ceiling | Recall@10 | Precision@10 | Recall@R |
|---|---|---|---|---|---|
| 1–20 | 2.9 | 3.45 | 0.1080 | 0.0307 | 0.0397 |
| 21–50 | 6.7 | 1.50 | 0.0715 | 0.0477 | 0.0525 |
| 51–200 | 18.3 | 0.55 | 0.0473 | 0.0828 | 0.0716 |
| 200+ | 59.3 | **0.17** | 0.0259 | **0.1362** | **0.0942** |

**Cause two — the harness.** The models were not excluding the same candidates. Popularity and
BPR skipped every movie a user had already rated; Item-KNN skipped only the ones they had rated
4 or above, so it wasted top-10 slots on films the user had already seen and disliked. That
penalty scaled with how many movies a user had rated below 4, which in turn scales with history
length — so the bug landed almost entirely on heavy users, the one group where BPR appeared to
win. Correcting it raised Item-KNN's overall Recall@10 by 16% and flipped that bucket outright:

| History | Item-KNN | BPR |
|---|---|---|
| 1–20 | 0.1080 | 0.0767 |
| 21–50 | 0.0715 | 0.0524 |
| 51–200 | 0.0473 | 0.0354 |
| 200+ | **0.0259** | 0.0202 |

Item-KNN wins every segment. The routing rule was withdrawn.

**Lesson: one believable explanation for a surprising result is not proof of the cause.**

### 3. Saved files that pointed at nothing

Item-KNN's neighbour table stores row numbers, and a separate lookup says which movie each row
means. The table was built once over 65,740 movies, then rebuilt over 43,417 — but only the
lookup was updated. The saved file still loaded, still had the right shape, and pointed into a
list of movies that no longer existed.

Fixed by storing the row numbers and the movie list in the same file, and refusing to load one
whose sizes disagree — making the failure impossible rather than merely unlikely.

**Lesson: a file of row numbers means nothing without the list it refers to, so they belong
together.**

## Scope

Candidates are the 43,417 movies with training history, not the full 87,585-movie catalog.
`n_neighbors=51` and `embedding_dim=64` were chosen, not tuned — the validation split exists and
has not been used for that yet. BPR draws one uniform negative per positive. Ratings below 4
carry no signal at all.

## Project layout

```
notebooks/   exploration and model development
src/         reusable code: data loading, models, metrics, evaluation
scripts/     run these to build artifacts and reproduce results
tests/       metric tests
data/        MovieLens files and processed splits (not in git)
models/      trained BPR checkpoint (not in git)
```

## Running it

```bash
pip install -r requirements.txt
pytest tests/
```

Download [MovieLens 32M](https://grouplens.org/datasets/movielens/) into `data/raw/ml-32m/`,
then run `notebooks/01_data_exploration.ipynb` to create the splits. After that:

```bash
python scripts/build_item_neighbors.py   # item neighbours + id mappings
python scripts/evaluate.py               # warm benchmark, three behavioural models
python scripts/evaluate_cold.py          # cold benchmark, three behavioural models
python scripts/evaluate_content.py       # content model, warm and cold
```

Every model indexes matrices and embeddings by row position, so the position → movieId mapping
is part of the model. Artifacts store the mapping in the same file as the data it indexes, and
loaders reject any file whose row count disagrees with its mapping.

Built with Python, pandas, NumPy, SciPy, scikit-learn and PyTorch.
