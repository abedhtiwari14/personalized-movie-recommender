---
title: Movie Recommender
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# Movie Recommender — MovieLens 32M

Item-to-item collaborative filtering over 32M ratings. Pick a few films you
like and the model ranks the 43,417 movies it knows about.

Two movies are similar if the same people liked both. Neighbours are
precomputed offline, so a recommendation is a lookup and a weighted sum —
p99 latency is around 1 ms.

Code and the full benchmark against popularity, BPR matrix factorization, a
content model and a two-tower network:
https://github.com/abedhtiwari14/personalized-movie-recommender
