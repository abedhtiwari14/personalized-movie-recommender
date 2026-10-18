"""Gradio demo: collaborative and content recommendations side by side.

Both models answer "what is similar to the films you picked", from different
evidence. Item-KNN uses who watched what; the content model uses genres and
user tags. They disagree, and both are right.

The picker offers the 2,000 most-watched titles so the dropdown stays
usable; both models rank all 43,417 movies.
"""

import gradio as gr
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.content import recommend_content
from src.models.item_knn import ItemKNN

TOP_K = 10

knn = ItemKNN.load("item_neighbors.npz")

with np.load("content_features.npz", allow_pickle=False) as blob:
    content_X = csr_matrix(
        (blob["data"], blob["indices"], blob["indptr"]), shape=tuple(blob["shape"])
    )
    content_ids = blob["movie_ids"]

content_idx = {int(m): i for i, m in enumerate(content_ids)}

titles = pd.read_csv("titles.csv")
picker = pd.read_csv("picker.csv").sort_values("title")

title_of = dict(zip(titles["movieId"], titles["title"]))
genres_of = dict(zip(titles["movieId"], titles["genres"]))
id_of = dict(zip(picker["title"], picker["movieId"]))

CHOICES = picker["title"].tolist()

EXAMPLES = [
    ["Shawshank Redemption, The (1994)"],
    ["Toy Story (1995)", "Lion King, The (1994)", "Finding Nemo (2003)"],
    ["Alien (1979)", "Blade Runner (1982)", "Thing, The (1982)"],
    ["Notting Hill (1999)", "Bridget Jones's Diary (2001)"],
]

EMPTY = pd.DataFrame(columns=["#", "Title", "Genres"])


def _table(movie_ids):
    return pd.DataFrame(
        [
            {
                "#": i,
                "Title": title_of.get(m, str(m)),
                "Genres": genres_of.get(m, "").replace("|", ", "),
            }
            for i, m in enumerate(movie_ids, start=1)
        ]
    )


def recommend(selected, k):
    if not selected:
        return EMPTY, EMPTY, "Pick at least one film."

    liked = [id_of[t] for t in selected if t in id_of]
    k = int(k)

    collaborative = knn.recommend(liked, k=k)
    content = recommend_content(
        content_X, content_idx, content_ids, liked, seen=liked, k=k
    )

    overlap = len(set(collaborative) & set(content))
    note = (
        f"Ranked {len(knn.movie_ids):,} movies from {len(liked)} "
        f"liked film{'s' if len(liked) != 1 else ''}. "
        f"The two lists share **{overlap} of {k}** titles."
    )

    return _table(collaborative), _table(content), note


with gr.Blocks(title="Movie Recommender") as demo:
    gr.Markdown(
        "# Movie Recommender — MovieLens 32M\n"
        "Two models, two notions of similarity, same input.\n\n"
        "**Collaborative** learns from behaviour: two films are similar if the "
        "same people liked both. It has never seen a genre or a plot.\n"
        "**Content** learns from description: genres and 2M user-written tags, "
        "TF-IDF weighted. It has never seen who watched what.\n\n"
        "Pick *Shawshank Redemption* and watch them disagree — one returns the "
        "films its audience also liked, the other returns prison movies."
    )

    with gr.Row():
        picked = gr.Dropdown(
            choices=CHOICES,
            multiselect=True,
            label="Films you like",
            info="The 2,000 most-watched titles; type to search.",
            scale=4,
        )
        k = gr.Slider(5, 25, value=TOP_K, step=5, label="How many", scale=1)

    gr.Examples(examples=EXAMPLES, inputs=picked, label="Try one")

    note = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Collaborative — who watched what")
            collab_out = gr.Dataframe(
                headers=["#", "Title", "Genres"], wrap=True, interactive=False
            )
        with gr.Column():
            gr.Markdown("### Content — genres and tags")
            content_out = gr.Dataframe(
                headers=["#", "Title", "Genres"], wrap=True, interactive=False
            )

    gr.Markdown(
        "---\n"
        "On the benchmark behind this demo the collaborative model is the "
        "stronger of the two: Recall@10 of 0.0749 against 0.0522, across "
        "193,622 users. The content model earns its place elsewhere — it is "
        "the only one of five that can rank a film nobody has watched yet.\n\n"
        "[Code, benchmark and write-up]"
        "(https://github.com/abedhtiwari14/personalized-movie-recommender)"
    )

    for event in (picked.change, k.change):
        event(recommend, inputs=[picked, k], outputs=[collab_out, content_out, note])


if __name__ == "__main__":
    demo.launch()
