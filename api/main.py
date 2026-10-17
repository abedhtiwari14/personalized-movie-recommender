"""FastAPI service for item-KNN recommendations.

Serving needs only the precomputed neighbour artifact and the title lookup,
about 30 MB; no ratings or training data. Both load once at startup.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import loaders
from src.models.item_knn import ItemKNN

MAX_LIKED = 50

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["knn"] = ItemKNN.load()
    movies = loaders.load_movies()
    state["title"] = dict(zip(movies["movieId"], movies["title"]))
    state["genres"] = dict(zip(movies["movieId"], movies["genres"]))
    state["catalog"] = [
        (int(m), t) for m, t in zip(movies["movieId"], movies["title"])
        if m in state["knn"].movie_to_idx
    ]
    yield
    state.clear()


app = FastAPI(title="Movie Recommender", version="1.0", lifespan=lifespan)


class RecommendRequest(BaseModel):
    liked: list[int] = Field(..., min_length=1, max_length=MAX_LIKED)
    k: int = Field(10, ge=1, le=50)


class Recommendation(BaseModel):
    movieId: int
    title: str
    genres: str


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    latency_ms: float
    unknown_ids: list[int]


@app.get("/health")
def health():
    knn = state.get("knn")

    return {
        "status": "ok" if knn is not None else "loading",
        "movies": len(knn.movie_ids) if knn is not None else 0,
    }


@app.get("/movies")
def search_movies(q: str, limit: int = 20):
    """Substring search over titles the model can actually rank."""
    needle = q.strip().lower()

    if not needle:
        return {"results": []}

    hits = [
        {"movieId": m, "title": t}
        for m, t in state["catalog"] if needle in t.lower()
    ][:limit]

    return {"results": hits}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    knn = state["knn"]
    started = time.perf_counter()

    known = [m for m in request.liked if m in knn.movie_to_idx]
    unknown = [m for m in request.liked if m not in knn.movie_to_idx]

    if not known:
        raise HTTPException(
            status_code=422,
            detail="None of the supplied movieIds are in the model's catalog.",
        )

    ranked = knn.recommend(known, k=request.k)

    return RecommendResponse(
        recommendations=[
            Recommendation(
                movieId=int(m),
                title=state["title"].get(m, str(m)),
                genres=state["genres"].get(m, ""),
            )
            for m in ranked
        ],
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        unknown_ids=unknown,
    )