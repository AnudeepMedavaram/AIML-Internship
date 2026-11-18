
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException

# Allow importing the recommender from the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.recommend import MovieRecommender


app = FastAPI(
    title="Movie Recommendation API",
    description="Popularity-based movie recommendations with watched-movie filtering.",
    version="1.0.0"
)

recommender = None


@app.on_event("startup")
def load_recommender():
    global recommender
    recommender = MovieRecommender()


@app.get("/")
def home():
    return {
        "project": "Movie Recommendation Engine",
        "status": "running",
        "recommendation_method": "Popularity-based with watched-movie filtering",
        "endpoints": [
            "/popular",
            "/recommend/{user_id}",
            "/health",
            "/docs"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if recommender is not None else "not ready"
    }


@app.get("/popular")
def popular_movies(top_k: int = 10):
    if top_k < 1 or top_k > 100:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 100"
        )

    try:
        result = recommender.recommend(
            user_id=None,
            top_k=top_k
        )

        return {
            "recommendation_type": "popular",
            "count": len(result),
            "recommendations": result.to_dict(orient="records")
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/recommend/{user_id}")
def recommend_for_user(user_id: int, top_k: int = 10):
    if user_id < 1:
        raise HTTPException(
            status_code=400,
            detail="user_id must be positive"
        )

    if top_k < 1 or top_k > 100:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 100"
        )

    try:
        result = recommender.recommend(
            user_id=user_id,
            top_k=top_k
        )

        return {
            "user_id": user_id,
            "recommendation_type": "popularity with watched-movie filtering",
            "count": len(result),
            "recommendations": result.to_dict(orient="records")
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
