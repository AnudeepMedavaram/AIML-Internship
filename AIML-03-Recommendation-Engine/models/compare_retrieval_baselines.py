
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

FEATURES = ROOT / "features" / "output"
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "output"

CHECKPOINT = MODEL_DIR / "two_tower_model.pt"
USER_MAP = FEATURES / "user_id_map.parquet"
MOVIE_MAP = FEATURES / "movie_id_map.parquet"
POPULAR_MOVIES = FEATURES / "popular_movies.parquet"

TRAIN_FILE = PROCESSED / "train.parquet"
VAL_FILE = PROCESSED / "validation.parquet"

OUTPUT_FILE = MODEL_DIR / "baseline_comparison.json"

TOP_K = 10
MAX_USERS = 200
RANDOM_SEED = 42

torch.set_num_threads(2)


def recall_at_k(recommended, relevant):
    if not relevant:
        return 0.0
    return len(set(recommended[:TOP_K]) & relevant) / len(relevant)


def ndcg_at_k(recommended, relevant):
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, movie in enumerate(recommended[:TOP_K], start=1)
        if movie in relevant
    )

    ideal_hits = min(len(relevant), TOP_K)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg


def calculate_metrics(recommendations, relevant_by_user):
    recalls = []
    ndcgs = []

    for user_id, relevant in relevant_by_user.items():
        recommended = recommendations.get(user_id, [])

        recalls.append(recall_at_k(recommended, relevant))
        ndcgs.append(ndcg_at_k(recommended, relevant))

    return {
        "recall_at_10": float(np.mean(recalls)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "users_evaluated": len(recalls),
    }


def main():
    print("=" * 65)
    print("POPULARITY VS TWO-TOWER COMPARISON")
    print("=" * 65)

    # --------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------

    print("\nStep 1: Loading data...")

    popular = pd.read_parquet(POPULAR_MOVIES)

    # Preserve the existing popularity ranking.
    candidate_movies = popular["movieId"].astype(int).tolist()
    candidate_set = set(candidate_movies)

    user_map = pd.read_parquet(USER_MAP)
    movie_map = pd.read_parquet(MOVIE_MAP)

    user_lookup = dict(
        zip(user_map["userId"], user_map["user_idx"])
    )

    movie_lookup = dict(
        zip(movie_map["movieId"], movie_map["movie_idx"])
    )

    candidate_movies = [
        movie for movie in candidate_movies
        if movie in movie_lookup
    ]

    candidate_set = set(candidate_movies)

    print("Popularity candidates:", len(candidate_movies))

    # --------------------------------------------------
    # STEP 2: TRAINING HISTORY
    # --------------------------------------------------

    print("\nStep 2: Loading training history...")

    train = pd.read_parquet(
        TRAIN_FILE,
        columns=["userId", "movieId"]
    )

    train["userId"] = train["userId"].astype(int)
    train["movieId"] = train["movieId"].astype(int)

    seen_by_user = (
        train.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    # --------------------------------------------------
    # STEP 3: VALIDATION RELEVANCE
    # --------------------------------------------------

    print("\nStep 3: Preparing validation users...")

    validation = pd.read_parquet(
        VAL_FILE,
        columns=["userId", "movieId", "rating"]
    )

    validation = validation[
        validation["rating"] >= 4.0
    ].copy()

    validation["userId"] = validation["userId"].astype(int)
    validation["movieId"] = validation["movieId"].astype(int)

    validation = validation[
        validation["userId"].isin(user_lookup)
        & validation["movieId"].isin(candidate_set)
    ].copy()

    validation["seen"] = validation.apply(
        lambda row: row["movieId"] in seen_by_user.get(
            row["userId"], set()
        ),
        axis=1
    )

    validation = validation[~validation["seen"]]

    relevant_by_user = (
        validation.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    eligible_users = sorted(relevant_by_user.keys())

    rng = np.random.default_rng(RANDOM_SEED)

    selected_users = rng.choice(
        eligible_users,
        size=min(MAX_USERS, len(eligible_users)),
        replace=False
    )

    selected_users = [int(user) for user in selected_users]

    relevant_by_user = {
        user: relevant_by_user[user]
        for user in selected_users
    }

    print("Selected users:", len(selected_users))

    # --------------------------------------------------
    # STEP 4: POPULARITY RECOMMENDATIONS
    # --------------------------------------------------

    print("\nStep 4: Evaluating popularity baseline...")

    popularity_recommendations = {}

    for user_id in selected_users:
        seen = seen_by_user.get(user_id, set())

        recommendations = [
            movie for movie in candidate_movies
            if movie not in seen
        ][:TOP_K]

        popularity_recommendations[user_id] = recommendations

    popularity_metrics = calculate_metrics(
        popularity_recommendations,
        relevant_by_user
    )

    # --------------------------------------------------
    # STEP 5: TWO-TOWER RECOMMENDATIONS
    # --------------------------------------------------

    print("\nStep 5: Evaluating two-tower model...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    state = checkpoint["model_state_dict"]

    user_embeddings = state["user_embedding.weight"].float()
    movie_embeddings = state["movie_embedding.weight"].float()

    user_bias = state["user_bias.weight"].float()
    movie_bias = state["movie_bias.weight"].float()

    embedding_dim = checkpoint["embedding_dim"]

    candidate_indices = np.array(
        [movie_lookup[movie] for movie in candidate_movies],
        dtype=np.int64
    )

    candidate_vectors = movie_embeddings[
        candidate_indices
    ].numpy()

    candidate_biases = movie_bias[
        candidate_indices
    ].numpy().reshape(-1)

    candidate_vectors = candidate_vectors / np.sqrt(embedding_dim)

    two_tower_recommendations = {}

    for number, user_id in enumerate(selected_users, start=1):

        user_idx = user_lookup[user_id]

        user_vector = user_embeddings[user_idx].numpy()
        bias = float(user_bias[user_idx].item())

        scores = (
            candidate_vectors @ user_vector
            + candidate_biases
            + bias
        )

        seen = seen_by_user.get(user_id, set())

        # Exclude movies watched during training.
        unseen_mask = np.array(
            [movie not in seen for movie in candidate_movies]
        )

        scores[~unseen_mask] = -np.inf

        top_positions = np.argsort(scores)[-TOP_K:][::-1]

        recommendations = [
            candidate_movies[position]
            for position in top_positions
            if np.isfinite(scores[position])
        ]

        two_tower_recommendations[user_id] = recommendations

        if number % 50 == 0:
            print(f"Evaluated {number}/{len(selected_users)} users")

    two_tower_metrics = calculate_metrics(
        two_tower_recommendations,
        relevant_by_user
    )

    # --------------------------------------------------
    # STEP 6: RESULTS
    # --------------------------------------------------

    results = {
        "evaluation": {
            "dataset": "MovieLens 25M validation",
            "users": len(selected_users),
            "candidate_movies": len(candidate_movies),
            "top_k": TOP_K,
            "random_seed": RANDOM_SEED,
            "relevance_rule": "rating >= 4.0",
            "seen_training_movies_excluded": True,
            "candidate_source": "popular_movies.parquet"
        },
        "popularity_baseline": popularity_metrics,
        "two_tower": two_tower_metrics
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    print("\nPOPULARITY BASELINE")
    print(f"Recall@10: {popularity_metrics['recall_at_10']:.5f}")
    print(f"NDCG@10:   {popularity_metrics['ndcg_at_10']:.5f}")

    print("\nTWO-TOWER MODEL")
    print(f"Recall@10: {two_tower_metrics['recall_at_10']:.5f}")
    print(f"NDCG@10:   {two_tower_metrics['ndcg_at_10']:.5f}")

    print("\nSaved:", OUTPUT_FILE)
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()

