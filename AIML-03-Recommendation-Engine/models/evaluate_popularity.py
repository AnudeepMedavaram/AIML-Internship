
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.parquet"

POPULAR_PATH = ROOT / "features" / "output" / "popular_movies.parquet"

RETRIEVAL_METRICS_PATH = ROOT / "models" / "output" / "retrieval_metrics.json"
OUTPUT_PATH = ROOT / "models" / "output" / "popularity_metrics.json"

TOP_K = 10
MAX_USERS = 200
RANDOM_SEED = 42


def recall_at_k(recommended, relevant, k):
    hits = len(set(recommended[:k]) & relevant)

    if not relevant:
        return 0.0

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0

    for rank, movie_id in enumerate(recommended[:k], start=1):
        if movie_id in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg


def main():

    print("=" * 65)
    print("POPULARITY BASELINE EVALUATION")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Load training history
    # --------------------------------------------------

    print("\nStep 1: Loading training history...")

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=["userId", "movieId"]
    )

    seen_by_user = (
        train.groupby("userId")["movieId"]
        .apply(lambda values: set(values.astype(int)))
        .to_dict()
    )

    print(f"Users with training history: {len(seen_by_user):,}")

    # --------------------------------------------------
    # 2. Load validation positives
    # --------------------------------------------------

    print("\nStep 2: Loading validation positives...")

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=["userId", "movieId", "rating"]
    )

    validation = validation[validation["rating"] >= 4.0].copy()

    validation["userId"] = validation["userId"].astype(int)
    validation["movieId"] = validation["movieId"].astype(int)

    # Use the same model ID map coverage as retrieval evaluation.
    user_map = pd.read_parquet(
        ROOT / "features" / "output" / "user_id_map.parquet"
    )

    movie_map = pd.read_parquet(
        ROOT / "features" / "output" / "movie_id_map.parquet"
    )

    known_users = set(user_map["userId"].astype(int))
    known_movies = set(movie_map["movieId"].astype(int))

    validation = validation[
        validation["userId"].isin(known_users)
        & validation["movieId"].isin(known_movies)
    ]

    relevant_by_user = (
        validation.groupby("userId")["movieId"]
        .apply(lambda values: set(values.astype(int)))
        .to_dict()
    )

    eligible_users = []

    for user_id, relevant_movies in relevant_by_user.items():

        seen = seen_by_user.get(user_id, set())

        unseen_relevant = relevant_movies - seen

        if unseen_relevant:
            relevant_by_user[user_id] = unseen_relevant
            eligible_users.append(user_id)

    print(f"Eligible validation users: {len(eligible_users):,}")

    # --------------------------------------------------
    # 3. Select the same reproducible user sample
    # --------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    selected_count = min(MAX_USERS, len(eligible_users))

    selected_users = rng.choice(
        eligible_users,
        size=selected_count,
        replace=False
    )

    print(f"Users selected: {selected_count}")

    # --------------------------------------------------
    # 4. Load popularity-ranked movies
    # --------------------------------------------------

    print("\nStep 3: Loading popular movies...")

    popular = pd.read_parquet(POPULAR_PATH)

    popular = popular.sort_values(
        ["rating_count", "average_rating"],
        ascending=[False, False]
    )

    popular_movie_ids = (
        popular["movieId"].astype(int).tolist()
    )

    print(f"Popularity candidates: {len(popular_movie_ids):,}")

    # --------------------------------------------------
    # 5. Evaluate popularity recommendations
    # --------------------------------------------------

    print("\nStep 4: Evaluating popularity recommendations...")

    recalls = []
    ndcgs = []

    for number, user_id in enumerate(selected_users, start=1):

        user_id = int(user_id)

        seen = seen_by_user.get(user_id, set())

        recommendations = []

        for movie_id in popular_movie_ids:

            if movie_id in seen:
                continue

            recommendations.append(movie_id)

            if len(recommendations) == TOP_K:
                break

        relevant = relevant_by_user[user_id]

        recalls.append(
            recall_at_k(recommendations, relevant, TOP_K)
        )

        ndcgs.append(
            ndcg_at_k(recommendations, relevant, TOP_K)
        )

        if number % 25 == 0:
            print(f"Evaluated {number}/{selected_count} users")

    # --------------------------------------------------
    # 6. Save metrics
    # --------------------------------------------------

    metrics = {
        "model": "popularity_baseline",
        "evaluation_dataset": "MovieLens 25M validation",
        "relevance_rule": "rating >= 4.0",
        "users_evaluated": int(selected_count),
        "random_seed": RANDOM_SEED,
        "top_k": TOP_K,
        "recall_at_10": float(np.mean(recalls)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "popularity_candidates": len(popular_movie_ids),
        "seen_training_movies_excluded": True
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # --------------------------------------------------
    # 7. Compare against two-tower model
    # --------------------------------------------------

    retrieval = json.load(
        open(RETRIEVAL_METRICS_PATH, encoding="utf-8")
    )

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(
        f"{'Metric':<18}"
        f"{'Popularity':>15}"
        f"{'Two-Tower':>15}"
    )

    print("-" * 48)

    print(
        f"{'Recall@10':<18}"
        f"{metrics['recall_at_10']:>15.6f}"
        f"{retrieval['recall_at_10']:>15.6f}"
    )

    print(
        f"{'NDCG@10':<18}"
        f"{metrics['ndcg_at_10']:>15.6f}"
        f"{retrieval['ndcg_at_10']:>15.6f}"
    )

    print(f"\nPopularity metrics saved to: {OUTPUT_PATH}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
