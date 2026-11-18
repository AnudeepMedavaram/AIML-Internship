
from pathlib import Path
import json

import faiss
import numpy as np
import pandas as pd
import torch


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = ROOT / "models" / "output" / "two_tower_model.pt"
USER_MAP_PATH = ROOT / "features" / "output" / "user_id_map.parquet"
MOVIE_MAP_PATH = ROOT / "features" / "output" / "movie_id_map.parquet"

INDEX_PATH = ROOT / "retrieval" / "output" / "movie_index.faiss"
MOVIE_IDS_PATH = ROOT / "retrieval" / "output" / "movie_ids.npy"

TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.parquet"

PAIRS_PATH = ROOT / "models" / "output" / "train_pairs.parquet"

OUTPUT_DIR = ROOT / "models" / "output"
METRICS_PATH = OUTPUT_DIR / "retrieval_metrics_by_coverage.json"

TOP_K = 10
MAX_USERS = 200
RANDOM_SEED = 42

torch.set_num_threads(2)


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def recall_at_k(recommended, relevant, k):

    if not relevant:
        return 0.0

    hits = len(set(recommended[:k]) & relevant)

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


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("=" * 70)
    print("TWO-TOWER RETRIEVAL — COVERAGE-WISE EVALUATION")
    print("=" * 70)

    # --------------------------------------------------
    # STEP 1: LOAD MODEL AND INDEX
    # --------------------------------------------------

    print("\nStep 1: Loading model and FAISS index...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    state = checkpoint["model_state_dict"]

    user_embeddings = state["user_embedding.weight"].float()
    user_biases = state["user_bias.weight"].float()

    embedding_dim = checkpoint["embedding_dim"]

    user_map = pd.read_parquet(USER_MAP_PATH)
    movie_map = pd.read_parquet(MOVIE_MAP_PATH)

    user_lookup = dict(
        zip(user_map["userId"], user_map["user_idx"])
    )

    movie_lookup = dict(
        zip(movie_map["movieId"], movie_map["movie_idx"])
    )

    index = faiss.read_index(str(INDEX_PATH))
    movie_ids = np.load(MOVIE_IDS_PATH)

    if index.d != embedding_dim + 2:
        raise ValueError("FAISS dimension mismatch.")

    if index.ntotal != len(movie_ids):
        raise ValueError("FAISS movie mapping mismatch.")

    print(f"Model users: {len(user_lookup):,}")
    print(f"Catalogue movies: {index.ntotal:,}")

    # --------------------------------------------------
    # STEP 2: IDENTIFY TRAINED EMBEDDINGS
    # --------------------------------------------------

    print("\nStep 2: Identifying trained embeddings...")

    pairs = pd.read_parquet(
        PAIRS_PATH,
        columns=["user_idx", "movie_idx"]
    )

    trained_users = set(
        pairs["user_idx"].unique()
    )

    trained_movies = set(
        pairs["movie_idx"].unique()
    )

    print(f"Users represented in pairs: {len(trained_users):,}")
    print(f"Movies represented in pairs: {len(trained_movies):,}")

    # --------------------------------------------------
    # STEP 3: LOAD TRAINING HISTORY
    # --------------------------------------------------

    print("\nStep 3: Loading training history...")

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=["userId", "movieId"]
    )

    train["userId"] = train["userId"].astype(int)
    train["movieId"] = train["movieId"].astype(int)

    seen_by_user = (
        train.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    print(f"Users with training history: {len(seen_by_user):,}")

    # --------------------------------------------------
    # STEP 4: LOAD VALIDATION POSITIVES
    # --------------------------------------------------

    print("\nStep 4: Loading validation positives...")

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=["userId", "movieId", "rating"]
    )

    validation = validation[
        validation["rating"] >= 4.0
    ].copy()

    validation["userId"] = validation["userId"].astype(int)
    validation["movieId"] = validation["movieId"].astype(int)

    # Retain only users and movies known to the model.

    validation = validation[
        validation["userId"].isin(user_lookup)
        & validation["movieId"].isin(movie_lookup)
    ].copy()

    # Remove movies already watched during training.

    validation["user_idx"] = validation["userId"].map(user_lookup)
    validation["movie_idx"] = validation["movieId"].map(movie_lookup)

    validation["user_trained"] = validation["user_idx"].isin(
        trained_users
    )

    validation["movie_trained"] = validation["movie_idx"].isin(
        trained_movies
    )

    validation["seen_in_train"] = validation.apply(
        lambda row: row["movieId"] in seen_by_user.get(
            row["userId"], set()
        ),
        axis=1
    )

    validation = validation[
        ~validation["seen_in_train"]
    ].copy()

    relevant_by_user = (
        validation.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    # --------------------------------------------------
    # STEP 5: SELECT USERS
    # --------------------------------------------------

    print("\nStep 5: Selecting evaluation users...")

    eligible_users = list(relevant_by_user.keys())

    if not eligible_users:
        raise ValueError("No eligible validation users.")

    rng = np.random.default_rng(RANDOM_SEED)

    selected_users = rng.choice(
        eligible_users,
        size=min(MAX_USERS, len(eligible_users)),
        replace=False
    )

    selected_users = [int(user_id) for user_id in selected_users]

    print(f"Selected users: {len(selected_users)}")

    # --------------------------------------------------
    # STEP 6: COVERAGE-WISE RETRIEVAL
    # --------------------------------------------------

    print("\nStep 6: Evaluating recommendations...")

    groups = {
        "warm_user_warm_movie": {
            "description": "User and relevant movie represented in training pairs",
            "recalls": [],
            "ndcgs": [],
            "users": 0,
            "relevant_movies": 0
        },
        "warm_user_cold_movie": {
            "description": "User represented, relevant movie absent from training pairs",
            "recalls": [],
            "ndcgs": [],
            "users": 0,
            "relevant_movies": 0
        },
        "cold_user": {
            "description": "User absent from training pairs",
            "recalls": [],
            "ndcgs": [],
            "users": 0,
            "relevant_movies": 0
        }
    }

    for number, user_id in enumerate(selected_users, start=1):

        user_idx = user_lookup[user_id]

        relevant_all = relevant_by_user[user_id]

        # Determine the coverage group for this user.

        if user_idx not in trained_users:

            group_name = "cold_user"

            relevant = relevant_all

        else:

            # Evaluate only warm relevant movies in this group.
            relevant = {
                movie_id
                for movie_id in relevant_all
                if movie_lookup[movie_id] in trained_movies
            }

            group_name = "warm_user_warm_movie"

            # Users with only cold relevant movies are evaluated
            # in the warm-user/cold-movie group instead.

            if not relevant:

                group_name = "warm_user_cold_movie"
                relevant = relevant_all

        group = groups[group_name]

        group["users"] += 1
        group["relevant_movies"] += len(relevant)

        # Use the exact two-tower scoring query.

        embedding = user_embeddings[user_idx].numpy()

        user_bias = user_biases[user_idx].item()

        query = np.concatenate([
            embedding,
            np.array([1.0, user_bias], dtype=np.float32)
        ]).reshape(1, -1)

        query = np.ascontiguousarray(
            query,
            dtype=np.float32
        )

        # Search the complete catalogue.

        _, positions = index.search(query, index.ntotal)

        seen = seen_by_user.get(user_id, set())

        recommendations = []

        for position in positions[0]:

            if position < 0:
                continue

            movie_id = int(movie_ids[position])

            if movie_id in seen:
                continue

            recommendations.append(movie_id)

            if len(recommendations) == TOP_K:
                break

        group["recalls"].append(
            recall_at_k(recommendations, relevant, TOP_K)
        )

        group["ndcgs"].append(
            ndcg_at_k(recommendations, relevant, TOP_K)
        )

        if number % 25 == 0:
            print(f"Evaluated {number}/{len(selected_users)} users")

    # --------------------------------------------------
    # STEP 7: CALCULATE METRICS
    # --------------------------------------------------

    print("\nStep 7: Calculating metrics...")

    results = {}

    for name, group in groups.items():

        if group["users"] == 0:
            results[name] = {
                "description": group["description"],
                "users_evaluated": 0,
                "relevant_movies": 0,
                "recall_at_10": None,
                "ndcg_at_10": None
            }
            continue

        results[name] = {
            "description": group["description"],
            "users_evaluated": group["users"],
            "relevant_movies": group["relevant_movies"],
            "recall_at_10": float(np.mean(group["recalls"])),
            "ndcg_at_10": float(np.mean(group["ndcgs"]))
        }

    metrics = {
        "model": "two_tower",
        "evaluation_dataset": "MovieLens 25M validation",
        "relevance_rule": "rating >= 4.0",
        "random_seed": RANDOM_SEED,
        "users_sampled": len(selected_users),
        "top_k": TOP_K,
        "evaluation_method": "full_catalogue_faiss_retrieval",
        "seen_training_movies_excluded": True,
        "coverage_definition": "Embedding represented in two_tower train_pairs.parquet",
        "coverage_groups": results
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # --------------------------------------------------
    # STEP 8: DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("COVERAGE-WISE EVALUATION RESULTS")
    print("=" * 70)

    for name, result in results.items():

        print(f"\n{name.upper()}")
        print("-" * 55)

        print(f"Users: {result['users_evaluated']}")
        print(f"Relevant movies: {result['relevant_movies']}")

        print(f"Recall@10: {result['recall_at_10']}")
        print(f"NDCG@10:   {result['ndcg_at_10']}")

    print("\nMetrics saved to:")
    print(METRICS_PATH)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()

   