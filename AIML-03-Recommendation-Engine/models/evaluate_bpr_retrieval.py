
from pathlib import Path
import json

import duckdb
import faiss
import numpy as np
import pandas as pd
import torch


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_FILE = ROOT / "models/output/two_tower_bpr_model.pt"

TRAIN_FILE = ROOT / "data/processed/train.parquet"
VALIDATION_FILE = ROOT / "data/processed/validation.parquet"

USER_MAP_FILE = ROOT / "features/output/user_id_map.parquet"
MOVIE_MAP_FILE = ROOT / "features/output/movie_id_map.parquet"

INDEX_FILE = ROOT / "retrieval/output/bpr_movie_index.faiss"
MOVIE_IDS_FILE = ROOT / "retrieval/output/bpr_movie_ids.npy"

OUTPUT_FILE = ROOT / "models/output/bpr_retrieval_metrics.json"

K = 10
NUM_EVAL_USERS = 200
SEED = 42
BATCH_SIZE = 32


# --------------------------------------------------
# Validate files
# --------------------------------------------------

required_files = [
    CHECKPOINT_FILE,
    TRAIN_FILE,
    VALIDATION_FILE,
    USER_MAP_FILE,
    MOVIE_MAP_FILE,
    INDEX_FILE,
    MOVIE_IDS_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")


print("=" * 65)
print("BPR RETRIEVAL EVALUATION")
print("=" * 65)


# --------------------------------------------------
# Load checkpoint and index
# --------------------------------------------------

print("\nStep 1: Loading BPR model and FAISS index...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location="cpu",
    weights_only=True,
)

state = checkpoint["model_state_dict"]

user_embeddings = (
    state["user_embedding.weight"]
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)

user_bias = (
    state["user_bias.weight"]
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
    .reshape(-1)
)

index = faiss.read_index(str(INDEX_FILE))
movie_ids = np.load(MOVIE_IDS_FILE)

num_users = int(checkpoint["num_users"])
num_movies = int(checkpoint["num_movies"])
embedding_dim = int(checkpoint["embedding_dim"])

assert index.ntotal == num_movies
assert index.d == embedding_dim + 2
assert len(movie_ids) == num_movies

print(f"Users: {num_users:,}")
print(f"Movies: {num_movies:,}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Indexed movies: {index.ntotal:,}")


# --------------------------------------------------
# Load ID maps
# --------------------------------------------------

print("\nStep 2: Loading ID maps...")

user_map = pd.read_parquet(USER_MAP_FILE)
movie_map = pd.read_parquet(MOVIE_MAP_FILE)

user_lookup = dict(
    zip(user_map["userId"], user_map["user_idx"])
)

movie_lookup = dict(
    zip(movie_map["movieId"], movie_map["movie_idx"])
)

movie_idx_to_id = np.zeros(num_movies, dtype=np.int64)

for row in movie_map.itertuples(index=False):
    movie_idx_to_id[int(row.movie_idx)] = int(row.movieId)

del user_map, movie_map

print(f"Mapped users: {len(user_lookup):,}")
print(f"Mapped movies: {len(movie_lookup):,}")


# --------------------------------------------------
# Load training history
# --------------------------------------------------

print("\nStep 3: Loading training history...")

connection = duckdb.connect()

train_path = TRAIN_FILE.as_posix()
validation_path = VALIDATION_FILE.as_posix()

train_history = connection.execute(
    f"""
    SELECT DISTINCT userId, movieId
    FROM read_parquet('{train_path}')
    """
).df()

train_history = train_history[
    train_history["userId"].isin(user_lookup)
    & train_history["movieId"].isin(movie_lookup)
]

seen_by_user = {}

for user_id, movie_id in train_history[
    ["userId", "movieId"]
].itertuples(index=False, name=None):

    user_idx = user_lookup[user_id]
    movie_idx = movie_lookup[movie_id]

    seen_by_user.setdefault(user_idx, set()).add(movie_idx)

del train_history

print(f"Users with training history: {len(seen_by_user):,}")


# --------------------------------------------------
# Load validation positives
# --------------------------------------------------

print("\nStep 4: Loading validation positives...")

validation = connection.execute(
    f"""
    SELECT userId, movieId
    FROM read_parquet('{validation_path}')
    WHERE rating >= 4.0
    """
).df()

validation["user_idx"] = validation["userId"].map(user_lookup)
validation["movie_idx"] = validation["movieId"].map(movie_lookup)

validation = validation.dropna(
    subset=["user_idx", "movie_idx"]
)

validation["user_idx"] = validation["user_idx"].astype(np.int64)
validation["movie_idx"] = validation["movie_idx"].astype(np.int64)

relevant_by_user = (
    validation.groupby("user_idx")["movie_idx"]
    .apply(set)
    .to_dict()
)

del validation
connection.close()

# Eligible users must have training history and relevant movies
eligible_users = sorted(
    set(seen_by_user).intersection(relevant_by_user)
)

eligible_users = [
    user_idx
    for user_idx in eligible_users
    if relevant_by_user[user_idx] - seen_by_user[user_idx]
]

print(f"Eligible validation users: {len(eligible_users):,}")

if not eligible_users:
    raise RuntimeError("No eligible users found.")

rng = np.random.default_rng(SEED)

sampled_users = rng.choice(
    eligible_users,
    size=min(NUM_EVAL_USERS, len(eligible_users)),
    replace=False,
)

print(f"Sampled evaluation users: {len(sampled_users):,}")


# --------------------------------------------------
# Evaluate recommendations
# --------------------------------------------------

print("\nStep 5: Computing Recall@10 and NDCG@10...")

recall_scores = []
ndcg_scores = []

for start in range(0, len(sampled_users), BATCH_SIZE):

    batch_users = sampled_users[start:start + BATCH_SIZE]

    query_vectors = np.concatenate(
        [
            user_embeddings[batch_users],
            np.ones((len(batch_users), 1), dtype=np.float32),
            user_bias[batch_users].reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)

    # Retrieve extra candidates to compensate for seen movies.
    candidate_count = min(
        num_movies,
        max(K + 500, 1000),
    )

    _, candidate_indices = index.search(
        query_vectors,
        candidate_count,
    )

    for row, user_idx in enumerate(batch_users):

        seen = seen_by_user[int(user_idx)]

        relevant = (
            relevant_by_user[int(user_idx)] - seen
        )

        ranked_movies = []

        for candidate_idx in candidate_indices[row]:

            if candidate_idx < 0:
                continue

            movie_idx = int(movie_ids[candidate_idx])

            if movie_idx not in seen:
                ranked_movies.append(movie_idx)

            if len(ranked_movies) == K:
                break

        if not ranked_movies:
            recall_scores.append(0.0)
            ndcg_scores.append(0.0)
            continue

        hits = [
            1 if movie in relevant else 0
            for movie in ranked_movies
        ]

        recall = sum(hits) / len(relevant)

        dcg = sum(
            hit / np.log2(position + 2)
            for position, hit in enumerate(hits)
        )

        ideal_hits = min(len(relevant), K)

        idcg = sum(
            1 / np.log2(position + 2)
            for position in range(ideal_hits)
        )

        ndcg = dcg / idcg if idcg > 0 else 0.0

        recall_scores.append(recall)
        ndcg_scores.append(ndcg)


# --------------------------------------------------
# Results
# --------------------------------------------------

metrics = {
    "model": "BPR Matrix Factorization",
    "evaluation_users": len(sampled_users),
    "catalogue_size": num_movies,
    "k": K,
    "recall_at_10": float(np.mean(recall_scores)),
    "ndcg_at_10": float(np.mean(ndcg_scores)),
    "seed": SEED,
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\n" + "=" * 65)
print("BPR RETRIEVAL EVALUATION RESULTS")
print("=" * 65)

print(f"Evaluation users: {metrics['evaluation_users']:,}")
print(f"Catalogue size: {metrics['catalogue_size']:,}")
print(f"Recall@10: {metrics['recall_at_10']:.6f}")
print(f"NDCG@10:   {metrics['ndcg_at_10']:.6f}")

print(f"\nMetrics saved: {OUTPUT_FILE}")

print("=" * 65)
