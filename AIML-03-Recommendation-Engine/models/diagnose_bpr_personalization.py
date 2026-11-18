
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = ROOT / "models/output/two_tower_bpr_model.pt"
USER_MAP = ROOT / "features/output/user_id_map.parquet"

SEED = 42
NUM_USERS = 10
TOP_K = 10

print("=" * 65)
print("BPR PERSONALIZATION DIAGNOSTIC")
print("=" * 65)

# Load model
checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

state = checkpoint["model_state_dict"]

users = state["user_embedding.weight"].numpy()
movies = state["movie_embedding.weight"].numpy()

user_bias = state["user_bias.weight"].numpy().ravel()
movie_bias = state["movie_bias.weight"].numpy().ravel()

embedding_dim = checkpoint["embedding_dim"]

# Select users with training history
user_map = pd.read_parquet(USER_MAP)

rng = np.random.default_rng(SEED)

available_users = user_map["user_idx"].to_numpy()

selected_users = rng.choice(
    available_users,
    size=NUM_USERS,
    replace=False,
)

del user_map

# Calculate score components
personalized_scores = (
    users[selected_users] @ movies.T
) / np.sqrt(embedding_dim)

popularity_scores = np.broadcast_to(
    movie_bias,
    personalized_scores.shape,
)

total_scores = (
    personalized_scores
    + popularity_scores
    + user_bias[selected_users, None]
)

print("\n1. SCORE COMPONENT ANALYSIS")
print("-" * 65)

print(
    f"{'User':>8} | {'Personalized SD':>16} | "
    f"{'Movie Bias SD':>14} | {'Score SD':>10}"
)

for i, user_idx in enumerate(selected_users):
    print(
        f"{user_idx:8d} | "
        f"{personalized_scores[i].std():16.6f} | "
        f"{popularity_scores[i].std():14.6f} | "
        f"{total_scores[i].std():10.6f}"
    )

print("\n2. TOP-10 RECOMMENDATION OVERLAP")
print("-" * 65)

top_movies = []

for i in range(NUM_USERS):
    top_indices = np.argsort(
        total_scores[i]
    )[-TOP_K:][::-1]

    top_movies.append(set(top_indices.tolist()))

overlaps = []

for i in range(NUM_USERS):
    for j in range(i + 1, NUM_USERS):
        overlap = len(
            top_movies[i].intersection(top_movies[j])
        )
        overlaps.append(overlap / TOP_K)

print(f"Users compared: {NUM_USERS}")
print(f"Average top-10 overlap: {np.mean(overlaps):.4f}")
print(f"Maximum top-10 overlap: {np.max(overlaps):.4f}")
print(f"Minimum top-10 overlap: {np.min(overlaps):.4f}")

print("\n3. PERSONALIZED SCORE VARIATION")
print("-" * 65)

print(
    "Average personalized score SD:",
    round(personalized_scores.std(axis=1).mean(), 6),
)

print(
    "Average movie bias SD:",
    round(movie_bias.std(), 6),
)

print(
    "Average total score SD:",
    round(total_scores.std(axis=1).mean(), 6),
)

print("\n" + "=" * 65)
print("DIAGNOSTIC COMPLETED")
print("=" * 65)
