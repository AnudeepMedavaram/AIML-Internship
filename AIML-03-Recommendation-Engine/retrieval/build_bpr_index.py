
from pathlib import Path
import json

import numpy as np
import torch
import faiss


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_FILE = (
    ROOT / "models" / "output" / "two_tower_bpr_model.pt"
)

OUTPUT_DIR = ROOT / "retrieval" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = OUTPUT_DIR / "bpr_movie_index.faiss"
MOVIE_IDS_FILE = OUTPUT_DIR / "bpr_movie_ids.npy"
METADATA_FILE = OUTPUT_DIR / "bpr_index_metadata.json"


# --------------------------------------------------
# Validate checkpoint
# --------------------------------------------------

if not CHECKPOINT_FILE.exists():
    raise FileNotFoundError(
        f"BPR checkpoint not found: {CHECKPOINT_FILE}"
    )

print("=" * 60)
print("BPR FAISS INDEX BUILDER")
print("=" * 60)

print("\nStep 1: Loading BPR checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location="cpu",
    weights_only=True,
)

num_users = int(checkpoint["num_users"])
num_movies = int(checkpoint["num_movies"])
embedding_dim = int(checkpoint["embedding_dim"])

state = checkpoint["model_state_dict"]

print(f"Users: {num_users:,}")
print(f"Movies: {num_movies:,}")
print(f"Embedding dimension: {embedding_dim}")


# --------------------------------------------------
# Extract learned embeddings
# --------------------------------------------------

print("\nStep 2: Extracting movie embeddings...")

movie_embeddings = (
    state["movie_embedding.weight"]
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)

movie_bias = (
    state["movie_bias.weight"]
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
    .reshape(-1, 1)
)

if movie_embeddings.shape != (num_movies, embedding_dim):
    raise ValueError("Unexpected movie embedding dimensions.")

if movie_bias.shape != (num_movies, 1):
    raise ValueError("Unexpected movie bias dimensions.")

# Reproduce the BPR scoring formula using inner product.
movie_vectors = np.concatenate(
    [
        movie_embeddings / np.sqrt(embedding_dim),
        movie_bias,
        np.ones((num_movies, 1), dtype=np.float32),
    ],
    axis=1,
)

print(f"Movie vectors shape: {movie_vectors.shape}")


# --------------------------------------------------
# Build FAISS index
# --------------------------------------------------

print("\nStep 3: Building FAISS inner-product index...")

index = faiss.IndexFlatIP(embedding_dim + 2)

index.add(movie_vectors)

print(f"Indexed movies: {index.ntotal:,}")


# --------------------------------------------------
# Save index and IDs
# --------------------------------------------------

print("\nStep 4: Saving index...")

movie_ids = np.arange(num_movies, dtype=np.int64)

faiss.write_index(index, str(INDEX_FILE))
np.save(MOVIE_IDS_FILE, movie_ids)

metadata = {
    "model_type": "BPR Matrix Factorization",
    "num_users": num_users,
    "num_movies": num_movies,
    "embedding_dim": embedding_dim,
    "index_type": "IndexFlatIP",
    "vector_dimension": embedding_dim + 2,
    "scoring_formula": (
        "dot(user_embedding, movie_embedding) / sqrt(embedding_dim)"
        " + user_bias + movie_bias"
    ),
}

with open(METADATA_FILE, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)


# --------------------------------------------------
# Verify saved index
# --------------------------------------------------

print("\nStep 5: Verifying saved index...")

loaded_index = faiss.read_index(str(INDEX_FILE))
loaded_movie_ids = np.load(MOVIE_IDS_FILE)

assert loaded_index.ntotal == num_movies
assert loaded_movie_ids.shape[0] == num_movies
assert loaded_index.d == embedding_dim + 2

print("Index verification successful.")

print("\n" + "=" * 60)
print("BPR FAISS INDEX BUILD COMPLETED")
print("=" * 60)

print(f"Index: {INDEX_FILE}")
print(f"Movie IDs: {MOVIE_IDS_FILE}")
print(f"Metadata: {METADATA_FILE}")

print("=" * 60)
