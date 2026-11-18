
from pathlib import Path
import json

import faiss
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = ROOT / "models/output/two_tower_model.pt"
MOVIE_MAP_PATH = ROOT / "features/output/movie_id_map.parquet"

OUTPUT_DIR = ROOT / "retrieval/output"
INDEX_PATH = OUTPUT_DIR / "movie_index.faiss"
MOVIE_IDS_PATH = OUTPUT_DIR / "movie_ids.npy"
METADATA_PATH = OUTPUT_DIR / "index_metadata.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(2)


def main():

    print("=" * 60)
    print("CORRECTED FAISS INDEX BUILDER")
    print("=" * 60)

    # 1. Load checkpoint
    print("\nStep 1: Loading trained model...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    state = checkpoint["model_state_dict"]

    num_movies = checkpoint["num_movies"]
    embedding_dim = checkpoint["embedding_dim"]

    # 2. Extract raw movie embeddings and biases
    print("\nStep 2: Extracting movie embeddings...")

    movie_embeddings = (
        state["movie_embedding.weight"]
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    movie_biases = (
        state["movie_bias.weight"]
        .cpu()
        .numpy()
        .astype(np.float32)
        .reshape(-1, 1)
    )

    # Training uses:
    # dot(user_embedding, movie_embedding) / sqrt(dim)
    #
    # Therefore divide movie embeddings by sqrt(dim).
    # Do NOT L2-normalize the embeddings.

    movie_embeddings = movie_embeddings / np.sqrt(embedding_dim)

    # 3. Validate movie ID mapping
    print("\nStep 3: Loading movie ID mapping...")

    movie_map = pd.read_parquet(MOVIE_MAP_PATH)

    movie_map = movie_map.sort_values(
        "movie_idx"
    ).reset_index(drop=True)

    movie_indices = movie_map["movie_idx"].to_numpy(
        dtype=np.int64
    )

    movie_ids = movie_map["movieId"].to_numpy(
        dtype=np.int64
    )

    if len(movie_map) != num_movies:
        raise ValueError("Movie count does not match checkpoint.")

    if not np.array_equal(
        movie_indices,
        np.arange(num_movies)
    ):
        raise ValueError("Movie indices are not consecutive.")

    # 4. Construct augmented vectors
    print("\nStep 4: Constructing exact model-score vectors...")

    # Movie vector:
    # [movie_embedding / sqrt(dim), movie_bias, 1]
    #
    # User query:
    # [user_embedding, 1, user_bias]
    #
    # Inner product reproduces the training model score.

    ones = np.ones(
        (num_movies, 1),
        dtype=np.float32
    )

    movie_vectors = np.concatenate(
        [
            movie_embeddings,
            movie_biases,
            ones
        ],
        axis=1
    )

    movie_vectors = np.ascontiguousarray(
        movie_vectors,
        dtype=np.float32
    )

    # 5. Build index
    print("\nStep 5: Building exact inner-product index...")

    index = faiss.IndexFlatIP(embedding_dim + 2)

    index.add(movie_vectors)

    print("Indexed movies:", index.ntotal)

    # 6. Save index and mappings
    print("\nStep 6: Saving files...")

    faiss.write_index(index, str(INDEX_PATH))
    np.save(MOVIE_IDS_PATH, movie_ids)

    metadata = {
        "index_type": "IndexFlatIP",
        "scoring": "raw_dot_product_divided_by_sqrt_embedding_dim_plus_biases",
        "embedding_normalization": False,
        "num_movies": int(num_movies),
        "embedding_dim": int(embedding_dim),
        "index_dimension": int(embedding_dim + 2)
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # 7. Verify
    print("\nStep 7: Verifying index...")

    loaded_index = faiss.read_index(str(INDEX_PATH))
    loaded_ids = np.load(MOVIE_IDS_PATH)

    assert loaded_index.ntotal == num_movies
    assert len(loaded_ids) == num_movies

    print("\n" + "=" * 60)
    print("CORRECTED FAISS INDEX COMPLETED")
    print("=" * 60)

    print("Movies indexed:", loaded_index.ntotal)
    print("Index dimension:", embedding_dim + 2)
    print("Scoring: exact two-tower model scoring")
    print("Index:", INDEX_PATH)


if __name__ == "__main__":
    main()

