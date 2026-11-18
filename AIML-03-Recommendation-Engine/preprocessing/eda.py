
import pandas as pd
import json
from pathlib import Path
from collections import Counter

# --------------------------------
# 1. File paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "raw" / "ml-25m"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_FILE = DATA_DIR / "movies.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------
# 2. Initialize statistics
# --------------------------------

total_ratings = 0
rating_sum = 0
rating_distribution = Counter()

unique_users = set()
unique_movies = set()

user_activity = Counter()
movie_popularity = Counter()

min_timestamp = None
max_timestamp = None

sample_frames = []

CHUNK_SIZE = 250_000

print("Starting MovieLens EDA...")
print("Processing ratings in chunks.")

# --------------------------------
# 3. Read ratings in chunks
# --------------------------------

for chunk_number, chunk in enumerate(
    pd.read_csv(RATINGS_FILE, chunksize=CHUNK_SIZE),
    start=1
):

    total_ratings += len(chunk)

    rating_sum += chunk["rating"].sum()

    rating_distribution.update(
        chunk["rating"].value_counts().to_dict()
    )

    unique_users.update(chunk["userId"].unique())
    unique_movies.update(chunk["movieId"].unique())

    user_activity.update(
        chunk["userId"].value_counts().to_dict()
    )

    movie_popularity.update(
        chunk["movieId"].value_counts().to_dict()
    )

    chunk_min = chunk["timestamp"].min()
    chunk_max = chunk["timestamp"].max()

    if min_timestamp is None or chunk_min < min_timestamp:
        min_timestamp = chunk_min

    if max_timestamp is None or chunk_max > max_timestamp:
        max_timestamp = chunk_max

    # Keep a small sample for later analysis
    if len(sample_frames) < 2:
        sample_frames.append(chunk.head(1000))

    print(
        f"Processed chunk {chunk_number} | "
        f"Total ratings: {total_ratings:,}"
    )

# --------------------------------
# 4. Calculate statistics
# --------------------------------

average_rating = rating_sum / total_ratings

start_date = pd.to_datetime(
    min_timestamp, unit="s"
).strftime("%Y-%m-%d")

end_date = pd.to_datetime(
    max_timestamp, unit="s"
).strftime("%Y-%m-%d")

summary = {
    "total_ratings": total_ratings,
    "unique_users": len(unique_users),
    "unique_movies": len(unique_movies),
    "average_rating": round(average_rating, 3),
    "rating_period_start": start_date,
    "rating_period_end": end_date
}

# --------------------------------
# 5. Save rating distribution
# --------------------------------

rating_df = pd.DataFrame(
    rating_distribution.items(),
    columns=["rating", "count"]
).sort_values("rating")

rating_df.to_csv(
    OUTPUT_DIR / "rating_distribution.csv",
    index=False
)

# --------------------------------
# 6. Save top 20 users
# --------------------------------

top_users = pd.DataFrame(
    user_activity.most_common(20),
    columns=["userId", "rating_count"]
)

top_users.to_csv(
    OUTPUT_DIR / "top_users.csv",
    index=False
)

# --------------------------------
# 7. Save top 20 movies
# --------------------------------

top_movies = pd.DataFrame(
    movie_popularity.most_common(20),
    columns=["movieId", "rating_count"]
)

top_movies = top_movies.merge(
    pd.read_csv(MOVIES_FILE),
    on="movieId",
    how="left"
)

top_movies.to_csv(
    OUTPUT_DIR / "top_movies.csv",
    index=False
)

# --------------------------------
# 8. Save sample data
# --------------------------------

sample_df = pd.concat(
    sample_frames,
    ignore_index=True
)

sample_df.to_csv(
    OUTPUT_DIR / "ratings_sample.csv",
    index=False
)

# --------------------------------
# 9. Save summary
# --------------------------------

with open(OUTPUT_DIR / "eda_summary.json", "w") as file:
    json.dump(summary, file, indent=4)

# --------------------------------
# 10. Display results
# --------------------------------

print("\n========== MOVIELENS EDA ==========")

for key, value in summary.items():
    print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")

print("\nTop 10 popular movies:")

print(
    top_movies[["movieId", "title", "rating_count"]]
    .head(10)
    .to_string(index=False)
)

print("\nEDA completed successfully!")
print("Results saved inside data/processed/")
