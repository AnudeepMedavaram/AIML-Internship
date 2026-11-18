
from pathlib import Path
import duckdb

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "data" / "processed" / "train.parquet"
USER_MAP = ROOT / "features" / "output" / "user_id_map.parquet"
MOVIE_MAP = ROOT / "features" / "output" / "movie_id_map.parquet"

OUTPUT_DIR = ROOT / "models" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "bpr_train_pairs.parquet"

PAIRS_PER_USER = 5

# --------------------------------------------------
# Validate input files
# --------------------------------------------------

for file_path in [TRAIN_FILE, USER_MAP, MOVIE_MAP]:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

print("=" * 60)
print("BPR TRAINING DATA PREPARATION")
print("=" * 60)

connection = duckdb.connect()

train_path = TRAIN_FILE.as_posix()
user_map_path = USER_MAP.as_posix()
movie_map_path = MOVIE_MAP.as_posix()
output_path = OUTPUT_FILE.as_posix()

# --------------------------------------------------
# Create same-user positive-negative pairs
# --------------------------------------------------

query = f"""
WITH positive_ranked AS (
    SELECT
        r.userId,
        u.user_idx,
        m.movie_idx AS positive_movie_idx,
        ROW_NUMBER() OVER (
            PARTITION BY r.userId
            ORDER BY HASH(r.movieId)
        ) AS pair_number
    FROM read_parquet('{train_path}') r
    INNER JOIN read_parquet('{user_map_path}') u
        ON r.userId = u.userId
    INNER JOIN read_parquet('{movie_map_path}') m
        ON r.movieId = m.movieId
    WHERE r.rating >= 4.0
),

negative_ranked AS (
    SELECT
        r.userId,
        m.movie_idx AS negative_movie_idx,
        ROW_NUMBER() OVER (
            PARTITION BY r.userId
            ORDER BY HASH(r.movieId)
        ) AS pair_number
    FROM read_parquet('{train_path}') r
    INNER JOIN read_parquet('{user_map_path}') u
        ON r.userId = u.userId
    INNER JOIN read_parquet('{movie_map_path}') m
        ON r.movieId = m.movieId
    WHERE r.rating <= 2.5
)

SELECT
    p.user_idx,
    p.positive_movie_idx,
    n.negative_movie_idx
FROM positive_ranked p
INNER JOIN negative_ranked n
    ON p.userId = n.userId
    AND p.pair_number = n.pair_number
WHERE p.pair_number <= {PAIRS_PER_USER}
"""

print("\nGenerating user-specific pairs...")
print("This may take a few minutes.")

connection.execute(
    f"""
    COPY ({query})
    TO '{output_path}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """
)

# --------------------------------------------------
# Inspect output
# --------------------------------------------------

stats = connection.execute(
    f"""
    SELECT
        COUNT(*) AS total_pairs,
        COUNT(DISTINCT user_idx) AS unique_users,
        COUNT(DISTINCT positive_movie_idx) AS unique_positive_movies,
        COUNT(DISTINCT negative_movie_idx) AS unique_negative_movies
    FROM read_parquet('{output_path}')
    """
).fetchone()

print("\nBPR DATASET SUMMARY")
print("-" * 60)
print(f"Total training pairs:       {stats[0]:,}")
print(f"Unique users:               {stats[1]:,}")
print(f"Unique positive movies:     {stats[2]:,}")
print(f"Unique negative movies:     {stats[3]:,}")

print("\nSample pairs:")
sample = connection.execute(
    f"SELECT * FROM read_parquet('{output_path}') LIMIT 10"
).df()

print(sample.to_string(index=False))

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nBPR data preparation completed.")
print("=" * 60)

connection.close()
