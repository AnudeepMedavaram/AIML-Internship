
from pathlib import Path
import json
import duckdb

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw" / "ml-25m"
FEATURE_DIR = ROOT / "features" / "output"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = (PROCESSED_DIR / "train.parquet").as_posix()
MOVIES_FILE = (RAW_DIR / "movies.csv").as_posix()

# Keep resource usage reasonable on a 16 GB laptop.
con = duckdb.connect()
con.execute("SET memory_limit = '6GB'")
con.execute("SET threads = 4")

print("=" * 55)
print("MOVIELENS FEATURE ENGINEERING")
print("=" * 55)

# --------------------------------------------------
# STEP 1: LOAD TRAINING INTERACTIONS
# --------------------------------------------------

print("\nStep 1: Loading training interactions...")

con.execute(f"""
    CREATE OR REPLACE TABLE ratings AS
    SELECT
        CAST(userId AS BIGINT) AS userId,
        CAST(movieId AS BIGINT) AS movieId,
        CAST(rating AS FLOAT) AS rating,
        CAST(timestamp AS BIGINT) AS timestamp
    FROM read_parquet('{TRAIN_FILE}')
""")

rating_count = con.execute(
    "SELECT COUNT(*) FROM ratings"
).fetchone()[0]

print(f"Training interactions loaded: {rating_count:,}")

# --------------------------------------------------
# STEP 2: USER FEATURES
# --------------------------------------------------

print("\nStep 2: Building user features...")

user_output = (FEATURE_DIR / "user_features.parquet").as_posix()

con.execute(f"""
    COPY (
        SELECT
            userId,
            COUNT(*) AS rating_count,
            AVG(rating) AS average_rating,
            STDDEV_SAMP(rating) AS rating_stddev
        FROM ratings
        GROUP BY userId
    )
    TO '{user_output}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

user_count = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{user_output}')
""").fetchone()[0]

print(f"User features created: {user_count:,}")

# --------------------------------------------------
# STEP 3: MOVIE METADATA
# --------------------------------------------------

print("\nStep 3: Loading movie metadata...")

con.execute(f"""
    CREATE OR REPLACE TABLE movies AS
    SELECT
        CAST(movieId AS BIGINT) AS movieId,
        title,
        genres,
        TRY_CAST(
            NULLIF(
                regexp_extract(title, '\\\\(([0-9]{{4}})\\\\)$', 1),
                ''
            ) AS INTEGER
        ) AS release_year
    FROM read_csv_auto('{MOVIES_FILE}', header = TRUE)
""")

catalogue_count = con.execute(
    "SELECT COUNT(*) FROM movies"
).fetchone()[0]

print(f"Movie catalogue loaded: {catalogue_count:,}")

# --------------------------------------------------
# STEP 4: MOVIE FEATURES
# --------------------------------------------------

print("\nStep 4: Building movie features...")

movie_output = (FEATURE_DIR / "movie_features.parquet").as_posix()

con.execute(f"""
    COPY (
        SELECT
            m.movieId,
            m.title,
            m.genres,
            m.release_year,
            COALESCE(r.rating_count, 0) AS rating_count,
            COALESCE(r.average_rating, 0.0) AS average_rating
        FROM movies m
        LEFT JOIN (
            SELECT
                movieId,
                COUNT(*) AS rating_count,
                AVG(rating) AS average_rating
            FROM ratings
            GROUP BY movieId
        ) r
        ON m.movieId = r.movieId
    )
    TO '{movie_output}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

movie_count = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{movie_output}')
""").fetchone()[0]

print(f"Movie features created: {movie_count:,}")

# --------------------------------------------------
# STEP 5: POPULAR MOVIES
# --------------------------------------------------

print("\nStep 5: Calculating popular movies...")

popular_output = (FEATURE_DIR / "popular_movies.parquet").as_posix()

con.execute(f"""
    COPY (
        SELECT
            m.movieId,
            m.title,
            m.genres,
            COUNT(r.rating) AS rating_count,
            AVG(r.rating) AS average_rating
        FROM movies m
        INNER JOIN ratings r
            ON m.movieId = r.movieId
        GROUP BY
            m.movieId,
            m.title,
            m.genres
        ORDER BY
            rating_count DESC,
            average_rating DESC
        LIMIT 1000
    )
    TO '{popular_output}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

popular_count = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{popular_output}')
""").fetchone()[0]

print(f"Popular movies saved: {popular_count:,}")

# --------------------------------------------------
# STEP 6: USER ID MAPPING
# --------------------------------------------------

print("\nStep 6: Creating user ID mapping...")

user_map_output = (FEATURE_DIR / "user_id_map.parquet").as_posix()

con.execute(f"""
    COPY (
        SELECT
            userId,
            ROW_NUMBER() OVER (ORDER BY userId) - 1 AS user_idx
        FROM (
            SELECT DISTINCT userId
            FROM ratings
        )
    )
    TO '{user_map_output}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

# --------------------------------------------------
# STEP 7: MOVIE ID MAPPING
# --------------------------------------------------

print("\nStep 7: Creating movie ID mapping...")

movie_map_output = (FEATURE_DIR / "movie_id_map.parquet").as_posix()

con.execute(f"""
    COPY (
        SELECT
            movieId,
            ROW_NUMBER() OVER (ORDER BY movieId) - 1 AS movie_idx
        FROM movies
    )
    TO '{movie_map_output}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

# --------------------------------------------------
# STEP 8: VALIDATION
# --------------------------------------------------

print("\nStep 8: Validating generated features...")

outputs = {
    "user_features": user_output,
    "movie_features": movie_output,
    "popular_movies": popular_output,
    "user_id_map": user_map_output,
    "movie_id_map": movie_map_output,
}

summary = {
    "training_interactions": rating_count,
    "users_with_training_interactions": user_count,
    "movies_in_catalogue": catalogue_count,
    "movie_features": movie_count,
    "popular_movies": popular_count,
    "files": {}
}

for name, path in outputs.items():
    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}')"
    ).fetchone()[0]

    summary["files"][name] = {
        "rows": count,
        "file": str(Path(path).relative_to(ROOT)).replace("\\", "/")
    }

    print(f"{name}: {count:,} rows")

# Check that ID mappings contain unique IDs and indices.
checks = [
    ("user_id_map", user_map_output, "userId", "user_idx"),
    ("movie_id_map", movie_map_output, "movieId", "movie_idx"),
]

for name, path, id_col, idx_col in checks:
    duplicate_ids = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT {id_col}
            FROM read_parquet('{path}')
            GROUP BY {id_col}
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    duplicate_indices = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT {idx_col}
            FROM read_parquet('{path}')
            GROUP BY {idx_col}
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    assert duplicate_ids == 0, f"Duplicate IDs in {name}"
    assert duplicate_indices == 0, f"Duplicate indices in {name}"

summary_path = FEATURE_DIR / "feature_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

con.close()

print("\n" + "=" * 55)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 55)
print(json.dumps(summary, indent=4))
print(f"\nSummary saved to: {summary_path}")
