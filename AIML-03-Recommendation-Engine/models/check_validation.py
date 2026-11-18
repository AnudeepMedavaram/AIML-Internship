
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]

con = duckdb.connect()
con.execute("SET memory_limit = '4GB'")
con.execute("SET threads = 4")

validation = (ROOT / "data/processed/validation.parquet").as_posix()
user_map = (ROOT / "features/output/user_id_map.parquet").as_posix()
movie_map = (ROOT / "features/output/movie_id_map.parquet").as_posix()

print("\n1. Original validation rating distribution")

distribution = con.execute(f"""
    SELECT
        CASE
            WHEN rating >= 4.0 THEN 'positive'
            WHEN rating <= 2.5 THEN 'negative'
            ELSE 'neutral'
        END AS category,
        COUNT(*) AS interactions
    FROM read_parquet('{validation}')
    GROUP BY category
    ORDER BY category
""").fetchdf()

print(distribution.to_string(index=False))

print("\n2. Validation interactions after ID mapping")

coverage = con.execute(f"""
    SELECT
        CASE
            WHEN r.rating >= 4.0 THEN 'positive'
            ELSE 'negative'
        END AS category,
        COUNT(*) AS original_interactions,
        COUNT(u.user_idx) AS known_users,
        COUNT(m.movie_idx) AS known_movies,
        COUNT(
            CASE
                WHEN u.user_idx IS NOT NULL
                 AND m.movie_idx IS NOT NULL
                THEN 1
            END
        ) AS usable_interactions
    FROM read_parquet('{validation}') r
    LEFT JOIN read_parquet('{user_map}') u
        ON r.userId = u.userId
    LEFT JOIN read_parquet('{movie_map}') m
        ON r.movieId = m.movieId
    WHERE r.rating >= 4.0 OR r.rating <= 2.5
    GROUP BY category
    ORDER BY category
""").fetchdf()

print(coverage.to_string(index=False))

print("\n3. Overall validation coverage")

total = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{validation}')
""").fetchone()[0]

usable = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{validation}') r
    INNER JOIN read_parquet('{user_map}') u
        ON r.userId = u.userId
    INNER JOIN read_parquet('{movie_map}') m
        ON r.movieId = m.movieId
    WHERE r.rating >= 4.0 OR r.rating <= 2.5
""").fetchone()[0]

print(f"Total validation interactions: {total:,}")
print(f"Usable positive/negative interactions: {usable:,}")
print(f"Coverage: {usable / total * 100:.2f}%")

con.close()
