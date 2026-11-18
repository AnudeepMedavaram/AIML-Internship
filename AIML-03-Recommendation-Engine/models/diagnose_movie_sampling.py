
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "processed" / "train.parquet"

con = duckdb.connect()
con.execute("SET memory_limit = '4GB'")
con.execute("SET threads = 2")

query = f"""
WITH ranked AS (
    SELECT
        userId,
        movieId,
        CASE
            WHEN rating >= 4.0 THEN 'positive'
            ELSE 'negative'
        END AS category,
        ROW_NUMBER() OVER (
            PARTITION BY
                CASE
                    WHEN rating >= 4.0 THEN 'positive'
                    ELSE 'negative'
                END,
                movieId
            ORDER BY hash(userId, movieId)
        ) AS movie_rank
    FROM read_parquet('{TRAIN_FILE.as_posix()}')
    WHERE rating >= 4.0 OR rating <= 2.5
)
SELECT
    category,
    COUNT(*) FILTER (WHERE movie_rank <= 10) AS rows_cap_10,
    COUNT(DISTINCT userId) FILTER (WHERE movie_rank <= 10) AS users_cap_10,
    COUNT(DISTINCT movieId) FILTER (WHERE movie_rank <= 10) AS movies_cap_10,

    COUNT(*) FILTER (WHERE movie_rank <= 20) AS rows_cap_20,
    COUNT(DISTINCT userId) FILTER (WHERE movie_rank <= 20) AS users_cap_20,
    COUNT(DISTINCT movieId) FILTER (WHERE movie_rank <= 20) AS movies_cap_20,

    COUNT(*) FILTER (WHERE movie_rank <= 30) AS rows_cap_30,
    COUNT(DISTINCT userId) FILTER (WHERE movie_rank <= 30) AS users_cap_30,
    COUNT(DISTINCT movieId) FILTER (WHERE movie_rank <= 30) AS movies_cap_30

FROM ranked
GROUP BY category
ORDER BY category
"""

print("MOVIE-BALANCED SAMPLING COMPARISON")
print("=" * 90)
print(con.execute(query).df().to_string(index=False))

con.close()
