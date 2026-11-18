
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "processed" / "train.parquet"

con = duckdb.connect()

con.execute("SET memory_limit = '4GB'")
con.execute("SET threads = 2")

print("=" * 65)
print("TRAINING DATA DISTRIBUTION")
print("=" * 65)

query = f"""
SELECT
    CASE
        WHEN rating >= 4.0 THEN 'positive'
        WHEN rating <= 2.5 THEN 'negative'
        ELSE 'neutral'
    END AS category,
    COUNT(*) AS interactions,
    COUNT(DISTINCT userId) AS users,
    COUNT(DISTINCT movieId) AS movies
FROM read_parquet('{TRAIN_FILE.as_posix()}')
GROUP BY 1
ORDER BY 1
"""

distribution = con.execute(query).df()

print(distribution.to_string(index=False))

print("\n" + "=" * 65)
print("MOVIE INTERACTION DISTRIBUTION")
print("=" * 65)

query = f"""
SELECT
    COUNT(*) AS total_movies,
    COUNT(*) FILTER (WHERE n >= 5) AS movies_5_plus,
    COUNT(*) FILTER (WHERE n >= 10) AS movies_10_plus,
    COUNT(*) FILTER (WHERE n >= 20) AS movies_20_plus,
    COUNT(*) FILTER (WHERE n >= 50) AS movies_50_plus
FROM (
    SELECT
        movieId,
        COUNT(*) AS n
    FROM read_parquet('{TRAIN_FILE.as_posix()}')
    WHERE rating >= 4.0 OR rating <= 2.5
    GROUP BY movieId
)
"""

movie_distribution = con.execute(query).df()

print(movie_distribution.to_string(index=False))

print("\nDiagnostic completed.")

con.close()
