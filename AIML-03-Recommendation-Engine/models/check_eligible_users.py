
import duckdb

connection = duckdb.connect()

query = """
SELECT COUNT(*) AS eligible_users
FROM (
    SELECT userId
    FROM read_parquet('data/processed/train.parquet')
    GROUP BY userId
    HAVING
        COUNT(*) FILTER (WHERE rating >= 4.0) > 0
        AND
        COUNT(*) FILTER (WHERE rating <= 2.5) > 0
)
"""

result = connection.execute(query).fetchone()

print("=" * 50)
print("BPR TRAINING ELIGIBILITY")
print("=" * 50)
print(f"Users with both positive and negative interactions: {result[0]}")
print("=" * 50)

connection.close()
