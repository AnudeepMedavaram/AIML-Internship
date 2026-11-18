
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import json
from pathlib import Path

# ---------------------------------------
# 1. Paths
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RATINGS_FILE = (
    BASE_DIR / "data" / "raw" / "ml-25m" / "ratings.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 250_000

# ---------------------------------------
# 2. Find approximate chronological
#    80% and 90% timestamp cutoffs
# ---------------------------------------

print("Step 1: Finding chronological cutoffs...")

timestamp_samples = []

for chunk in pd.read_csv(
    RATINGS_FILE,
    usecols=["timestamp"],
    chunksize=CHUNK_SIZE
):

    sample = chunk["timestamp"].sample(
        frac=0.01,
        random_state=42
    )

    timestamp_samples.append(sample)

sampled_timestamps = pd.concat(
    timestamp_samples,
    ignore_index=True
)

train_cutoff = int(
    sampled_timestamps.quantile(0.80)
)

validation_cutoff = int(
    sampled_timestamps.quantile(0.90)
)

print("Training cutoff:", train_cutoff)
print("Validation cutoff:", validation_cutoff)

# ---------------------------------------
# 3. Prepare Parquet output files
# ---------------------------------------

train_path = OUTPUT_DIR / "train.parquet"
validation_path = OUTPUT_DIR / "validation.parquet"
test_path = OUTPUT_DIR / "test.parquet"

# Remove previous outputs to avoid mixing runs
for path in [train_path, validation_path, test_path]:
    if path.exists():
        path.unlink()

writers = {
    "train": None,
    "validation": None,
    "test": None
}

counts = {
    "train": 0,
    "validation": 0,
    "test": 0
}

# ---------------------------------------
# 4. Split data chronologically
# ---------------------------------------

print("\nStep 2: Creating chronological splits...")

for chunk_number, chunk in enumerate(
    pd.read_csv(RATINGS_FILE, chunksize=CHUNK_SIZE),
    start=1
):

    train = chunk[
        chunk["timestamp"] <= train_cutoff
    ]

    validation = chunk[
        (chunk["timestamp"] > train_cutoff)
        & (chunk["timestamp"] <= validation_cutoff)
    ]

    test = chunk[
        chunk["timestamp"] > validation_cutoff
    ]

    split_frames = {
        "train": train,
        "validation": validation,
        "test": test
    }

    for split_name, frame in split_frames.items():

        if frame.empty:
            continue

        frame = frame.copy()

        table = pa.Table.from_pandas(
            frame,
            preserve_index=False
        )

        if writers[split_name] is None:

            output_path = {
                "train": train_path,
                "validation": validation_path,
                "test": test_path
            }[split_name]

            writers[split_name] = pq.ParquetWriter(
                output_path,
                table.schema,
                compression="snappy"
            )

        writers[split_name].write_table(table)

        counts[split_name] += len(frame)

    print(
        f"Processed chunk {chunk_number} | "
        f"Train: {counts['train']:,} | "
        f"Validation: {counts['validation']:,} | "
        f"Test: {counts['test']:,}"
    )

# ---------------------------------------
# 5. Close Parquet writers
# ---------------------------------------

for writer in writers.values():
    if writer is not None:
        writer.close()

# ---------------------------------------
# 6. Verify chronological ordering
# ---------------------------------------

train_min = None
train_max = None
validation_min = None
validation_max = None
test_min = None
test_max = None

for split_name, path in [
    ("train", train_path),
    ("validation", validation_path),
    ("test", test_path)
]:

    data = pq.read_table(
        path,
        columns=["timestamp"]
    ).to_pandas()

    min_time = int(data["timestamp"].min())
    max_time = int(data["timestamp"].max())

    if split_name == "train":
        train_min, train_max = min_time, max_time

    elif split_name == "validation":
        validation_min, validation_max = min_time, max_time

    else:
        test_min, test_max = min_time, max_time

# ---------------------------------------
# 7. Check split correctness
# ---------------------------------------

assert counts["train"] > 0
assert counts["validation"] > 0
assert counts["test"] > 0

assert train_max <= validation_min
assert validation_max <= test_min

assert sum(counts.values()) == 25_000_095

# ---------------------------------------
# 8. Save summary
# ---------------------------------------

summary = {
    "split_method": "chronological",
    "train_cutoff": train_cutoff,
    "validation_cutoff": validation_cutoff,
    "train_rows": counts["train"],
    "validation_rows": counts["validation"],
    "test_rows": counts["test"],
    "total_rows": sum(counts.values()),
    "train_date_range": [
        pd.to_datetime(train_min, unit="s").isoformat(),
        pd.to_datetime(train_max, unit="s").isoformat()
    ],
    "validation_date_range": [
        pd.to_datetime(validation_min, unit="s").isoformat(),
        pd.to_datetime(validation_max, unit="s").isoformat()
    ],
    "test_date_range": [
        pd.to_datetime(test_min, unit="s").isoformat(),
        pd.to_datetime(test_max, unit="s").isoformat()
    ],
    "chronological_order_verified": True
}

with open(
    OUTPUT_DIR / "split_summary.json",
    "w"
) as file:

    json.dump(summary, file, indent=4)

print("\n========== SPLIT SUMMARY ==========")

print(json.dumps(summary, indent=4))

print("\nChronological splitting completed successfully!")
