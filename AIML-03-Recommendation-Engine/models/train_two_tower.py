
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "features" / "output"
MODEL_DIR = ROOT / "models" / "output"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = (PROCESSED / "train.parquet").as_posix()
VAL_FILE = (PROCESSED / "validation.parquet").as_posix()
USER_MAP = (FEATURES / "user_id_map.parquet").as_posix()
MOVIE_MAP = (FEATURES / "movie_id_map.parquet").as_posix()

DEVICE = torch.device("cpu")
EMBEDDING_DIM = 64
BATCH_SIZE = 8192
EPOCHS = 3
LEARNING_RATE = 0.003

# Movie-balanced training sample
MAX_PER_MOVIE_PER_CLASS = 30

# Keep validation sampling unchanged
VAL_PER_CLASS = 50_000

print("=" * 60)
print("TWO-TOWER RECOMMENDATION MODEL")
print("=" * 60)

# --------------------------------------------------
# STEP 1: CREATE TRAINING AND VALIDATION PAIRS
# --------------------------------------------------

con = duckdb.connect()
con.execute("SET memory_limit = '6GB'")
con.execute("SET threads = 4")


def create_train_pairs(source_file, output_file):
    """Sample up to 30 positive and negative interactions per movie."""

    output_file = output_file.as_posix()

    query = f"""
    COPY (
        WITH categorized AS (
            SELECT
                userId,
                movieId,
                CASE
                    WHEN rating >= 4.0 THEN 1
                    ELSE 0
                END AS label
            FROM read_parquet('{source_file}')
            WHERE rating >= 4.0 OR rating <= 2.5
        ),
        ranked AS (
            SELECT
                userId,
                movieId,
                label,
                ROW_NUMBER() OVER (
                    PARTITION BY movieId, label
                    ORDER BY hash(userId, movieId)
                ) AS movie_rank
            FROM categorized
        ),
        sampled AS (
            SELECT userId, movieId, label
            FROM ranked
            WHERE movie_rank <= {MAX_PER_MOVIE_PER_CLASS}
        )
        SELECT
            u.user_idx,
            m.movie_idx,
            s.label
        FROM sampled s
        INNER JOIN read_parquet('{USER_MAP}') u
            ON s.userId = u.userId
        INNER JOIN read_parquet('{MOVIE_MAP}') m
            ON s.movieId = m.movieId
    )
    TO '{output_file}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(query)

    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_file}')"
    ).fetchone()[0]

    print(f"Created {output_file}: {count:,} pairs")
    return output_file


def create_validation_pairs(source_file, output_file, per_class):
    """Keep the original globally balanced validation sampling."""

    output_file = output_file.as_posix()

    query = f"""
    COPY (
        WITH positive AS (
            SELECT userId, movieId, 1 AS label
            FROM read_parquet('{source_file}')
            WHERE rating >= 4.0
            QUALIFY ROW_NUMBER() OVER (
                ORDER BY hash(userId, movieId)
            ) <= {per_class}
        ),
        negative AS (
            SELECT userId, movieId, 0 AS label
            FROM read_parquet('{source_file}')
            WHERE rating <= 2.5
            QUALIFY ROW_NUMBER() OVER (
                ORDER BY hash(userId, movieId)
            ) <= {per_class}
        ),
        combined AS (
            SELECT * FROM positive
            UNION ALL
            SELECT * FROM negative
        )
        SELECT
            u.user_idx,
            m.movie_idx,
            c.label
        FROM combined c
        INNER JOIN read_parquet('{USER_MAP}') u
            ON c.userId = u.userId
        INNER JOIN read_parquet('{MOVIE_MAP}') m
            ON c.movieId = m.movieId
    )
    TO '{output_file}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(query)

    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_file}')"
    ).fetchone()[0]

    print(f"Created {output_file}: {count:,} pairs")
    return output_file


print("\nStep 1: Preparing interaction pairs...")

train_pairs_path = create_train_pairs(
    TRAIN_FILE,
    MODEL_DIR / "train_pairs.parquet"
)

val_pairs_path = create_validation_pairs(
    VAL_FILE,
    MODEL_DIR / "validation_pairs.parquet",
    VAL_PER_CLASS
)

con.close()

# --------------------------------------------------
# STEP 2: LOAD PAIRS
# --------------------------------------------------

print("\nStep 2: Loading training data...")

train_df = pd.read_parquet(train_pairs_path)
val_df = pd.read_parquet(val_pairs_path)

if train_df.empty or val_df.empty:
    raise ValueError("Training or validation pairs are empty.")

print("Training pairs:", len(train_df))
print("Validation pairs:", len(val_df))

print("\nTraining class distribution:")
print(train_df["label"].value_counts().sort_index())

print("Unique training users:", train_df["user_idx"].nunique())
print("Unique training movies:", train_df["movie_idx"].nunique())

train_users = torch.tensor(
    train_df["user_idx"].to_numpy(dtype=np.int64)
)
train_movies = torch.tensor(
    train_df["movie_idx"].to_numpy(dtype=np.int64)
)
train_labels = torch.tensor(
    train_df["label"].to_numpy(dtype=np.float32)
)

val_users = torch.tensor(
    val_df["user_idx"].to_numpy(dtype=np.int64)
)
val_movies = torch.tensor(
    val_df["movie_idx"].to_numpy(dtype=np.int64)
)
val_labels = val_df["label"].to_numpy(dtype=np.float32)

train_dataset = TensorDataset(
    train_users, train_movies, train_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# --------------------------------------------------
# STEP 3: DEFINE TWO-TOWER MODEL
# --------------------------------------------------

print("\nStep 3: Building two-tower architecture...")


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users, embedding_dim
        )

        self.movie_embedding = nn.Embedding(
            num_movies, embedding_dim
        )

        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.movie_embedding.weight, std=0.05)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_ids, movie_ids):
        user_vector = self.user_embedding(user_ids)
        movie_vector = self.movie_embedding(movie_ids)

        interaction = (
            user_vector * movie_vector
        ).sum(dim=1) / (user_vector.shape[1] ** 0.5)

        user_bias = self.user_bias(user_ids).squeeze(1)
        movie_bias = self.movie_bias(movie_ids).squeeze(1)

        return interaction + user_bias + movie_bias


num_users = len(pd.read_parquet(USER_MAP, columns=["user_idx"]))
num_movies = len(pd.read_parquet(MOVIE_MAP, columns=["movie_idx"]))

model = TwoTowerModel(
    num_users,
    num_movies,
    EMBEDDING_DIM
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.BCEWithLogitsLoss()

print("Users:", num_users)
print("Movies:", num_movies)
print("Embedding dimension:", EMBEDDING_DIM)
print("Device:", DEVICE)

# --------------------------------------------------
# STEP 4: TRAIN MODEL
# --------------------------------------------------

print("\nStep 4: Training...")

history = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_rows = 0

    for batch_users, batch_movies, batch_labels in train_loader:
        batch_users = batch_users.to(DEVICE)
        batch_movies = batch_movies.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(batch_users, batch_movies)
        loss = criterion(predictions, batch_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_labels)
        total_rows += len(batch_labels)

    average_loss = total_loss / total_rows

    history.append({
        "epoch": epoch + 1,
        "training_loss": average_loss
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Training loss: {average_loss:.5f}"
    )

# --------------------------------------------------
# STEP 5: VALIDATION
# --------------------------------------------------

print("\nStep 5: Evaluating on validation pairs...")

model.eval()

with torch.no_grad():
    validation_logits = []

    for start in range(0, len(val_users), BATCH_SIZE):
        end = start + BATCH_SIZE

        logits = model(
            val_users[start:end],
            val_movies[start:end]
        )

        validation_logits.append(logits.numpy())

    logits = np.concatenate(validation_logits)
    probabilities = 1.0 / (1.0 + np.exp(-logits))

predicted_labels = (probabilities >= 0.5).astype(int)

auc = roc_auc_score(val_labels, probabilities)
accuracy = accuracy_score(val_labels, predicted_labels)

print(f"Validation ROC-AUC: {auc:.5f}")
print(f"Validation accuracy: {accuracy:.5f}")

# --------------------------------------------------
# STEP 6: SAVE MODEL
# --------------------------------------------------

print("\nStep 6: Saving model...")

model_path = MODEL_DIR / "two_tower_model.pt"

torch.save({
    "model_state_dict": model.state_dict(),
    "num_users": num_users,
    "num_movies": num_movies,
    "embedding_dim": EMBEDDING_DIM
}, model_path)

metrics = {
    "model": "Two-Tower Retrieval Model",
    "sampling": "Movie-balanced, max 30 per movie per class",
    "embedding_dim": EMBEDDING_DIM,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "training_pairs": len(train_df),
    "training_users": int(train_df["user_idx"].nunique()),
    "training_movies": int(train_df["movie_idx"].nunique()),
    "validation_pairs": len(val_df),
    "validation_roc_auc": float(auc),
    "validation_accuracy": float(accuracy),
    "training_history": history,
    "model_file": "models/output/two_tower_model.pt"
}

metrics_path = MODEL_DIR / "training_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\n" + "=" * 60)
print("TWO-TOWER MODEL TRAINING COMPLETED")
print("=" * 60)
print(json.dumps(metrics, indent=4))
