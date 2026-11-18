
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

PAIR_FILE = ROOT / "models" / "output" / "bpr_train_pairs.parquet"
USER_MAP = ROOT / "features" / "output" / "user_id_map.parquet"
MOVIE_MAP = ROOT / "features" / "output" / "movie_id_map.parquet"

OUTPUT_DIR = ROOT / "models" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = OUTPUT_DIR / "two_tower_bpr_model.pt"
METRICS_FILE = OUTPUT_DIR / "training_metrics_bpr.json"

EMBEDDING_DIM = 64
EPOCHS = 5
BATCH_SIZE = 8192
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-6
VALIDATION_FRACTION = 0.10
SEED = 42

DEVICE = torch.device("cpu")


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# Model
# --------------------------------------------------

class BPRModel(nn.Module):

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

        self.embedding_dim = embedding_dim

    def score(self, user_ids, movie_ids):

        user_vectors = self.user_embedding(user_ids)
        movie_vectors = self.movie_embedding(movie_ids)

        interaction = (
            user_vectors * movie_vectors
        ).sum(dim=1) / np.sqrt(self.embedding_dim)

        user_bias = self.user_bias(user_ids).squeeze(1)
        movie_bias = self.movie_bias(movie_ids).squeeze(1)

        return interaction + user_bias + movie_bias

    def forward(self, users, positive_movies, negative_movies):

        positive_scores = self.score(users, positive_movies)
        negative_scores = self.score(users, negative_movies)

        return positive_scores, negative_scores


# --------------------------------------------------
# Validate files
# --------------------------------------------------

for path in [PAIR_FILE, USER_MAP, MOVIE_MAP]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


print("=" * 60)
print("BAYESIAN PERSONALIZED RANKING MODEL")
print("=" * 60)

print("\nStep 1: Loading training pairs...")

pairs = pd.read_parquet(PAIR_FILE)

required_columns = {
    "user_idx",
    "positive_movie_idx",
    "negative_movie_idx",
}

if not required_columns.issubset(pairs.columns):
    raise ValueError("BPR dataset has missing columns.")

pairs = pairs.astype({
    "user_idx": "int64",
    "positive_movie_idx": "int64",
    "negative_movie_idx": "int64",
})

print(f"Total pairs: {len(pairs):,}")


# --------------------------------------------------
# Load embedding dimensions from ID maps
# --------------------------------------------------

print("\nStep 2: Checking ID maps...")

user_map = pd.read_parquet(USER_MAP, columns=["user_idx"])
movie_map = pd.read_parquet(MOVIE_MAP, columns=["movie_idx"])

num_users = int(user_map["user_idx"].max()) + 1
num_movies = int(movie_map["movie_idx"].max()) + 1

del user_map, movie_map

print(f"Users: {num_users:,}")
print(f"Movies: {num_movies:,}")


# --------------------------------------------------
# Validate indices
# --------------------------------------------------

if pairs["user_idx"].min() < 0 or pairs["user_idx"].max() >= num_users:
    raise ValueError("User index outside embedding range.")

for column in ["positive_movie_idx", "negative_movie_idx"]:
    if pairs[column].min() < 0 or pairs[column].max() >= num_movies:
        raise ValueError(f"{column} outside embedding range.")

if (pairs["positive_movie_idx"] == pairs["negative_movie_idx"]).any():
    raise ValueError("Found identical positive and negative movies.")


# --------------------------------------------------
# Train-validation split
# --------------------------------------------------

print("\nStep 3: Splitting training and validation data...")

shuffled_indices = np.random.permutation(len(pairs))

validation_size = int(len(pairs) * VALIDATION_FRACTION)

validation_indices = shuffled_indices[:validation_size]
training_indices = shuffled_indices[validation_size:]

training_data = pairs.iloc[training_indices].to_numpy(dtype=np.int64)
validation_data = pairs.iloc[validation_indices].to_numpy(dtype=np.int64)

del pairs, shuffled_indices, validation_indices, training_indices

print(f"Training pairs:   {len(training_data):,}")
print(f"Validation pairs: {len(validation_data):,}")


# --------------------------------------------------
# Convert to tensors
# --------------------------------------------------

train_tensor = torch.from_numpy(training_data).to(DEVICE)
validation_tensor = torch.from_numpy(validation_data).to(DEVICE)

del training_data, validation_data


# --------------------------------------------------
# Initialize model
# --------------------------------------------------

print("\nStep 4: Initializing BPR model...")

model = BPRModel(
    num_users=num_users,
    num_movies=num_movies,
    embedding_dim=EMBEDDING_DIM,
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# --------------------------------------------------
# Validation function
# --------------------------------------------------

def evaluate(model, data):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for start in range(0, len(data), BATCH_SIZE):

            batch = data[start:start + BATCH_SIZE]

            users = batch[:, 0]
            positive_movies = batch[:, 1]
            negative_movies = batch[:, 2]

            positive_scores, negative_scores = model(
                users,
                positive_movies,
                negative_movies,
            )

            difference = positive_scores - negative_scores

            loss = F.softplus(-difference).mean()

            batch_size = len(batch)

            total_loss += loss.item() * batch_size
            correct += (difference > 0).sum().item()
            total += batch_size

    return {
        "loss": total_loss / total,
        "pairwise_accuracy": correct / total,
    }


# --------------------------------------------------
# Training
# --------------------------------------------------

print("\nStep 5: Training BPR model...")
print(f"Device: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")

history = []

for epoch in range(EPOCHS):

    model.train()

    permutation = torch.randperm(len(train_tensor))

    total_loss = 0.0
    total_examples = 0

    for start in range(0, len(train_tensor), BATCH_SIZE):

        indices = permutation[start:start + BATCH_SIZE]
        batch = train_tensor[indices]

        users = batch[:, 0]
        positive_movies = batch[:, 1]
        negative_movies = batch[:, 2]

        optimizer.zero_grad()

        positive_scores, negative_scores = model(
            users,
            positive_movies,
            negative_movies,
        )

        difference = positive_scores - negative_scores

        loss = F.softplus(-difference).mean()

        loss.backward()
        optimizer.step()

        batch_size = len(batch)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    train_loss = total_loss / total_examples

    validation_metrics = evaluate(model, validation_tensor)

    epoch_metrics = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": validation_metrics["loss"],
        "validation_pairwise_accuracy": (
            validation_metrics["pairwise_accuracy"]
        ),
    }

    history.append(epoch_metrics)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Validation Loss: {validation_metrics['loss']:.6f} | "
        f"Validation Pairwise Accuracy: "
        f"{validation_metrics['pairwise_accuracy']:.4f}"
    )


# --------------------------------------------------
# Save checkpoint
# --------------------------------------------------

print("\nStep 6: Saving model...")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_users": num_users,
        "num_movies": num_movies,
        "embedding_dim": EMBEDDING_DIM,
        "model_type": "BPR",
    },
    CHECKPOINT_FILE,
)


# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics = {
    "model": "BPR Matrix Factorization",
    "num_users": num_users,
    "num_movies": num_movies,
    "embedding_dim": EMBEDDING_DIM,
    "training_pairs": int(
        len(train_tensor)
    ),
    "validation_pairs": int(
        len(validation_tensor)
    ),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "history": history,
    "final_validation": history[-1],
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("BPR TRAINING COMPLETED")
print("=" * 60)

print(f"Final validation loss: {history[-1]['validation_loss']:.6f}")

print(
    "Final pairwise accuracy: "
    f"{history[-1]['validation_pairwise_accuracy']:.4f}"
)

print(f"\nModel saved: {CHECKPOINT_FILE}")
print(f"Metrics saved: {METRICS_FILE}")

print("=" * 60)
