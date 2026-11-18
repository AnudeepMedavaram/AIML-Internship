
from pathlib import Path
import argparse

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

POPULAR_MOVIES_PATH = (
    ROOT / "features" / "output" / "popular_movies.parquet"
)

TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"


class MovieRecommender:

    def __init__(self):

        print("Loading recommendation system...")

        self.popular_movies = pd.read_parquet(
            POPULAR_MOVIES_PATH
        )

        self.train = pd.read_parquet(
            TRAIN_PATH,
            columns=["userId", "movieId"]
        )

        self.train["userId"] = self.train["userId"].astype(int)
        self.train["movieId"] = self.train["movieId"].astype(int)

        # Build user watch-history lookup.
        self.seen_by_user = (
            self.train.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        print("Recommendation system loaded.")
        print(f"Popularity catalogue: {len(self.popular_movies):,}")
        print(f"Users with training history: {len(self.seen_by_user):,}")

    def recommend(self, user_id=None, top_k=10):

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Get watched movies if user exists.
        if user_id is not None:
            seen_movies = self.seen_by_user.get(
                int(user_id), set()
            )
        else:
            seen_movies = set()

        # Exclude watched movies and return popular unseen movies.
        recommendations = self.popular_movies[
            ~self.popular_movies["movieId"].isin(seen_movies)
        ].head(top_k).copy()

        if recommendations.empty:
            return pd.DataFrame()

        recommendations.insert(
            0,
            "rank",
            range(1, len(recommendations) + 1)
        )

        return recommendations[
            [
                "rank",
                "movieId",
                "title",
                "genres",
                "rating_count",
                "average_rating"
            ]
        ]


def main():

    parser = argparse.ArgumentParser(
        description="Popularity-based Movie Recommendation System"
    )

    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="MovieLens user ID"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations"
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    recommender = MovieRecommender()

    print("\n" + "=" * 75)

    if args.user_id is None:
        print("TOP POPULAR MOVIE RECOMMENDATIONS")
    else:
        print(f"TOP {args.top_k} RECOMMENDATIONS FOR USER {args.user_id}")

    print("=" * 75)

    recommendations = recommender.recommend(
        user_id=args.user_id,
        top_k=args.top_k
    )

    if recommendations.empty:
        print("No recommendations available.")
    else:
        print(
            recommendations.to_string(
                index=False,
                formatters={
                    "average_rating": "{:.2f}".format
                }
            )
        )

    print("\nRecommendation generation completed.")


if __name__ == "__main__":
    main()

