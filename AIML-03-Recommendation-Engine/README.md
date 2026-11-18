# Movie Recommendation Engine

A movie recommendation system built using the **MovieLens 25M dataset**. The project explores popularity-based recommendations and neural collaborative filtering using a two-tower retrieval model.

## 1. Project Overview

The system recommends movies based on popularity while excluding movies a user has already watched.

Two approaches were evaluated:

- **Popularity-based recommendation:** Uses movie interaction counts and average ratings to rank popular movies.
- **Two-tower neural retrieval:** Learns user and movie embeddings to generate personalized recommendations.

The popularity baseline performed better in the current retrieval evaluation and is used by the final application.

## 2. Technology Stack

- Python
- Pandas, NumPy, PyArrow
- PyTorch
- FAISS
- DuckDB
- Scikit-learn
- FastAPI
- Streamlit

## 3. Dataset

**MovieLens 25M**

- 25 million ratings
- 162,541 unique users
- 59,047 movies with ratings
- Ratings from 1995 to 2019

The data was split chronologically into training, validation and test datasets.

## 4. Project Architecture

1. Data exploration and preprocessing
2. Feature engineering
3. Popularity baseline generation
4. Two-tower model training
5. FAISS retrieval experiments
6. Recommendation evaluation
7. FastAPI backend
8. Streamlit dashboard

## 5. Model Evaluation

Popularity baseline and two-tower retrieval were compared using the same 200 evaluation users and 1,000 candidate movies.

| Metric | Popularity | Two-Tower |
|---|---:|---:|
| Recall@10 | 0.06179 | 0.01164 |
| NDCG@10 | 0.10561 | 0.01928 |

**Observation:** The popularity baseline outperformed the current two-tower model on both evaluation metrics. Therefore, popularity-based recommendations were selected for the final application.

## 6. Application Features

- MovieLens user ID input
- Configurable recommendation count
- Popular movie recommendations
- Exclusion of previously watched movies for known users
- General popular recommendations for new or unknown users
- FastAPI recommendation endpoints
- Interactive Streamlit dashboard

## 7. Running the Project

### Install dependencies

```bash
pip install pandas numpy pyarrow torch faiss-cpu duckdb scikit-learn fastapi uvicorn streamlit requests
```

### Start the API

```bash
uvicorn api.main:app --reload
```

API documentation:

`http://127.0.0.1:8000/docs`

### Start the dashboard

In another terminal:

```bash
streamlit run dashboard/app.py
```

### Run recommendations directly

```bash
python retrieval/recommend.py
```

For a specific user:

```bash
python retrieval/recommend.py --user-id 1 --top-k 10
```

## 8. Repository Structure

```text
AIML-03-Recommendation-Engine/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── features/
│   ├── build_features.py
│   └── output/
├── models/
│   ├── train_two_tower.py
│   ├── evaluate_retrieval.py
│   ├── compare_retrieval_baselines.py
│   └── output/
├── preprocessing/
│   ├── eda.py
│   └── temporal_split.py
├── retrieval/
│   ├── build_index.py
│   └── recommend.py
├── tests/
└── README.md
```

## 9. Future Improvements

- Improve personalized recommendation performance.
- Experiment with hybrid recommendation approaches.
- Add movie search and genre filters.
- Improve cold-start recommendations.
- Deploy the application.

## 10. Conclusion

This project demonstrates an end-to-end movie recommendation workflow, including data processing, feature engineering, model experimentation, retrieval evaluation, API development and dashboard integration.

The final application uses popularity-based recommendations with watched-movie filtering, based on the comparative evaluation results.