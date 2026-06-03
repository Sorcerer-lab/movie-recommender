import pickle
import json
import sys
import re
import scipy.sparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.recommender import (
    load_ratings, load_movies, load_tmdb,
    build_content_model, build_collab_model,
    build_svd_model, hybrid_recommend,
    get_content_recommendations,
    build_clustered_collab_model
)
from src.sentiment import (
    load_imdb, build_vader, load_distilbert,
    rerank_with_sentiment, distilbert_score
)
from src.explainer import explain_recommendations

# ── initialise app ─────────────────────────────────────────────
app = FastAPI(
    title="Movie Recommender API",
    description="Hybrid recommendation engine with sentiment re-ranking",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── global state — loaded once at startup ──────────────────────

print("Loading models and data...")

ratings  = load_ratings(sample=True)
movies   = load_movies()
imdb     = load_imdb()

# ── load pre-trained TF-IDF if available ──────────────────────
tmdb_pkl = Path("models/tmdb_clean.pkl")
if tmdb_pkl.exists():
    print("  Loading pre-trained TF-IDF from disk...")
    tmdb_clean   = pd.read_pickle("models/tmdb_clean.pkl")
    tfidf_matrix = scipy.sparse.load_npz(
        "models/tfidf_matrix.npz"
    )
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    # rebuild id_to_idx and search_df
    id_to_idx = pd.Series(
        tmdb_clean.index, index=tmdb_clean['id']
    )
    search_df = tmdb_clean[
        ['id', 'title', 'release_date',
         'genre_names', 'vote_count']
    ].copy()
    search_df['title_lower'] = (
        search_df['title']
        .fillna('').str.lower().str.strip()
    )
    print("✓ Pre-trained TF-IDF loaded")
else:
    print("  Building TF-IDF from scratch...")
    tmdb = load_tmdb()
    tmdb_clean, tfidf_matrix, tfidf, id_to_idx, search_df = \
        build_content_model(tmdb)
    print("✓ TF-IDF built")

# ── load pre-trained SVD if available ─────────────────────────
svd_pkl = Path("models/svd_model.pkl")
if svd_pkl.exists():
    with open(svd_pkl, "rb") as f:
        svd_data = pickle.load(f)
    print("✓ Pre-trained SVD loaded")
else:
    svd_data = build_svd_model(ratings)
    print("✓ SVD built")

# ── collaborative filter — always build fresh ─────────────────
# ── collaborative filter + clusters ───────────────────────────
from src.recommender import build_clustered_collab_model
user_movie_matrix, ratings_filtered = build_collab_model(ratings)
cluster_data = build_clustered_collab_model(ratings)
vader_analyzer = build_vader()

# ── load DistilBERT ───────────────────────────────────────────
try:
    db_model, db_tokenizer = load_distilbert()
    USE_DISTILBERT = True
    print("✓ DistilBERT loaded")
except Exception as e:
    print(f"⚠ DistilBERT not available ({e}), using VADER only")
    USE_DISTILBERT = False
    # ── load optimal weights ──────────────────────────────────────
weights_path = Path("models/optimal_weights.json")
if weights_path.exists():
    with open(weights_path) as f:
        w = json.load(f)
    ALPHA = w['alpha']
    BETA  = w['beta']
    GAMMA = w['gamma']
    print(f"✓ Optimal weights: α={ALPHA} β={BETA} γ={GAMMA}")
else:
    ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3
    print("⚠ Using default weights α=0.4 β=0.3 γ=0.3")

print("✓ All models ready — API is live!")

# ══════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════

def get_sentiment_score(text):
    """Use DistilBERT if available, else VADER."""
    if USE_DISTILBERT:
        raw = distilbert_score(text, db_model, db_tokenizer)
        return raw  # already 0-1
    else:
        from src.sentiment import vader_score
        raw = vader_score(text, vader_analyzer)
        return (raw + 1) / 2   # map -1..+1 → 0..1


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Movie Recommender API is running!",
        "endpoints": ["/recommend", "/similar", "/user/{user_id}/profile"]
    }


@app.get("/recommend")
def recommend(user_id: int, n: int = 10,
              sentiment_weight: float = 0.2):
    """
    Get hybrid recommendations for a user, re-ranked by sentiment.

    Parameters:
        user_id          — MovieLens user ID
        n                — number of recommendations (default 10)
        sentiment_weight — how much sentiment affects ranking (0-1)
    """
    try:
        # get hybrid recommendations
        hybrid_recs = hybrid_recommend(
            user_id           = user_id,
            user_movie_matrix = user_movie_matrix,
            ratings_df        = ratings_filtered,
            movies_df         = movies,
            tmdb_df           = tmdb_clean,
            tfidf_matrix      = tfidf_matrix,
            id_to_idx         = id_to_idx,
            search_df         = search_df,
            svd_data          = svd_data,
            alpha=ALPHA, beta=BETA, gamma=GAMMA,
            n=n * 2
        )

        if not hybrid_recs:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for user {user_id}"
            )

        # re-rank with sentiment
        reranked = rerank_with_sentiment(
            hybrid_recs,
            vader_analyzer,
            tmdb_df          = tmdb_clean,
            sentiment_weight = sentiment_weight
        )[:n]
        for r in reranked:
            s = r.get('sentiment_score', 0.5)
            r['audience_label'] = (
                "Highly rated"    if s > 0.70 else
                "Well rated"      if s > 0.50 else
                "Mixed reception" if s > 0.30 else
                "Limited ratings"
            )
        explained = explain_recommendations(
            reranked, ratings, movies, user_id
        )
        return {
            "user_id":         user_id,
            "count":           len(reranked),
            "sentiment_weight": sentiment_weight,
            "recommendations":explained
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar")
def similar(title: str, n: int = 10,
            genre: str = None, lang: str = None):
    try:
        recs = get_content_recommendations(
            title, tmdb_clean, tfidf_matrix,
            id_to_idx, search_df, n=n,
            genre_filter=genre,
            lang_filter=lang
        )
        if not recs:
            raise HTTPException(
                status_code=404,
                detail=f"No results for '{title}'."
            )
        return {
            "query_title": title,
            "count":       len(recs),
            "similar":     recs
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/profile")
def user_profile(user_id: int):
    """
    Get a user's taste profile — genre preferences and rating history.
    Used by the Streamlit dashboard to build charts.
    """
    user_ratings = ratings[ratings['userId'] == user_id]

    if user_ratings.empty:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found"
        )

    # join with movies to get genres
    merged = user_ratings.merge(movies, on='movieId')

    # extract decade from title year
    def extract_decade(title):
        match = re.search(r'\((\d{4})\)', title)
        if match:
            year    = int(match.group(1))
            decade  = (year // 10) * 10
            return f"{decade}s"
        return "Unknown"

    merged['decade'] = merged['title'].apply(extract_decade)

    # genre breakdown
    genre_counts = {}
    for _, row in merged.iterrows():
        for genre in str(row['genres']).split('|'):
            genre = genre.strip()
            if genre and genre != 'nan':
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

    # decade breakdown
    decade_counts = merged['decade'].value_counts().to_dict()

    # top rated movies
    top_rated = (
        merged.sort_values('rating', ascending=False)
        .head(10)[['title', 'rating', 'genres']]
        .to_dict('records')
    )

    return {
        "user_id":       user_id,
        "total_ratings": len(user_ratings),
        "avg_rating":    round(float(user_ratings['rating'].mean()), 2),
        "genre_counts":  genre_counts,
        "decade_counts": decade_counts,
        "top_rated":     top_rated
    }
@app.get("/search-title")
def search_title(q: str):
    from rapidfuzz import process, fuzz
    matches = process.extract(
        q.lower(), list(search_df['title_lower']),
        scorer=fuzz.token_sort_ratio, limit=5
    )
    return {
        "query":   q,
        "matches": [{"title": m, "score": s}
                    for m, s, _ in matches]
    }