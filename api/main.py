from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import sys
import re
from src.explainer import explain_recommendations
from pathlib import Path

# add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.recommender import (
    load_ratings, load_movies, load_tmdb,
    build_content_model, build_collab_model,
    build_svd_model, hybrid_recommend,
    get_content_recommendations
)
from src.sentiment import (
    load_imdb, build_vader, load_distilbert,
    rerank_with_sentiment, distilbert_score
)

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
print("Loading all models and data...")

ratings  = load_ratings(sample=True)
movies   = load_movies()
tmdb     = load_tmdb()
imdb     = load_imdb()

tmdb_clean, tfidf_matrix, tfidf, title_to_idx = build_content_model(tmdb)
user_movie_matrix, ratings_filtered            = build_collab_model(ratings)
svd_data                                       = build_svd_model(ratings)
vader_analyzer                                 = build_vader()

# load fine-tuned DistilBERT
try:
    db_model, db_tokenizer = load_distilbert()
    USE_DISTILBERT = True
    print("✓ DistilBERT loaded")
except Exception as e:
    print(f"⚠ DistilBERT not available ({e}), using VADER only")
    USE_DISTILBERT = False

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
            title_to_idx      = title_to_idx,
            svd_data          = svd_data,
            alpha=0.4, beta=0.3, gamma=0.3,
            n=n * 2   # get extra candidates for re-ranking
        )

        if not hybrid_recs:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for user {user_id}"
            )

        # re-rank with sentiment
        reranked = rerank_with_sentiment(
            hybrid_recs, vader_analyzer,
            sentiment_weight=sentiment_weight
        )[:n]
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
            title_to_idx, n=n,
            genre_filter=genre,
            lang_filter=lang
        )

        if not recs:
            raise HTTPException(
                status_code=404,
                detail=f"No results for '{title}'. "
                       f"Check spelling or try a different genre."
            )

        return {
            "query_title": title,
            "count":       len(recs),
            "genre_filter": genre,
            "lang_filter":  lang,
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