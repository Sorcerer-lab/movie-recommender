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

# ── collaborative filter + clusters ───────────────────────────
user_movie_matrix, ratings_filtered = build_collab_model(ratings)
cluster_data = build_clustered_collab_model(ratings)
vader_analyzer = build_vader()

# ── load DistilBERT ───────────────────────────────────────────
try:
    db_model, db_tokenizer = load_distilbert()
    USE_DISTILBERT = True
    print("✓ DistilBERT loaded")
except Exception as e:
    print(f"⚠ DistilBERT not available ({e}), using VADER fallback")
    db_model      = None
    db_tokenizer  = None
    USE_DISTILBERT = False

# ── load optimal hybrid weights ───────────────────────────────
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

# ── final score weights (fixed per your spec) ─────────────────
HYBRID_W     = 0.70   # weight for hybrid_score
TMDB_W       = 0.15   # weight for TMDB vote quality score
DISTILBERT_W = 0.15   # weight for DistilBERT sentiment score

print(f"✓ Final score = {HYBRID_W}×hybrid + {TMDB_W}×tmdb + {DISTILBERT_W}×distilbert")
print("✓ All models ready — API is live!")


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Movie Recommender API is running!",
        "endpoints": ["/recommend", "/similar", "/user/{user_id}/profile"],
        "scoring": {
            "formula": "final = 0.70×hybrid + 0.15×tmdb + 0.15×distilbert",
            "distilbert_active": USE_DISTILBERT
        }
    }


@app.get("/recommend")
def recommend(user_id: int, n: int = 10):
    """
    Get hybrid recommendations for a user, re-ranked by the
    three-component scoring formula:

        final_score = 0.70 × hybrid_score
                    + 0.15 × tmdb_score          (vote quality × confidence)
                    + 0.15 × distilbert_score     (sentiment of movie overview)

    Parameters
    ----------
    user_id : int   — MovieLens user ID
    n       : int   — number of recommendations to return (default 10)
    """
    try:
        # ── step 1: generate hybrid candidates ────────────────
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
            n=n * 2          # oversample, trim after re-ranking
        )

        if not hybrid_recs:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for user {user_id}"
            )

        # ── step 2: re-rank with 3-component formula ──────────
        reranked = rerank_with_sentiment(
            hybrid_recs,
            vader_analyzer,
            tmdb_df              = tmdb_clean,
            distilbert_model     = db_model,        # None → VADER fallback
            distilbert_tokenizer = db_tokenizer,
            hybrid_weight        = HYBRID_W,
            tmdb_weight          = TMDB_W,
            distilbert_weight    = DISTILBERT_W
        )[:n]

        # ── step 3: add audience label for dashboard ──────────
        for r in reranked:
            s = r.get('distilbert_score', r.get('sentiment_score', 0.5))
            r['audience_label'] = (
                "Audiences loved it" if s > 0.75 else
                "Highly rated"       if s > 0.60 else
                "Well rated"         if s > 0.45 else
                "Mixed reception"    if s > 0.30 else
                "Limited ratings"
            )

        # ── step 4: add rule-based explanations ───────────────
        explained = explain_recommendations(
            reranked, ratings, movies, user_id
        )

        return {
            "user_id":   user_id,
            "count":     len(explained),
            "scoring": {
                "hybrid_weight":     HYBRID_W,
                "tmdb_weight":       TMDB_W,
                "distilbert_weight": DISTILBERT_W,
                "distilbert_active": USE_DISTILBERT
            },
            "recommendations": explained
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar")
def similar(title: str, n: int = 10,
            genre: str = None, lang: str = None):
    """
    Find movies similar to the given title using TF-IDF content similarity.

    Parameters
    ----------
    title : str         — movie title (fuzzy matched)
    n     : int         — number of results (default 10)
    genre : str | None  — optional genre filter e.g. "Action"
    lang  : str | None  — optional language code e.g. "en", "hi", "ko"
    """
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
                detail=f"No results found for '{title}'."
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
    Return a user's taste profile: genre breakdown, decade breakdown,
    average rating, and top-rated movies.
    Used by the Streamlit dashboard to render charts.
    """
    user_ratings = ratings[ratings['userId'] == user_id]

    if user_ratings.empty:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found"
        )

    merged = user_ratings.merge(movies, on='movieId')

    def extract_decade(title):
        match = re.search(r'\((\d{4})\)', title)
        if match:
            decade = (int(match.group(1)) // 10) * 10
            return f"{decade}s"
        return "Unknown"

    merged['decade'] = merged['title'].apply(extract_decade)

    genre_counts = {}
    for _, row in merged.iterrows():
        for genre in str(row['genres']).split('|'):
            genre = genre.strip()
            if genre and genre != 'nan':
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

    decade_counts = merged['decade'].value_counts().to_dict()

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


@app.get("/health")
def health():
    return {
        "status":            "ok",
        "distilbert_active": USE_DISTILBERT,
        "scoring_formula":   f"{HYBRID_W}×hybrid + {TMDB_W}×tmdb + {DISTILBERT_W}×distilbert"
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