import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════
# 1. RULE-BASED EXPLAINER
# ══════════════════════════════════════════════════════════════

def extract_year(title):
    """Extract year from MovieLens title format 'Movie Name (1999)'."""
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else None


def get_shared_genres(movie_genres, user_top_genres):
    """Return genres that appear in both movie and user's top genres."""
    movie_set = set(g.strip() for g in movie_genres.split('|')
                    if g.strip())
    return movie_set & user_top_genres


def build_user_taste_profile(user_id, ratings_df, movies_df):
    """
    Build a simple taste profile for a user:
    - top genres they rate highly
    - favourite decades
    - directors/actors they like (if available)
    """
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    merged       = user_ratings.merge(movies_df, on='movieId')

    # top genres from highly rated movies (4.0+)
    liked = merged[merged['rating'] >= 4.0]
    genre_counts = {}
    for _, row in liked.iterrows():
        for genre in str(row['genres']).split('|'):
            genre = genre.strip()
            if genre and genre != 'nan':
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

    top_genres = set(
        sorted(genre_counts, key=genre_counts.get, reverse=True)[:5]
    )

    # favourite decades
    decade_counts = {}
    for _, row in liked.iterrows():
        year = extract_year(str(row['title']))
        if year:
            decade = (year // 10) * 10
            decade_counts[decade] = decade_counts.get(decade, 0) + 1

    top_decade = (max(decade_counts, key=decade_counts.get)
                  if decade_counts else None)

    # top rated movie titles (for "because you liked X" reasons)
    top_movies = (
        merged.sort_values('rating', ascending=False)
        .head(5)['title']
        .tolist()
    )

    return {
        'top_genres':  top_genres,
        'top_decade':  top_decade,
        'top_movies':  top_movies,
        'total_rated': len(user_ratings)
    }


def explain_recommendation(rec_title, rec_genres, sentiment_score,
                            hybrid_score, user_profile,
                            source='hybrid'):
    """
    Generate a human-readable explanation for a recommendation.

    Priority order:
    1. Genre match with user's taste
    2. Decade match
    3. High sentiment (audience loved it)
    4. Similar to a movie they liked
    5. Generic fallback
    """
    reasons = []

    # ── reason 1: genre match ──────────────────────────────────
    if rec_genres and rec_genres != 'nan':
        shared = get_shared_genres(
            rec_genres.replace(' ', '|'),
            user_profile['top_genres']
        )
        if shared:
            genre_str = ' & '.join(list(shared)[:2])
            reasons.append(
                f"Matches your love of {genre_str}"
            )

    # ── reason 2: decade match ─────────────────────────────────
    year = extract_year(rec_title)
    if year and user_profile['top_decade']:
        decade = (year // 10) * 10
        if decade == user_profile['top_decade']:
            reasons.append(
                f"From the {decade}s — your favourite era"
            )

    # ── reason 3: high audience sentiment ─────────────────────
    if sentiment_score > 0.75:
        reasons.append("Audiences absolutely loved this film")
    elif sentiment_score > 0.60:
        reasons.append("Consistently strong audience reception")

    # ── reason 4: similar to liked movie ──────────────────────
    if user_profile['top_movies']:
        seed_title = user_profile['top_movies'][0]
        # strip year for cleaner display
        seed_clean = re.sub(r'\s*\(\d{4}\)\s*$', '', seed_title)
        reasons.append(f"Because you liked {seed_clean}")

    # ── reason 5: fallback ────────────────────────────────────
    if not reasons:
        reasons.append(
            "Highly rated by users with similar taste to you"
        )

    # pick the two most specific reasons
    primary   = reasons[0]
    secondary = reasons[1] if len(reasons) > 1 else None

    explanation = primary
    if secondary:
        explanation += f" · {secondary}"

    return explanation


def explain_recommendations(recs, ratings_df, movies_df, user_id):
    """
    Add explanations to a full recommendation list.
    Returns the same list with an 'explanation' field added.
    """
    profile = build_user_taste_profile(user_id, ratings_df, movies_df)

    explained = []
    for rec in recs:
        # try to find genre from movies_df
        title    = rec['title']
        clean_t  = re.sub(r'\s*\(\d{4}\)\s*$', '', title)
        genre_row = movies_df[
            movies_df['title'].str.contains(
                re.escape(clean_t), case=False, na=False
            )
        ]
        genres = (genre_row['genres'].values[0]
                  if len(genre_row) > 0 else '')

        explanation = explain_recommendation(
            rec_title      = title,
            rec_genres     = genres,
            sentiment_score= rec.get('sentiment_score', 0.5),
            hybrid_score   = rec.get('hybrid_score', 0.5),
            user_profile   = profile
        )

        explained.append({**rec, 'explanation': explanation})

    return explained


# ══════════════════════════════════════════════════════════════
# 2. LIME EXPLAINER — why did sentiment score this way?
# ══════════════════════════════════════════════════════════════

def explain_sentiment_with_lime(review_text, model, tokenizer,
                                 n_words=8):
    """
    Use LIME to find which words drove the sentiment prediction.

    LIME works by:
    1. Taking the review and randomly masking words
    2. Running each masked version through the model
    3. Finding which words, when removed, change the score most
    4. Those words are the most influential
    """
    try:
        from lime.lime_text import LimeTextExplainer
        import torch

        explainer = LimeTextExplainer(class_names=['negative', 'positive'])

        def predict_proba(texts):
            """Wrapper so LIME can call our DistilBERT model."""
            results = []
            for text in texts:
                enc = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                with torch.no_grad():
                    out   = model(**enc)
                    probs = torch.softmax(out.logits, dim=1)
                    results.append(probs[0].numpy())
            return np.array(results)

        exp = explainer.explain_instance(
            review_text,
            predict_proba,
            num_features=n_words,
            num_samples=100    # lower = faster, higher = more accurate
        )

        # get word → importance score pairs
        word_scores = exp.as_list()
        return word_scores

    except Exception as e:
        return [("LIME unavailable", 0.0)]


def format_lime_explanation(word_scores):
    """Turn LIME word scores into a readable string."""
    positive_words = [w for w, s in word_scores if s > 0]
    negative_words = [w for w, s in word_scores if s < 0]

    parts = []
    if positive_words:
        parts.append(f"Positive signals: {', '.join(positive_words[:4])}")
    if negative_words:
        parts.append(f"Negative signals: {', '.join(negative_words[:4])}")

    return ' | '.join(parts) if parts else "No clear signals found"


# ══════════════════════════════════════════════════════════════
# MAIN — test explainer
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.recommender import load_ratings, load_movies

    print("=" * 55)
    print("PHASE 6 — Explainability Layer")
    print("=" * 55)

    ratings = load_ratings(sample=True)
    movies  = load_movies()

    # build a dummy profile for user 187
    print("\nBuilding taste profile for user 187...")
    profile = build_user_taste_profile(187, ratings, movies)

    print(f"  Top genres : {profile['top_genres']}")
    print(f"  Top decade : {profile['top_decade']}s")
    print(f"  Top movies : {profile['top_movies'][:3]}")

    # test explanations on dummy recs
    print("\n--- Rule-based explanations ---")
    dummy_recs = [
        {'title': 'The Matrix (1999)',
         'hybrid_score': 0.48,
         'sentiment_score': 0.82,
         'final_score': 0.55},
        {'title': 'Inception (2010)',
         'hybrid_score': 0.41,
         'sentiment_score': 0.91,
         'final_score': 0.51},
        {'title': 'Toy Story (1995)',
         'hybrid_score': 0.47,
         'sentiment_score': 0.76,
         'final_score': 0.53},
        {'title': 'The Dark Knight (2008)',
         'hybrid_score': 0.38,
         'sentiment_score': 0.88,
         'final_score': 0.49},
        {'title': 'Forrest Gump (1994)',
         'hybrid_score': 0.35,
         'sentiment_score': 0.79,
         'final_score': 0.46},
    ]

    explained = explain_recommendations(
        dummy_recs, ratings, movies, user_id=187
    )

    print(f"\n{'#':<4} {'Title':<35} {'Explanation'}")
    print("-" * 80)
    for i, r in enumerate(explained, 1):
        print(f"{i:<4} {r['title']:<35} {r['explanation']}")

    # test LIME on a sample review
    print("\n--- LIME sentiment explanation ---")
    sample_review = ("This film is an absolute masterpiece. "
                     "The acting is incredible and the story "
                     "keeps you on the edge of your seat.")

    print(f"  Review: {sample_review}")

    try:
        from src.sentiment import load_distilbert
        model, tokenizer = load_distilbert()
        word_scores      = explain_sentiment_with_lime(
            sample_review, model, tokenizer
        )
        print(f"  LIME:   {format_lime_explanation(word_scores)}")
    except Exception as e:
        print(f"  LIME skipped: {e}")