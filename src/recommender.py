import pandas as pd
import numpy as np
from pathlib import Path
import ast
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

# ── Paths ──────────────────────────────────────────────────────
DATA_RAW = Path("data/raw")


# ══════════════════════════════════════════════════════════════
# 1. DATA LOADERS
# ══════════════════════════════════════════════════════════════

def load_ratings(sample=True):
    for name in ["ratings.csv", "rating.csv"]:
        path = DATA_RAW / "ml-25m" / name
        if path.exists():
            df = pd.read_csv(path)
            if sample:
                df = df.sample(n=min(2_000_000, len(df)), random_state=42)
            print(f"✓ Ratings loaded: {df.shape}")
            return df
    raise FileNotFoundError("ratings.csv not found in data/raw/ml-25m/")


def load_movies():
    for name in ["movies.csv", "movie.csv"]:
        path = DATA_RAW / "ml-25m" / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"✓ Movies loaded: {df.shape}")
            return df
    raise FileNotFoundError("movies.csv not found in data/raw/ml-25m/")


def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return ' '.join([g['name'] for g in genres])
    except:
        return ''


def load_tmdb():
    path = DATA_RAW / "tmdb" / "movies_metadata.csv"
    df   = pd.read_csv(path, low_memory=False)
    df   = df[pd.to_numeric(df['id'], errors='coerce').notna()]
    df['id']          = df['id'].astype(int)
    df['genre_names'] = df['genres'].apply(extract_genres)
    df = df[['id', 'title', 'genre_names', 'overview',
             'release_date', 'vote_average', 'vote_count',
             'original_language']].copy()

    # ── load keywords if available ────────────────────────────
    kw_path = DATA_RAW / "tmdb" / "keywords.csv"
    if kw_path.exists():
        kw = pd.read_csv(kw_path)
        def extract_keywords(kw_str):
            try:
                items = ast.literal_eval(kw_str)
                return ' '.join([k['name'].replace(' ', '')
                                 for k in items])
            except:
                return ''
        kw['keyword_names'] = kw['keywords'].apply(extract_keywords)
        df = df.merge(kw[['id', 'keyword_names']], on='id', how='left')
        df['keyword_names'] = df['keyword_names'].fillna('')
        print("✓ Keywords merged")
    else:
        df['keyword_names'] = ''
        print("⚠ keywords.csv not found — skipping")

    # ── load credits if available ─────────────────────────────
    cr_path = DATA_RAW / "tmdb" / "credits.csv"
    if cr_path.exists():
        cr = pd.read_csv(cr_path)

        def extract_cast(cast_str, n=3):
            try:
                cast = ast.literal_eval(cast_str)
                return ' '.join([c['name'].replace(' ', '')
                                 for c in cast[:n]])
            except:
                return ''

        def extract_director(crew_str):
            try:
                crew = ast.literal_eval(crew_str)
                for c in crew:
                    if c['job'] == 'Director':
                        return c['name'].replace(' ', '')
                return ''
            except:
                return ''

        cr['cast_names']    = cr['cast'].apply(extract_cast)
        cr['director_name'] = cr['crew'].apply(extract_director)
        df = df.merge(
            cr[['id', 'cast_names', 'director_name']],
            on='id', how='left'
        )
        df['cast_names']    = df['cast_names'].fillna('')
        df['director_name'] = df['director_name'].fillna('')
        print("✓ Credits merged")
    else:
        df['cast_names']    = ''
        df['director_name'] = ''
        print("⚠ credits.csv not found — skipping")

    print(f"✓ TMDB loaded: {df.shape}")
    return df


def load_imdb():
    path = DATA_RAW / "imdb" / "IMDB Dataset.csv"
    df   = pd.read_csv(path)
    print(f"✓ IMDB loaded: {df.shape}")
    print(f"  Sentiment:\n{df['sentiment'].value_counts()}")
    return df


# ══════════════════════════════════════════════════════════════
# 2. CONTENT-BASED FILTERING (TF-IDF)
# ══════════════════════════════════════════════════════════════

# ── title aliases — maps what users type to TMDB title ────────
TITLE_ALIASES = {
    'frozen':         'frozen fever',
    'batman':         'the dark knight',
    'avengers':       'the avengers',
    'lion king':      'the lion king',
    'little mermaid': 'the little mermaid',
    'jungle book':    'the jungle book',
    'spiderman':      'spider-man',
    'spider man':     'spider-man',
}


def resolve_title(title):
    """Map common user-typed titles to their TMDB equivalents."""
    return TITLE_ALIASES.get(title.lower().strip(), title)


def detect_language(title):
    """
    Simple heuristic to detect non-English queries.
    Returns language code or None.
    """
    if any(ord(c) > 127 for c in title):
        return 'hi'

    hindi_signals = [
        'golmaal', 'dangal', 'dhoom', 'krrish', 'baahubali',
        'sholay', 'dilwale', 'kabir', 'singham', 'dabangg',
        'bajrangi', 'dil', 'pyaar', 'ishq', 'zindagi', 'muqabla',
        'hrithik', 'salman', 'shahrukh', 'aamir', 'deepika',
    ]
    if any(h in title.lower() for h in hindi_signals):
        return 'hi'

    return None


def build_content_model(tmdb_df):
    """
    Richer soup = better recommendations.
    genres x3 + keywords x2 + director x2 + cast + overview
    """
    df = tmdb_df.copy().reset_index(drop=True)
    df['overview']      = df['overview'].fillna('')
    df['genre_names']   = df['genre_names'].fillna('')
    df['keyword_names'] = df['keyword_names'].fillna('') \
                          if 'keyword_names' in df.columns else ''
    df['cast_names']    = df['cast_names'].fillna('') \
                          if 'cast_names' in df.columns else ''
    df['director_name'] = df['director_name'].fillna('') \
                          if 'director_name' in df.columns else ''

    df['soup'] = (
        df['genre_names']   + ' ' +
        df['genre_names']   + ' ' +
        df['genre_names']   + ' ' +
        df['keyword_names'] + ' ' +
        df['keyword_names'] + ' ' +
        df['director_name'] + ' ' +
        df['director_name'] + ' ' +
        df['cast_names']    + ' ' +
        df['overview']
    )

    # filter movies with almost no metadata
    df['soup_len'] = df['soup'].str.split().str.len()
    df = df[df['soup_len'] >= 10].reset_index(drop=True)
    print(f"  After filtering sparse movies: {len(df)} remaining")

    print("\nBuilding TF-IDF matrix...")
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=15000,
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    print(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")

    title_to_idx = pd.Series(df.index, index=df['title'].str.lower())

    return df, tfidf_matrix, tfidf, title_to_idx


def get_content_recommendations(title, tmdb_df, tfidf_matrix,
                                title_to_idx, n=10,
                                genre_filter=None,
                                lang_filter=None):
    # auto detect language if not specified
    if lang_filter is None:
        lang_filter = detect_language(title)
        if lang_filter:
            print(f"  → Language detected: {lang_filter}")

    title       = resolve_title(title)
    title_lower = title.lower().strip()

    if title_lower not in title_to_idx:
        # try partial match
        matches = [t for t in title_to_idx.index
                   if t.startswith(title_lower)]
        if not matches:
            print(f"  ✗ '{title}' not found.")
            return []
        title_lower = min(matches, key=len)
        print(f"  → Matched to: '{title_lower}'")

    idx        = int(title_to_idx[title_lower])
    movie_vec  = tfidf_matrix[idx]
    sim_scores = cosine_similarity(movie_vec, tfidf_matrix).flatten()

    # ── popularity boost ──────────────────────────────────────
    max_votes = tmdb_df['vote_count'].fillna(0).max()
    if max_votes > 0:
        pop_bonus  = (tmdb_df['vote_count'].fillna(0) / max_votes
                      ).values * 0.35
        sim_scores = sim_scores * 0.65 + pop_bonus

    sim_indices = np.argsort(sim_scores)[::-1][1:n * 4 + 1]

    results = []
    for i in sim_indices:
        if len(results) >= n:
            break
        try:
            score = float(sim_scores[i])
            if score < 0.10:
                continue
            row    = tmdb_df.iloc[int(i)]
            lang   = str(row.get('original_language', 'en'))
            genres = str(row['genre_names'])
            votes = int(row.get('vote_count', 0) or 0)
            if votes < 100:          # skip virtually unknown movies
                continue
            if lang_filter and lang != lang_filter:
                continue
            if genre_filter and \
               genre_filter.lower() not in genres.lower():
                continue

            results.append({
                'title':             row['title'],
                'genre_names':       genres,
                'similarity_score':  round(score, 4),
                'vote_count':        int(row.get('vote_count', 0) or 0),
                'original_language': lang
            })
        except IndexError:
            continue

    return results


# ══════════════════════════════════════════════════════════════
# 3. COLLABORATIVE FILTERING
# ══════════════════════════════════════════════════════════════

def build_collab_model(ratings_df,
                       min_movie_ratings=50,
                       min_user_ratings=20):
    print("\nBuilding collaborative filtering model...")

    movie_counts   = ratings_df['movieId'].value_counts()
    popular_movies = movie_counts[
        movie_counts >= min_movie_ratings
    ].index
    df = ratings_df[ratings_df['movieId'].isin(popular_movies)]

    user_counts  = df['userId'].value_counts()
    active_users = user_counts[
        user_counts >= min_user_ratings
    ].index
    df = df[df['userId'].isin(active_users)]

    print(f"  After filtering: {df['userId'].nunique()} users, "
          f"{df['movieId'].nunique()} movies")

    print("  Building user-movie matrix (this takes ~30 seconds)...")
    user_movie_matrix = df.pivot_table(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)

    print(f"✓ User-movie matrix shape: {user_movie_matrix.shape}")
    return user_movie_matrix, df


def get_collab_recommendations(user_id, user_movie_matrix,
                                ratings_df, n=10):
    if user_id not in user_movie_matrix.index:
        print(f"  ✗ User {user_id} not found.")
        return []

    user_vec    = user_movie_matrix.loc[user_id].values.reshape(1, -1)
    all_vecs    = user_movie_matrix.values
    sim_scores  = cosine_similarity(user_vec, all_vecs).flatten()

    sim_indices   = np.argsort(sim_scores)[::-1][1:21]
    similar_users = user_movie_matrix.index[sim_indices].tolist()

    seen_movies = set(
        ratings_df[ratings_df['userId'] == user_id]['movieId']
    )

    candidate_scores = {}
    for sim_user in similar_users:
        sim_user_ratings = ratings_df[
            (ratings_df['userId'] == sim_user) &
            (ratings_df['rating'] >= 4.0) &
            (~ratings_df['movieId'].isin(seen_movies))
        ]
        for _, row in sim_user_ratings.iterrows():
            mid = row['movieId']
            candidate_scores[mid] = (candidate_scores.get(mid, 0)
                                     + row['rating'])

    top_movies = sorted(candidate_scores.items(),
                        key=lambda x: x[1], reverse=True)[:n]

    return [{'movieId': mid, 'score': round(score, 2)}
            for mid, score in top_movies]


# ══════════════════════════════════════════════════════════════
# 4. SVD MATRIX FACTORIZATION
# ══════════════════════════════════════════════════════════════

def build_svd_model(ratings_df, n_factors=20):
    print("\nBuilding SVD model...")

    top_movies = (ratings_df['movieId']
                  .value_counts().head(500).index)
    top_users  = (ratings_df['userId']
                  .value_counts().head(2000).index)

    filtered = ratings_df[
        ratings_df['movieId'].isin(top_movies) &
        ratings_df['userId'].isin(top_users)
    ]

    print(f"  Filtered to: {filtered['userId'].nunique()} users "
          f"x {filtered['movieId'].nunique()} movies")

    user_movie = filtered.pivot_table(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)

    user_ids  = list(user_movie.index)
    movie_ids = list(user_movie.columns)

    print(f"  Matrix shape: {user_movie.shape}")

    mat = csr_matrix(user_movie.values, dtype=np.float32)
    k   = min(n_factors, min(mat.shape) - 1)

    print(f"  Running SVD with {k} factors...")
    U, sigma, Vt  = svds(mat, k=k)
    sigma_diag    = np.diag(sigma)
    predicted_mat = np.dot(np.dot(U, sigma_diag), Vt)

    print("✓ SVD complete!")

    svd_data = {
        'predicted_matrix': predicted_mat,
        'user_ids':         user_ids,
        'movie_ids':        movie_ids
    }

    with open("models/svd_model.pkl", "wb") as f:
        pickle.dump(svd_data, f)
    print("✓ SVD model saved to models/svd_model.pkl")

    return svd_data


def get_svd_recommendations(user_id, svd_data, ratings_df,
                             movies_df, n=10):
    user_ids  = svd_data['user_ids']
    movie_ids = svd_data['movie_ids']
    pred_mat  = svd_data['predicted_matrix']

    if user_id not in user_ids:
        print(f"  ✗ User {user_id} not in SVD model.")
        return []

    user_idx   = user_ids.index(user_id)
    user_preds = pred_mat[user_idx]

    seen_movies = set(
        ratings_df[ratings_df['userId'] == user_id]['movieId']
    )

    candidates = []
    for movie_idx, pred_rating in enumerate(user_preds):
        movie_id = movie_ids[movie_idx]
        if movie_id not in seen_movies:
            candidates.append((movie_id, pred_rating))

    candidates.sort(key=lambda x: x[1], reverse=True)

    results = []
    for movie_id, pred_rating in candidates[:n]:
        title = movies_df[
            movies_df['movieId'] == movie_id
        ]['title'].values
        title = title[0] if len(title) > 0 else "Unknown"
        results.append({
            'movieId':          movie_id,
            'title':            title,
            'predicted_rating': round(float(pred_rating), 3)
        })

    return results


# ══════════════════════════════════════════════════════════════
# 5. WEIGHTED ENSEMBLE
# ══════════════════════════════════════════════════════════════

def normalize_scores(recs, score_key):
    if not recs:
        return recs
    scores = [r[score_key] for r in recs]
    min_s  = min(scores)
    max_s  = max(scores)
    rng    = max_s - min_s if max_s != min_s else 1.0
    for r in recs:
        r['normalized_score'] = round(
            (r[score_key] - min_s) / rng, 4
        )
    return recs


def hybrid_recommend(user_id, user_movie_matrix, ratings_df,
                     movies_df, tmdb_df, tfidf_matrix,
                     title_to_idx, svd_data,
                     alpha=0.4, beta=0.3, gamma=0.3, n=10):
    print(f"\nGenerating hybrid recommendations for user {user_id}...")
    print(f"  Weights — Collaborative: {alpha} | "
          f"Content: {beta} | SVD: {gamma}")

    collab_recs = get_collab_recommendations(
        user_id, user_movie_matrix, ratings_df, n=50
    )
    svd_recs = get_svd_recommendations(
        user_id, svd_data, ratings_df, movies_df, n=50
    )

    user_top_movies = (
        ratings_df[ratings_df['userId'] == user_id]
        .sort_values('rating', ascending=False)
        .head(3)['movieId'].tolist()
    )

    content_recs = []
    for movie_id in user_top_movies:
        title_row = movies_df[
            movies_df['movieId'] == movie_id
        ]['title']
        if len(title_row) == 0:
            continue
        title       = title_row.values[0]
        title_clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()

        if ', The' in title_clean:
            title_clean = 'The ' + title_clean.replace(', The', '')
        if ', A ' in title_clean:
            title_clean = 'A ' + title_clean.replace(', A ', ' ')

        recs = get_content_recommendations(
            title_clean, tmdb_df, tfidf_matrix, title_to_idx, n=20
        )
        content_recs.extend(recs)

    collab_recs  = normalize_scores(collab_recs,  'score')
    svd_recs     = normalize_scores(svd_recs,      'predicted_rating')
    content_recs = normalize_scores(content_recs,  'similarity_score')

    combined = {}

    for r in collab_recs:
        title = movies_df[
            movies_df['movieId'] == r['movieId']
        ]['title'].values
        if len(title) == 0:
            continue
        title = title[0]
        combined[title] = (combined.get(title, 0)
                           + alpha * r['normalized_score'])

    for r in svd_recs:
        combined[r['title']] = (combined.get(r['title'], 0)
                                + gamma * r['normalized_score'])

    for r in content_recs:
        combined[r['title']] = (combined.get(r['title'], 0)
                                + beta * r['normalized_score'])

    seen_titles = set()
    seen_ids    = ratings_df[
        ratings_df['userId'] == user_id
    ]['movieId'].tolist()

    for mid in seen_ids:
        t = movies_df[movies_df['movieId'] == mid]['title'].values
        if len(t) > 0:
            seen_titles.add(t[0])

    for title in seen_titles:
        combined.pop(title, None)

    ranked = sorted(combined.items(),
                    key=lambda x: x[1], reverse=True)

    return [{'title': title, 'hybrid_score': round(score, 4)}
            for title, score in ranked[:n]]


# ══════════════════════════════════════════════════════════════
# MAIN — test everything
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("PHASE 2 — Loading datasets")
    print("=" * 50)

    ratings = load_ratings(sample=True)
    movies  = load_movies()
    tmdb    = load_tmdb()
    imdb    = load_imdb()

    print("\n" + "=" * 50)
    print("PHASE 2 — Content-Based Recommender")
    print("=" * 50)

    tmdb_clean, tfidf_matrix, tfidf, title_to_idx = \
        build_content_model(tmdb)

    for test_movie in ["Inception", "Toy Story", "The Dark Knight"]:
        print(f"\n--- Similar to '{test_movie}' ---")
        recs = get_content_recommendations(
            test_movie, tmdb_clean, tfidf_matrix, title_to_idx, n=5
        )
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['title']:<40} "
                  f"| {r['genre_names']:<30} "
                  f"| score: {r['similarity_score']}")

    print("\n" + "=" * 50)
    print("PHASE 2 — Collaborative Filtering")
    print("=" * 50)

    user_movie_matrix, ratings_filtered = build_collab_model(ratings)
    test_user = int(ratings_filtered['userId'].iloc[0])
    print(f"\nRecommendations for user {test_user}:")

    collab_recs = get_collab_recommendations(
        test_user, user_movie_matrix, ratings_filtered, n=10
    )
    for i, r in enumerate(collab_recs, 1):
        title = movies[
            movies['movieId'] == r['movieId']
        ]['title'].values
        title = title[0] if len(title) > 0 else "Unknown"
        print(f"  {i}. {title:<45} | score: {r['score']}")

    print("\n" + "=" * 50)
    print("PHASE 2 — SVD Recommender")
    print("=" * 50)

    svd_data      = build_svd_model(ratings, n_factors=50)
    test_user_svd = svd_data['user_ids'][0]
    print(f"\nSVD Recommendations for user {test_user_svd}:")

    svd_recs = get_svd_recommendations(
        test_user_svd, svd_data, ratings, movies, n=10
    )
    for i, r in enumerate(svd_recs, 1):
        print(f"  {i}. {r['title']:<45} "
              f"| predicted rating: {r['predicted_rating']}")

    print("\n" + "=" * 50)
    print("PHASE 3 — Hybrid Ensemble")
    print("=" * 50)

    ensemble_user = svd_data['user_ids'][0]
    hybrid_recs   = hybrid_recommend(
        user_id           = ensemble_user,
        user_movie_matrix = user_movie_matrix,
        ratings_df        = ratings_filtered,
        movies_df         = movies,
        tmdb_df           = tmdb_clean,
        tfidf_matrix      = tfidf_matrix,
        title_to_idx      = title_to_idx,
        svd_data          = svd_data,
        alpha=0.4, beta=0.3, gamma=0.3,
        n=10
    )

    print(f"\nTop 10 Hybrid Recommendations for user {ensemble_user}:")
    print(f"{'Rank':<5} {'Title':<45} {'Hybrid Score'}")
    print("-" * 65)
    for i, r in enumerate(hybrid_recs, 1):
        print(f"  {i:<4} {r['title']:<45} {r['hybrid_score']}")