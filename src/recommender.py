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
from rapidfuzz import process, fuzz

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
    raise FileNotFoundError("ratings.csv not found")


def load_movies():
    for name in ["movies.csv", "movie.csv"]:
        path = DATA_RAW / "ml-25m" / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"✓ Movies loaded: {df.shape}")
            return df
    raise FileNotFoundError("movies.csv not found")


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

    # keywords
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
        print("⚠ keywords.csv not found")

    # credits
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
        print("⚠ credits.csv not found")

    print(f"✓ TMDB loaded: {df.shape}")
    return df


def load_imdb():
    path = DATA_RAW / "imdb" / "IMDB Dataset.csv"
    df   = pd.read_csv(path)
    print(f"✓ IMDB loaded: {df.shape}")
    print(f"  Sentiment:\n{df['sentiment'].value_counts()}")
    return df


# ══════════════════════════════════════════════════════════════
# 2. TITLE MATCHING — ID based, no duplicate issues
# ══════════════════════════════════════════════════════════════

def detect_language(title):
    if any(ord(c) > 127 for c in title):
        return 'hi'
    return None


def find_best_title_match(title, search_df, threshold=60):
    """
    Returns TMDB movie ID (not title string).
    Uses search_df which has one row per movie with
    id, title, release_date, genre_names, title_lower.

    When multiple movies share a title, picks the one
    with the highest vote_count (most popular / well-known).
    No user input prompts — fully automatic.
    """
    title_lower = title.lower().strip()

    # ── 1. exact match ────────────────────────────────────────
    exact = search_df[search_df['title_lower'] == title_lower]
    if len(exact) == 1:
        return int(exact.iloc[0]['id'])
    if len(exact) > 1:
        # pick most popular among duplicates
        best = exact.loc[exact['vote_count'].idxmax()]
        print(f"  → Multiple exact matches for '{title}', "
              f"picked most popular: '{best['title']}' "
              f"({str(best['release_date'])[:4]})")
        return int(best['id'])

    # ── 2. normalised match ───────────────────────────────────
    def normalise(t):
        t = re.sub(r'[^\w\s]', '', t.lower())
        t = re.sub(r'\b(the|a|an)\b', '', t)
        return re.sub(r'\s+', ' ', t).strip()

    norm_query = normalise(title_lower)
    search_df  = search_df.copy()
    search_df['title_norm'] = search_df['title_lower'].apply(normalise)
    norm_match = search_df[search_df['title_norm'] == norm_query]
    if len(norm_match) >= 1:
        best = norm_match.loc[norm_match['vote_count'].idxmax()]
        print(f"  → Normalised match: '{best['title']}'")
        return int(best['id'])

    # ── 3. fuzzy token sort ───────────────────────────────────
    all_titles  = search_df['title_lower'].tolist()
    match, score, _ = process.extractOne(
        title_lower, all_titles,
        scorer=fuzz.token_sort_ratio
    )
    if score >= threshold:
        matched_rows = search_df[search_df['title_lower'] == match]
        best = matched_rows.loc[matched_rows['vote_count'].idxmax()]
        print(f"  → Fuzzy matched '{title}' → "
              f"'{best['title']}' (score:{score})")
        return int(best['id'])

    # ── 4. partial ratio ──────────────────────────────────────
    match, score, _ = process.extractOne(
        title_lower, all_titles,
        scorer=fuzz.partial_ratio
    )
    if score >= threshold + 10:
        matched_rows = search_df[search_df['title_lower'] == match]
        best = matched_rows.loc[matched_rows['vote_count'].idxmax()]
        print(f"  → Partial matched '{title}' → "
              f"'{best['title']}' (score:{score})")
        return int(best['id'])

    print(f"  ✗ No match found for '{title}'")
    return None


# ══════════════════════════════════════════════════════════════
# 3. CONTENT-BASED FILTERING (TF-IDF)
# ══════════════════════════════════════════════════════════════

def build_content_model(tmdb_df):
    """
    Returns:
        df          — cleaned TMDB dataframe
        tfidf_matrix
        tfidf       — fitted vectorizer
        id_to_idx   — TMDB movie id → dataframe row index
        search_df   — lightweight search table for fuzzy matching
    """
    df = tmdb_df.copy().reset_index(drop=True)
    df['overview']      = df['overview'].fillna('')
    df['genre_names']   = df['genre_names'].fillna('')
    df['keyword_names'] = df['keyword_names'].fillna('') \
                          if 'keyword_names' in df.columns \
                          else pd.Series([''] * len(df))
    df['cast_names']    = df['cast_names'].fillna('') \
                          if 'cast_names' in df.columns \
                          else pd.Series([''] * len(df))
    df['director_name'] = df['director_name'].fillna('') \
                          if 'director_name' in df.columns \
                          else pd.Series([''] * len(df))
    df['vote_count']    = pd.to_numeric(
        df['vote_count'], errors='coerce'
    ).fillna(0)

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

    # ── ID based index — no duplicate title issues ─────────────
    id_to_idx = pd.Series(df.index, index=df['id'])

    # ── search table for fuzzy matching ───────────────────────
    search_df = df[['id', 'title', 'release_date',
                    'genre_names', 'vote_count']].copy()
    search_df['title_lower'] = (
        search_df['title'].fillna('').str.lower().str.strip()
    )

    return df, tfidf_matrix, tfidf, id_to_idx, search_df


def get_content_recommendations(title, tmdb_df, tfidf_matrix,
                                id_to_idx, search_df, n=10,
                                genre_filter=None,
                                lang_filter=None):
    # auto detect language
    if lang_filter is None:
        lang_filter = detect_language(title)
        if lang_filter:
            print(f"  → Language detected: {lang_filter}")

    # get TMDB movie ID via fuzzy match
    matched_id = find_best_title_match(title, search_df)
    if matched_id is None:
        return []

    # get dataframe index from ID
    if matched_id not in id_to_idx:
        print(f"  ✗ Movie ID {matched_id} not in index.")
        return []

    idx        = int(id_to_idx[matched_id])
    movie_vec  = tfidf_matrix[idx]
    sim_scores = cosine_similarity(movie_vec, tfidf_matrix).flatten()

    # popularity boost
    max_votes = tmdb_df['vote_count'].fillna(0).max()
    if max_votes > 0:
        pop_bonus  = (tmdb_df['vote_count'].fillna(0)
                      / max_votes).values * 0.35
        sim_scores = sim_scores * 0.65 + pop_bonus

    sim_indices = np.argsort(sim_scores)[::-1][1:n * 4 + 1]

    results = []
    for i in sim_indices:
        if len(results) >= n:
            break
        try:
            score  = float(sim_scores[i])
            if score < 0.10:
                continue
            row    = tmdb_df.iloc[int(i)]
            lang   = str(row.get('original_language', 'en'))
            genres = str(row['genre_names'])
            votes  = int(row.get('vote_count', 0) or 0)

            if votes < 100:
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
                'vote_count':        votes,
                'original_language': lang
            })
        except IndexError:
            continue

    return results


# ══════════════════════════════════════════════════════════════
# 4. COLLABORATIVE FILTERING
# ══════════════════════════════════════════════════════════════

def build_collab_model(ratings_df,
                       min_movie_ratings=50,
                       min_user_ratings=20):
    print("\nBuilding collaborative filtering model...")

    movie_counts   = ratings_df['movieId'].value_counts()
    popular_movies = movie_counts[
        movie_counts >= min_movie_ratings].index
    df = ratings_df[ratings_df['movieId'].isin(popular_movies)]

    user_counts  = df['userId'].value_counts()
    active_users = user_counts[
        user_counts >= min_user_ratings].index
    df = df[df['userId'].isin(active_users)]

    print(f"  After filtering: {df['userId'].nunique()} users, "
          f"{df['movieId'].nunique()} movies")

    print("  Building user-movie matrix (~30 seconds)...")
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

    user_vec   = user_movie_matrix.loc[user_id].values.reshape(1, -1)
    sim_scores = cosine_similarity(
        user_vec, user_movie_matrix.values
    ).flatten()

    sim_indices   = np.argsort(sim_scores)[::-1][1:21]
    similar_users = user_movie_matrix.index[sim_indices].tolist()

    seen_movies = set(
        ratings_df[ratings_df['userId'] == user_id]['movieId']
    )

    candidate_scores = {}
    for sim_user in similar_users:
        sim_ratings = ratings_df[
            (ratings_df['userId'] == sim_user) &
            (ratings_df['rating'] >= 4.0) &
            (~ratings_df['movieId'].isin(seen_movies))
        ]
        for _, row in sim_ratings.iterrows():
            mid = row['movieId']
            candidate_scores[mid] = (candidate_scores.get(mid, 0)
                                     + row['rating'])

    top_movies = sorted(candidate_scores.items(),
                        key=lambda x: x[1], reverse=True)[:n]
    return [{'movieId': mid, 'score': round(score, 2)}
            for mid, score in top_movies]


# ══════════════════════════════════════════════════════════════
# 5. SVD MATRIX FACTORIZATION
# ══════════════════════════════════════════════════════════════

def build_svd_model(ratings_df, n_factors=20):
    print("\nBuilding SVD model...")

    top_movies = ratings_df['movieId'].value_counts().head(500).index
    top_users  = ratings_df['userId'].value_counts().head(2000).index

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
    predicted_mat = np.dot(np.dot(U, np.diag(sigma)), Vt)
    print("✓ SVD complete!")

    svd_data = {
        'predicted_matrix': predicted_mat,
        'user_ids':         user_ids,
        'movie_ids':        movie_ids
    }
    with open("models/svd_model.pkl", "wb") as f:
        pickle.dump(svd_data, f)
    print("✓ SVD model saved")
    return svd_data


def get_svd_recommendations(user_id, svd_data, ratings_df,
                             movies_df, n=10):
    user_ids  = svd_data['user_ids']
    movie_ids = svd_data['movie_ids']
    pred_mat  = svd_data['predicted_matrix']

    if user_id not in user_ids:
        print(f"  ✗ User {user_id} not in SVD model.")
        return []

    user_preds  = pred_mat[user_ids.index(user_id)]
    seen_movies = set(
        ratings_df[ratings_df['userId'] == user_id]['movieId']
    )

    candidates = [
        (movie_ids[i], pred_rating)
        for i, pred_rating in enumerate(user_preds)
        if movie_ids[i] not in seen_movies
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    results = []
    for movie_id, pred_rating in candidates[:n]:
        title = movies_df[
            movies_df['movieId'] == movie_id
        ]['title'].values
        results.append({
            'movieId':          movie_id,
            'title':            title[0] if len(title) > 0 else "Unknown",
            'predicted_rating': round(float(pred_rating), 3)
        })
    return results


# ══════════════════════════════════════════════════════════════
# 6. WEIGHTED ENSEMBLE
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
            (r[score_key] - min_s) / rng, 4)
    return recs


def hybrid_recommend(user_id, user_movie_matrix, ratings_df,
                     movies_df, tmdb_df, tfidf_matrix,
                     id_to_idx, search_df, svd_data,
                     alpha=0.4, beta=0.3, gamma=0.3, n=10):
    print(f"\nGenerating hybrid recommendations for user {user_id}...")

    collab_recs = get_collab_recommendations(
        user_id, user_movie_matrix, ratings_df, n=50)
    svd_recs    = get_svd_recommendations(
        user_id, svd_data, ratings_df, movies_df, n=50)

    user_top = (
        ratings_df[ratings_df['userId'] == user_id]
        .sort_values('rating', ascending=False)
        .head(3)['movieId'].tolist()
    )

    content_recs = []
    for movie_id in user_top:
        title_row = movies_df[movies_df['movieId'] == movie_id]['title']
        if len(title_row) == 0:
            continue
        title = title_row.values[0]
        title = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
        if ', The' in title:
            title = 'The ' + title.replace(', The', '')
        if ', A ' in title:
            title = 'A ' + title.replace(', A ', ' ')
        recs = get_content_recommendations(
            title, tmdb_df, tfidf_matrix,
            id_to_idx, search_df, n=20
        )
        content_recs.extend(recs)

    collab_recs  = normalize_scores(collab_recs,  'score')
    svd_recs     = normalize_scores(svd_recs,      'predicted_rating')
    content_recs = normalize_scores(content_recs,  'similarity_score')

    combined = {}

    for r in collab_recs:
        t = movies_df[
            movies_df['movieId'] == r['movieId']
        ]['title'].values
        if len(t) == 0:
            continue
        combined[t[0]] = (combined.get(t[0], 0)
                          + alpha * r['normalized_score'])

    for r in svd_recs:
        combined[r['title']] = (combined.get(r['title'], 0)
                                + gamma * r['normalized_score'])

    for r in content_recs:
        combined[r['title']] = (combined.get(r['title'], 0)
                                + beta * r['normalized_score'])

    seen_ids = ratings_df[
        ratings_df['userId'] == user_id
    ]['movieId'].tolist()
    for mid in seen_ids:
        t = movies_df[movies_df['movieId'] == mid]['title'].values
        if len(t) > 0:
            combined.pop(t[0], None)

    ranked = sorted(combined.items(),
                    key=lambda x: x[1], reverse=True)
    return [{'title': t, 'hybrid_score': round(s, 4)}
            for t, s in ranked[:n]]


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ratings = load_ratings(sample=True)
    movies  = load_movies()
    tmdb    = load_tmdb()
    imdb    = load_imdb()

    tmdb_clean, tfidf_matrix, tfidf, id_to_idx, search_df = \
        build_content_model(tmdb)

    for test in ["Inception", "Moana", "Golmaal",
                 "Pirates of Caribbean", "Toy Story", "Batman"]:
        print(f"\n--- Similar to '{test}' ---")
        recs = get_content_recommendations(
            test, tmdb_clean, tfidf_matrix,
            id_to_idx, search_df, n=5
        )
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['title']:<40} "
                  f"| score: {r['similarity_score']}")