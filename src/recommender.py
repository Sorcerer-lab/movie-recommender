import pandas as pd
import numpy as np
from collections import defaultdict
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
    Loads pretrained TF-IDF from disk if available.
    Only builds from scratch if no saved model exists.
    """
    import scipy.sparse

    tmdb_pkl = Path("models/tmdb_clean.pkl")

    if tmdb_pkl.exists():
        print("Loading pre-trained TF-IDF from disk...")
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
            search_df['title'].fillna('').str.lower().str.strip()
        )
        print(f"✓ Pre-trained TF-IDF loaded "
              f"({tmdb_clean.shape[0]} movies)")
        return tmdb_clean, tfidf_matrix, tfidf, id_to_idx, search_df

    # ── no pretrained model — build from scratch ──────────────
    print("Building TF-IDF from scratch...")

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

    tmdb['soup'] = (
    tmdb['original_language']   + ' ' +
    tmdb['original_language']   + ' ' +
    tmdb['original_language']   + ' ' +
    tmdb['genre_names']   + ' ' +
    tmdb['genre_names']   + ' ' +
    tmdb['genre_names']   + ' ' +
    tmdb['genre_names']   + ' ' +
    tmdb['genre_names']   + ' ' +
    tmdb['genre_names']   + ' ' +
    tmdb['overview']   + ' ' +
    tmdb['overview']   + ' ' +
    tmdb['keyword_names'] + ' ' +
    tmdb['keyword_names'] + ' ' +
    tmdb['keyword_names'] + ' ' +
    tmdb['keyword_names'] + ' ' +
    tmdb['keyword_names'] + ' ' +
    tmdb['director_name'] + ' ' +
    tmdb['director_name'] + ' ' +
    tmdb['director_name'] + ' ' +
    tmdb['cast_names']    + ' ' +
    tmdb['cast_names']    + ' ' +
    tmdb['cast_names']    + ' ' +
    tmdb['cast_names']
)

    df['soup_len'] = df['soup'].str.split().str.len()
    df = df[df['soup_len'] >= 20].reset_index(drop=True)
    print(f"  After filtering: {len(df)} movies remaining")

    print("Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=15000,
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    print(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")

    # save for future runs
    df.to_pickle("models/tmdb_clean.pkl")
    scipy.sparse.save_npz("models/tfidf_matrix.npz", tfidf_matrix)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("✓ TF-IDF saved to disk")

    id_to_idx = pd.Series(df.index, index=df['id'])
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

    # small popularity nudge — keeps obscure films from dominating
    # but does NOT override genuine content similarity
    max_votes = tmdb_df['vote_count'].fillna(0).max()
    if max_votes > 0:
        pop_bonus  = (tmdb_df['vote_count'].fillna(0)
                      / max_votes).values * 0.10
        sim_scores = sim_scores * 0.90 + pop_bonus

    sim_indices = np.argsort(sim_scores)[::-1][1:n * 8 + 1]

    results    = []
    seen_titles = set()  # deduplicate by normalised title

    for i in sim_indices:
        if len(results) >= n:
            break
        try:
            score  = float(sim_scores[i])
            if score < 0.15:          # raised from 0.10 — filters weak matches
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

            # deduplicate: normalise title to lower-stripped form
            norm_title = str(row['title']).lower().strip()
            if norm_title in seen_titles:
                continue
            seen_titles.add(norm_title)

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

def train_cluster_model(ratings_df, n_clusters=100):
    """
    Train user clusters locally.
    Groups users with similar taste into clusters.
    CF then only compares within the same cluster
    instead of against all users — faster and more accurate.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    print("\nTraining user clusters locally...")

    top_movies = ratings_df['movieId'].value_counts().head(3000).index
    top_users  = ratings_df['userId'].value_counts().head(50000).index

    filtered = ratings_df[
        ratings_df['movieId'].isin(top_movies) &
        ratings_df['userId'].isin(top_users)
    ]

    print(f"  {filtered['userId'].nunique()} users x "
          f"{filtered['movieId'].nunique()} movies")

    user_movie = filtered.pivot_table(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)

    user_ids = list(user_movie.index)
    mat      = normalize(
        user_movie.values.astype(np.float32), norm='l2'
    )

    print(f"  Clustering into {n_clusters} groups...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42,
        batch_size=2000, n_init=10
    )
    cluster_labels = kmeans.fit_predict(mat)

    user_cluster  = {
        user_ids[i]: int(cluster_labels[i])
        for i in range(len(user_ids))
    }
    cluster_users = {}
    for uid, cid in user_cluster.items():
        cluster_users.setdefault(cid, []).append(uid)

    cluster_data = {
        'user_cluster':  user_cluster,
        'cluster_users': cluster_users,
        'user_ids':      user_ids,
        'movie_ids':     list(user_movie.columns),
        'n_clusters':    n_clusters,
        'kmeans':        kmeans
    }

    with open("models/cluster_data.pkl", "wb") as f:
        pickle.dump(cluster_data, f)

    sizes = [len(v) for v in cluster_users.values()]
    print(f"✓ Clusters saved — "
          f"min={min(sizes)} max={max(sizes)} "
          f"mean={int(np.mean(sizes))} users per cluster")

    return cluster_data


def build_clustered_collab_model(ratings_df):
    """
    Load pre-trained clusters if available.
    Auto-trains if not found.
    """
    cluster_path = Path("models/cluster_data.pkl")
    if cluster_path.exists():
        with open(cluster_path, "rb") as f:
            cluster_data = pickle.load(f)
        print(f"✓ Cluster model loaded — "
              f"{cluster_data['n_clusters']} clusters, "
              f"{len(cluster_data['user_cluster'])} users")
        return cluster_data
    else:
        print("  No cluster model found — training now...")
        return train_cluster_model(ratings_df)

def get_user_cluster(user_id, cluster_data, ratings_df):
    """
    Get cluster for any user — known or unknown.
    For unknown users: build their rating vector
    and assign to nearest cluster centroid.
    """
    user_cluster  = cluster_data['user_cluster']
    cluster_users = cluster_data['cluster_users']

    # known user — direct lookup
    if user_id in user_cluster:
        return user_cluster[user_id]

    # unknown user — find nearest cluster
    kmeans    = cluster_data.get('kmeans')
    movie_ids = cluster_data['movie_ids']

    if kmeans is None:
        # fallback to largest cluster
        return max(cluster_users, key=lambda c: len(cluster_users[c]))

    from sklearn.preprocessing import normalize
     # build this user's rating vector
    user_ratings = ratings_df[
        ratings_df['userId'] == user_id
    ].set_index('movieId')['rating']

    user_vec = np.array([
        user_ratings.get(mid, 0.0) for mid in movie_ids
    ], dtype=np.float32).reshape(1, -1)

    user_vec_norm = normalize(user_vec, norm='l2')
    cluster_id    = int(kmeans.predict(user_vec_norm)[0])

    print(f"  → Unknown user {user_id} assigned to "
          f"cluster {cluster_id}")
    return cluster_id

def get_clustered_collab_recommendations(user_id, cluster_data,
                                          ratings_df, n=10):
    cluster_users = cluster_data['cluster_users']

    # get cluster — works for both known and unknown users
    cluster_id    = get_user_cluster(
        user_id, cluster_data, ratings_df
    )
    similar_users = cluster_users.get(cluster_id, [])

    print(f"  → User {user_id} cluster {cluster_id} "
          f"({len(similar_users)} similar users)")

    seen_movies = set(
        ratings_df[ratings_df['userId'] == user_id]['movieId']
    )

    candidate_scores = defaultdict(float)
    candidate_counts = defaultdict(int)

    for sim_user in similar_users:
        if sim_user == user_id:
            continue
        sim_ratings = ratings_df[
            (ratings_df['userId'] == sim_user) &
            (ratings_df['rating'] >= 4.0) &
            (~ratings_df['movieId'].isin(seen_movies))
        ]
        for _, row in sim_ratings.iterrows():
            mid = row['movieId']
            candidate_scores[mid] += row['rating']
            candidate_counts[mid] += 1

    # weight by avg rating × log(number of recommenders)
    weighted = {
        mid: (candidate_scores[mid] / candidate_counts[mid])
             * np.log1p(candidate_counts[mid])
        for mid in candidate_scores
    }

    top = sorted(weighted.items(),
                 key=lambda x: x[1], reverse=True)[:n]

    return [{'movieId': mid, 'score': round(score, 4)}
            for mid, score in top]

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

def build_svd_model(ratings_df, n_factors=50):
    """
    Loads pretrained SVD from disk if available.
    Only trains from scratch if no saved model exists.
    """
    svd_path = Path("models/svd_model.pkl")
    if svd_path.exists():
        with open(svd_path, "rb") as f:
            data = pickle.load(f)
        print("✓ Loaded pre-trained SVD from disk")
        return data

    print("\nBuilding SVD model from scratch...")

    top_movies = ratings_df['movieId'].value_counts().head(1000).index
    top_users  = ratings_df['userId'].value_counts().head(3000).index

    filtered = ratings_df[
        ratings_df['movieId'].isin(top_movies) &
        ratings_df['userId'].isin(top_users)
    ]

    print(f"  Filtered: {filtered['userId'].nunique()} users "
          f"x {filtered['movieId'].nunique()} movies")

    user_movie = filtered.pivot_table(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)

    user_ids  = list(user_movie.index)
    movie_ids = list(user_movie.columns)
    mat       = user_movie.values.astype(np.float32)

    # mean center
    user_means = np.true_divide(
        mat.sum(axis=1),
        (mat != 0).sum(axis=1).clip(min=1)
    )
    mat_centered = mat.copy()
    for i, mean in enumerate(user_means):
        mat_centered[i][mat[i] != 0] -= mean

    sparse_mat    = csr_matrix(mat_centered)
    k             = min(n_factors, min(sparse_mat.shape) - 1)
    print(f"  Running SVD with {k} factors...")

    U, sigma, Vt  = svds(sparse_mat, k=k)
    predicted_mat = np.dot(np.dot(U, np.diag(sigma)), Vt)
    predicted_mat += user_means.reshape(-1, 1)
    predicted_mat  = np.clip(predicted_mat, 0.5, 5.0)

    svd_data = {
        'predicted_matrix': predicted_mat,
        'user_ids':         user_ids,
        'movie_ids':        movie_ids
    }
    with open(svd_path, "wb") as f:
        pickle.dump(svd_data, f)
    print("✓ SVD saved")
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

    # use clustered CF if available, else item-based CF
    # use clustered CF if available — load once, reuse
    # _cluster_cache avoids reloading pkl on every call
    if not hasattr(hybrid_recommend, '_cluster_cache'):
        hybrid_recommend._cluster_cache = \
            build_clustered_collab_model(ratings_df)

    cluster_data = hybrid_recommend._cluster_cache

    if cluster_data:
        collab_recs = get_clustered_collab_recommendations(
            user_id, cluster_data, ratings_df, n=50
        )
    else:
        collab_recs = get_collab_recommendations(
            user_id, user_movie_matrix, ratings_df, n=50
        )
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

    print("Collab:", collab_recs[:3])
    print("SVD:", svd_recs[:3])
    print("Content:", content_recs[:3])

    combined = {}  # movieId -> score

    for r in collab_recs:
        mid = r['movieId']
        combined[mid] = (
            combined.get(mid, 0)
            + alpha * r['normalized_score']
        )

    for r in svd_recs:
        row = movies_df[movies_df['title'] == r['title']]
        if len(row) == 0:
            continue
        mid = int(row.iloc[0]['movieId'])
        combined[mid] = (
            combined.get(mid, 0)
            + gamma * r['normalized_score']
        )

    for r in content_recs:
        row = movies_df[movies_df['title'] == r['title']]
        if len(row) == 0:
            continue
        mid = int(row.iloc[0]['movieId'])
        combined[mid] = (
            combined.get(mid, 0)
            + beta * r['normalized_score']
        )

    seen_ids = set(
        ratings_df[ratings_df['userId'] == user_id]['movieId']
    )
    for mid in seen_ids:
        combined.pop(mid, None)

    movie_lookup = (
        movies_df
        .set_index('movieId')['title']
        .to_dict()
    )

    ranked = sorted(
        combined.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for mid, score in ranked[:n]:
        results.append({
            'movieId':     int(mid),
            'title':       movie_lookup.get(mid, ''),
            'hybrid_score': round(score, 4)
        })
    return results


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