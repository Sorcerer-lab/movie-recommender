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


def _clean_movielens_title(title):
    """
    Convert a MovieLens title to a clean search string for TMDB lookup.

    MovieLens encodes titles like:
        "Spirited Away (Sen to Chihiro no kamikakushi) (2001)"
        "Dark Knight, The (2008)"
        "Matrix, The (1999)"

    We want:
        "Spirited Away"      ← strip ALL parentheticals, not just year
        "The Dark Knight"    ← flip trailing article
        "The Matrix"
    """
    # strip every parenthetical — year AND foreign subtitles
    title = re.sub(r'\s*\([^)]*\)', '', title).strip()
    # flip trailing article: "Knight, The" → "The Knight"
    title = re.sub(r',\s*(The|A|An)\s*$', r'\1 ' +
                   title.split(',')[0].strip(),
                   title, flags=re.IGNORECASE)
    # simpler rewrite that actually works:
    m = re.match(r'^(.*),\s*(The|A|An)\s*$', title, re.IGNORECASE)
    if m:
        title = m.group(2) + ' ' + m.group(1)
    return title.strip()


def _normalise_for_match(t):
    """Strip punctuation, articles, and extra spaces for fuzzy matching."""
    t = re.sub(r'[^\w\s]', '', t.lower())
    t = re.sub(r'\b(the|a|an)\b', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def find_best_title_match(title, search_df, threshold=60):
    """
    Returns TMDB movie ID (not title string).

    Match pipeline:
      1. Exact match on title_lower
      2. Normalised match (strip punctuation + articles)
      3. WRatio fuzzy match at score >= 88
         WRatio handles substring / transposition cases better than
         token_sort_ratio, and is safer than partial_ratio which
         matched "Ed" inside "Spirited Away" at score 100.
      4. Return None — never returns a wrong match.
    """
    title_lower = title.lower().strip()

    # ── 1. exact match ────────────────────────────────────────
    exact = search_df[search_df['title_lower'] == title_lower]
    if len(exact) == 1:
        return int(exact.iloc[0]['id'])
    if len(exact) > 1:
        best = exact.loc[exact['vote_count'].idxmax()]
        print(f"  → Multiple exact matches for '{title}', "
              f"picked most popular: '{best['title']}' "
              f"({str(best['release_date'])[:4]})")
        return int(best['id'])

    # ── 2. normalised match ───────────────────────────────────
    norm_query  = _normalise_for_match(title_lower)
    search_copy = search_df.copy()
    search_copy['title_norm'] = search_copy['title_lower'].apply(
        _normalise_for_match
    )
    norm_match = search_copy[search_copy['title_norm'] == norm_query]
    if len(norm_match) >= 1:
        best = norm_match.loc[norm_match['vote_count'].idxmax()]
        print(f"  → Normalised match: '{best['title']}'")
        return int(best['id'])

    # ── 3. WRatio fuzzy match ─────────────────────────────────
    # WRatio combines token_set, token_sort, and partial_ratio
    # internally but applies length penalties that prevent short
    # strings like "Ed" from matching long titles.
    # Threshold 88 = strong match required; returns None otherwise.
    all_titles = search_df['title_lower'].tolist()
    match, score, _ = process.extractOne(
        title_lower, all_titles,
        scorer=fuzz.WRatio
    )
    if score >= 88:
        matched_rows = search_df[search_df['title_lower'] == match]
        best = matched_rows.loc[matched_rows['vote_count'].idxmax()]
        print(f"  → Fuzzy matched '{title}' → "
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

    df['soup'] = (
        df['original_language'] + ' ' +
        df['original_language'] + ' ' +
        df['original_language'] + ' ' +
        df['genre_names']       + ' ' +
        df['genre_names']       + ' ' +
        df['genre_names']       + ' ' +
        df['genre_names']       + ' ' +
        df['genre_names']       + ' ' +
        df['genre_names']       + ' ' +
        df['overview']          + ' ' +
        df['overview']          + ' ' +
        df['keyword_names']     + ' ' +
        df['keyword_names']     + ' ' +
        df['keyword_names']     + ' ' +
        df['keyword_names']     + ' ' +
        df['keyword_names']     + ' ' +
        df['director_name']     + ' ' +
        df['director_name']     + ' ' +
        df['director_name']     + ' ' +
        df['cast_names']        + ' ' +
        df['cast_names']        + ' ' +
        df['cast_names']        + ' ' +
        df['cast_names']
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
                'tmdb_id':           int(row['id']),
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

def build_svd_model(ratings_df, n_factors=100):
    """
    Trains SVD on the FULL dataset without ever building a dense matrix.

    Design:
    - Builds a sparse CSR matrix directly from ratings (no pivot_table).
    - Runs svds(..., k=100) on the sparse matrix — memory is O(nnz),
      not O(users × movies).
    - Saves only U (users × k), sigma (k,), Vt (k × movies), and ID
      mappings. No predicted_matrix is stored.
    - Recommendations are generated on-demand per user in O(k × movies)
      which is fast (~45K movies × 100 factors = 4.5M ops, <1ms).

    Memory estimate for full ML-25M:
        CSR matrix  : ~600 MB  (26M non-zeros × float32 + indices)
        U           : ~108 MB  (270K × 100 × float32)
        Vt          : ~18 MB   (100 × 45K × float32)
        Total saved : ~130 MB  (U + Vt + sigma + mappings)

    Delete models/svd_model.pkl to retrain.
    """
    svd_path = Path("models/svd_model.pkl")
    if svd_path.exists():
        with open(svd_path, "rb") as f:
            data = pickle.load(f)
        print(f"✓ SVD loaded — "
              f"{len(data['user_ids']):,} users  "
              f"{len(data['movie_ids']):,} movies  "
              f"k={len(data['sigma'])}")
        return data

    print("\nBuilding SVD on full dataset (sparse, no dense matrix)...")
    print(f"  Ratings: {len(ratings_df):,}")

    # ── 1. build integer indices for users and movies ──────────
    user_ids  = sorted(ratings_df['userId'].unique())
    movie_ids = sorted(ratings_df['movieId'].unique())
    user_idx  = {u: i for i, u in enumerate(user_ids)}
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    n_users  = len(user_ids)
    n_movies = len(movie_ids)
    print(f"  Users: {n_users:,}  Movies: {n_movies:,}")

    # ── 2. mean-centre each user's ratings ────────────────────
    # compute per-user mean from raw ratings — no matrix needed
    user_mean_series = (
        ratings_df.groupby('userId')['rating'].mean()
    )
    user_means = np.array(
        [user_mean_series.get(u, 0.0) for u in user_ids],
        dtype=np.float32
    )

    # ── 3. build sparse CSR matrix (centred) ──────────────────
    print("  Building sparse CSR matrix...")
    rows    = ratings_df['userId'].map(user_idx).values
    cols    = ratings_df['movieId'].map(movie_idx).values
    vals    = (ratings_df['rating'].values.astype(np.float32)
               - user_means[rows])

    sparse_mat = csr_matrix(
        (vals, (rows, cols)),
        shape=(n_users, n_movies),
        dtype=np.float32
    )
    print(f"  Sparse matrix: {sparse_mat.shape}  "
          f"nnz={sparse_mat.nnz:,}")

    # ── 4. run sparse SVD ─────────────────────────────────────
    k = min(n_factors, min(n_users, n_movies) - 1)
    print(f"  Running svds(k={k}) — this takes ~5–15 min on CPU, "
          f"~1–2 min on Colab T4...")

    U, sigma, Vt = svds(sparse_mat, k=k)

    # svds returns singular values in ascending order — reverse all
    order  = np.argsort(sigma)[::-1]
    sigma  = sigma[order]
    U      = U[:, order]
    Vt     = Vt[order, :]

    print(f"  Top 5 singular values: {sigma[:5].round(1)}")

    # ── 5. save decomposition only — no predicted_matrix ──────
    svd_data = {
        'U':          U.astype(np.float32),      # (n_users,  k)
        'sigma':      sigma.astype(np.float32),  # (k,)
        'Vt':         Vt.astype(np.float32),     # (k, n_movies)
        'user_ids':   user_ids,
        'movie_ids':  movie_ids,
        'user_idx':   user_idx,
        'movie_idx':  movie_idx,
        'user_means': user_means,                # (n_users,)
    }

    Path("models").mkdir(exist_ok=True)
    with open(svd_path, "wb") as f:
        pickle.dump(svd_data, f)

    import os
    size_mb = os.path.getsize(svd_path) / 1e6
    print(f"✓ SVD saved to {svd_path} ({size_mb:.0f} MB)")
    return svd_data


def get_svd_recommendations(user_id, svd_data, ratings_df,
                             movies_df, n=10):
    """
    On-demand prediction for one user.

    Known user  : u_vec = U[i] @ diag(sigma), then dot with Vt.
    Unknown user: project their rating vector into latent space
                  using Vt and sigma (same algebra, no stored U row).

    Memory: O(k × n_movies) per call — never loads a full matrix.
    """
    user_ids   = svd_data['user_ids']
    movie_ids  = svd_data['movie_ids']
    user_idx   = svd_data['user_idx']
    movie_idx  = svd_data['movie_idx']
    user_means = svd_data['user_means']
    U          = svd_data['U']
    sigma      = svd_data['sigma']
    Vt         = svd_data['Vt']

    seen_movies = set(
        ratings_df[ratings_df['userId'] == user_id]['movieId']
    )

    if user_id in user_idx:
        # ── known user: retrieve stored U row ─────────────────
        i        = user_idx[user_id]
        u_mean   = float(user_means[i])
        # predicted = (U[i] * sigma) @ Vt  +  user_mean
        u_vec    = U[i] * sigma             # (k,)
        preds    = u_vec @ Vt + u_mean      # (n_movies,)

    else:
        # ── unknown user: project from their ratings ───────────
        user_ratings = (
            ratings_df[ratings_df['userId'] == user_id]
            .set_index('movieId')['rating']
        )
        if user_ratings.empty:
            print(f"  ✗ User {user_id} is cold-start — no ratings.")
            return []

        u_mean  = float(user_ratings.mean())
        r_vec   = np.zeros(len(movie_ids), dtype=np.float32)
        for mid, rating in user_ratings.items():
            if mid in movie_idx:
                r_vec[movie_idx[mid]] = rating - u_mean

        # project: u_latent = r_vec @ Vt.T / sigma
        sigma_inv = np.where(sigma > 1e-10, 1.0 / sigma, 0.0)
        u_vec     = (r_vec @ Vt.T) * sigma_inv   # (k,)
        preds     = u_vec @ Vt + u_mean           # (n_movies,)
        print(f"  → User {user_id} projected into SVD space")

    preds = np.clip(preds, 0.5, 5.0)

    # ── rank unseen movies ─────────────────────────────────────
    candidates = [
        (movie_ids[i], float(preds[i]))
        for i in range(len(movie_ids))
        if movie_ids[i] not in seen_movies
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    # ── build result list ──────────────────────────────────────
    mid_to_title = (
        movies_df.set_index('movieId')['title'].to_dict()
    )
    return [
        {
            'movieId':          mid,
            'title':            mid_to_title.get(mid, 'Unknown'),
            'predicted_rating': round(score, 3)
        }
        for mid, score in candidates[:n]
    ]


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
        
    svd_recs = get_svd_recommendations(
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
        # Use your global robust cleaning function instead of inline regexes
        cleaned_title = _clean_movielens_title(title)
        
        recs = get_content_recommendations(
            cleaned_title, tmdb_df, tfidf_matrix,
            id_to_idx, search_df, n=20
        )
        content_recs.extend(recs)

    collab_recs  = normalize_scores(collab_recs,  'score')
    svd_recs     = normalize_scores(svd_recs,      'predicted_rating')
    content_recs = normalize_scores(content_recs,  'similarity_score')

    # Global normalization utility targeting both frameworks 
    def _global_norm(t):
        t = str(t).lower().strip()
        t = re.sub(r'\s*\(\d{4}\)\s*$', '', t)
        t = re.sub(r'[^\w\s]', '', t)
        return t.strip()

    movies_norm = {
        _global_norm(_clean_movielens_title(t)): int(mid)
        for t, mid in zip(movies_df['title'], movies_df['movieId'])
    }

    tmdb_to_ml = {}
    for _, row in tmdb_df.iterrows():
        norm_tmdb_title = _global_norm(str(row['title']))
        mid = movies_norm.get(norm_tmdb_title)
        if mid is not None:
            tmdb_to_ml[int(row['id'])] = int(mid)

    combined = {}  # movieId (int) -> score

    # ── 1. Collaborative Filtering Contributions ──
    for r in collab_recs:
        mid = int(r['movieId'])
        combined[mid] = combined.get(mid, 0.0) + alpha * r.get('normalized_score', 0.0)

    # ── 2. Matrix Factorization (SVD) Contributions ──
    for r in svd_recs:
        mid = int(r['movieId'])
        combined[mid] = combined.get(mid, 0.0) + gamma * r.get('normalized_score', 0.0)

    # ── 3. Content Filtering Contributions ──
    for r in content_recs:
        mid = tmdb_to_ml.get(r.get('tmdb_id')) 
        if mid is None:
            mid = movies_norm.get(_global_norm(r['title']))
        if mid is None:
            continue
            
        mid = int(mid)
        combined[mid] = combined.get(mid, 0.0) + beta * r.get('normalized_score', 0.0)

    # ── Filter out user history ──
    seen_ids = set(int(x) for x in ratings_df[ratings_df['userId'] == user_id]['movieId'])
    for mid in seen_ids:
        combined.pop(mid, None)

    movie_lookup = movies_df.set_index('movieId')['title'].to_dict()
    # Mild inverse popularity penalty to reduce popularity bias
    movie_popularity = (
       ratings_df['movieId'].value_counts().to_dict()
)
    max_pop = max(movie_popularity.values()) if movie_popularity else 1
    for mid in list(combined.keys()):
      pop = movie_popularity.get(mid, 0)
      pop_penalty = 0.15 * (pop / max_pop)   # penalise up to 15%
      combined[mid] = max(0.0, combined[mid] - pop_penalty)
      ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    print(f"  contributions → collab:{len(collab_recs)}"
          f"  svd:{len(svd_recs)}"
          f"  content:{len(content_recs)}"
          f"  pool:{len(combined)}")

    return [
        {
            'movieId':      int(mid),
            'title':        movie_lookup.get(mid, 'Unknown Title'),
            'hybrid_score': round(float(score), 4)
        }
        for mid, score in ranked[:n]
    ]
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