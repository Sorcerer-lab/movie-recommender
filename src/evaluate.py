import pandas as pd
import numpy as np
import sys
import pickle
import scipy.sparse
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from src.recommender import (
    load_ratings, load_movies, load_tmdb,
    build_content_model, build_collab_model,
    build_svd_model, get_content_recommendations,
    get_collab_recommendations, get_svd_recommendations,
    get_clustered_collab_recommendations,
    build_clustered_collab_model,
    hybrid_recommend
)


# ══════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════

def precision_at_k(recommended, relevant, k=10):
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / k

def average_precision_at_k(
    recommended,
    relevant,
    k=10
):
    score = 0.0
    hits = 0

    for i, item in enumerate(recommended[:k], 1):
        if item in relevant:
            hits += 1
            score += hits / i

    if len(relevant) == 0:
        return 0.0

    return score / min(len(relevant), k)

def recall_at_k(recommended, relevant, k=10):
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    dcg  = 0.0
    for i, item in enumerate(recommended[:k], 1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1)
               for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(recommended, relevant, k=10):
    return int(
        len(set(recommended[:k]) & relevant) > 0
    )

def mrr_at_k(
    recommended,
    relevant,
    k=10
):
    for rank, item in enumerate(
        recommended[:k],
        1
    ):
        if item in relevant:
            return 1.0 / rank

    return 0.0


import re
from itertools import combinations


def normalize_title(title):
    """
    Normalise a movie title for cross-dataset matching.
    Handles the two common MovieLens <-> TMDB mismatches:
      "Toy Story (1995)"   -> "toy story"
      "Dark Knight, The"   -> "dark knight"
      "The Dark Knight"    -> "dark knight"
    """
    title = str(title)
    title = re.sub(r'\s*\(\d{4}\)\s*$', '', title)
    title = re.sub(r',\s*(The|A|An)\s*$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^(The|A|An)\s+', '', title, flags=re.IGNORECASE)
    return title.strip().lower()


def _build_tmdb_norm_index(tmdb_df):
    """Pre-compute normalised-title -> integer iloc position once."""
    return {normalize_title(t): i
            for i, t in enumerate(tmdb_df['title'])}


def diversity_score(recs, tmdb_df, _norm_idx=None):
    """
    Intra-list diversity: 1 - mean pairwise Jaccard genre similarity.
    Uses normalised title matching so MovieLens/TMDB format differences
    no longer silently return 0 matches (root cause of Avg Diversity=0.0).
    """
    if _norm_idx is None:
        _norm_idx = _build_tmdb_norm_index(tmdb_df)

    genre_sets = []
    for r in recs:
        i = _norm_idx.get(normalize_title(r['title']))
        if i is None:
            continue
        genres = set(str(tmdb_df.iloc[i]['genre_names']).split())
        if genres:
            genre_sets.append(genres)

    if len(genre_sets) < 2:
        return 0.0

    similarities = []
    for g1, g2 in combinations(genre_sets, 2):
        union = g1 | g2
        if union:
            similarities.append(len(g1 & g2) / len(union))

    if not similarities:
        return 0.0

    return round(1 - float(np.mean(similarities)), 4)


# ══════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════

def split_user_ratings(ratings_df, test_ratio=0.2,
                        min_ratings=20):
    """
    Each user's most recent 20% of ratings = test set.
    Older 80% = train set.
    Simulates real usage: train on history, predict future.
    """
    train_list = []
    test_list  = []

    users = ratings_df['userId'].value_counts()
    users = users[users >= min_ratings].index

    for user_id in users:
        user_df = ratings_df[
            ratings_df['userId'] == user_id
        ].sort_values('timestamp')

        n_test = max(1, int(len(user_df) * test_ratio))
        train_list.append(user_df.iloc[:-n_test])
        test_list.append(user_df.iloc[-n_test:])

    train = pd.concat(train_list).reset_index(drop=True)
    test  = pd.concat(test_list).reset_index(drop=True)

    print(f"✓ Train: {len(train)} | Test: {len(test)} ratings")
    return train, test


# ══════════════════════════════════════════════════════════════
# EVALUATE CLUSTERED COLLAB ONLY
# ══════════════════════════════════════════════════════════════

def evaluate_clustered_collab(train_ratings, test_ratings,
                               movies_df, n_users=50, k=10):
    """
    Evaluate the clustered collaborative filter specifically.
    Uses cluster-based recommendations, not old user-based CF.
    """
    print(f"\nEvaluating Clustered CF (n_users={n_users}, k={k})...")

    cluster_data = build_clustered_collab_model(train_ratings)

    # pick test users who are IN the cluster model
    user_cluster = cluster_data['user_cluster']
    all_test_users = test_ratings['userId'].unique()
    known_users = [u for u in all_test_users if u in user_cluster]

    print(f"  Known test users: {len(known_users)} / "
          f"{len(all_test_users)}")

    rng = np.random.default_rng(42)

    if len(known_users) > n_users:
      test_users = rng.choice(
        known_users,
        size=n_users,
        replace=False
    )
    else:
       test_users = known_users

    precisions = []
    recalls    = []
    ndcgs      = []
    hit_rates  = []
    mrrs       = []
    maps       = []
    skipped    = 0

    for user_id in test_users:
        relevant = set(
            test_ratings[
                (test_ratings['userId'] == user_id) &
                (test_ratings['rating'] >= 3.0)
            ]['movieId'].tolist()
        )
        if not relevant:
            skipped += 1
            continue

        recs    = get_clustered_collab_recommendations(
            user_id, cluster_data, train_ratings, n=k * 2
        )
        rec_ids = [r['movieId'] for r in recs]

        if not rec_ids:
            skipped += 1
            continue

        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))
        hit_rates.append(hit_rate_at_k(rec_ids, relevant, k))
        maps.append(average_precision_at_k(rec_ids, relevant, k))
        mrrs.append(mrr_at_k(rec_ids, relevant, k))
    print(f"  Evaluated: {len(precisions)} | "
          f"Skipped: {skipped}")

    return {
        f'Precision@{k}':  round(np.mean(precisions), 4)
                           if precisions else 0,
        f'Recall@{k}':     round(np.mean(recalls), 4)
                           if recalls else 0,
        f'Hit Rate@{k}':   round(np.mean(hit_rates), 4)
                           if hit_rates else 0,
        f'MAP@{k}':        round(np.mean(maps), 4)
                           if maps else 0,
        f'NDCG@{k}':       round(np.mean(ndcgs), 4)
                           if ndcgs else 0,
        f'MRR@{k}':        round(np.mean(mrrs), 4)
                           if mrrs else 0,
        'Users evaluated': len(precisions)
    }


# ══════════════════════════════════════════════════════════════
# EVALUATE FULL HYBRID SYSTEM — what users actually experience
# ══════════════════════════════════════════════════════════════

def evaluate_hybrid(train_ratings, test_ratings,
                    movies_df, tmdb_df, tfidf_matrix,
                    id_to_idx, search_df, svd_data,
                    user_movie_matrix,
                    n_users=30, k=10):
    """
    Evaluate the FULL hybrid pipeline cleanly in a single pass.
    """
    print(f"\nEvaluating Full Hybrid System (n_users={n_users}, k={k})...")

    cluster_data   = build_clustered_collab_model(train_ratings)
    user_cluster   = cluster_data['user_cluster']
    all_test_users = test_ratings['userId'].unique()

    known = [u for u in all_test_users if u in user_cluster]
    rng = np.random.default_rng(42)

    if len(known) > n_users:
        test_users = rng.choice(known, size=n_users, replace=False)
    else:
        test_users = known

    print(f"  Testing {len(test_users)} users...")

    precisions      = []
    recalls         = []
    ndcgs           = []
    hit_rates       = []
    mrrs            = []
    maps            = []
    diversities     = []
    skipped         = 0
    all_recommended = set()
    
    from collections import Counter
    movie_freq = Counter()

    tmdb_norm_idx = _build_tmdb_norm_index(tmdb_df)
    mid_to_title = movies_df.set_index('movieId')['title'].to_dict()
    # Clear stale cache so it rebuilds from train_ratings only
    if hasattr(hybrid_recommend, '_cluster_cache'):
       del hybrid_recommend._cluster_cache
    for user_id in test_users:
        relevant = set(
            int(x) for x in test_ratings[
                (test_ratings['userId'] == user_id) &
                (test_ratings['rating'] >= 3.0)
            ]['movieId'].tolist()
        )
        if not relevant:
            skipped += 1
            continue

        try:
            # RUN RECOMMENDATION PIPELINE EXACTLY ONCE
            recs = hybrid_recommend(
                user_id           = user_id,
                user_movie_matrix = user_movie_matrix,
                ratings_df        = train_ratings,
                movies_df         = movies_df,
                tmdb_df           = tmdb_df,
                tfidf_matrix      = tfidf_matrix,
                id_to_idx         = id_to_idx,
                search_df         = search_df,
                svd_data          = svd_data,
                alpha=0.4, beta=0.3, gamma=0.3,
                n=k
            )
        except Exception as e:
            skipped += 1
            continue

        if not recs:
            skipped += 1
            continue

        # Force structural item representations to pure clean integers
        rec_ids = [int(r['movieId']) for r in recs]

        if not rec_ids:
            skipped += 1
            continue

        all_recommended.update(rec_ids[:k])
        movie_freq.update(rec_ids[:k])

        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))
        diversities.append(diversity_score(recs, tmdb_df, _norm_idx=tmdb_norm_idx))
        hit_rates.append(hit_rate_at_k(rec_ids, relevant, k))
        maps.append(average_precision_at_k(rec_ids, relevant, k))
        mrrs.append(mrr_at_k(rec_ids, relevant, k))

    print(f"  Evaluated: {len(precisions)} | Skipped: {skipped}")

    n_users_eval = max(len(precisions), 1)
    print(f"\n  ── Popularity bias (top 10 most recommended) ──")
    for mid, cnt in movie_freq.most_common(10):
        pct   = cnt / n_users_eval * 100
        title = mid_to_title.get(mid, f'movieId={mid}')
        print(f"    {title[:45]:<45}  {cnt}/{n_users_eval} users  ({pct:.0f}%)")
    print()

    coverage = len(all_recommended) / movies_df['movieId'].nunique() if all_recommended else 0.0

    return {
        f'Hybrid Precision@{k}':   round(np.mean(precisions), 4) if precisions else 0,
        f'Hybrid Recall@{k}':      round(np.mean(recalls), 4) if recalls else 0,
        f'Hybrid Hit Rate@{k}':    round(np.mean(hit_rates), 4) if hit_rates else 0,
        f'Hybrid NDCG@{k}':       round(np.mean(ndcgs), 4) if ndcgs else 0,
        'MAP@10':                  round(np.mean(maps), 4) if maps else 0,
        'MRR@10':                  round(np.mean(mrrs), 4) if mrrs else 0,
        'Avg Diversity':           round(np.mean(diversities), 4) if diversities else 0,
        'Coverage':                round(coverage, 4),
        'Users evaluated':         len(precisions)
    }
# ══════════════════════════════════════════════════════════════
# CONTENT EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_content(tmdb_df, tfidf_matrix,
                     id_to_idx, search_df, k=10):
    print(f"\nEvaluating Content-Based Filter (k={k})...")

    test_movies = [
        "Inception", "Toy Story", "The Dark Knight",
        "Forrest Gump", "The Matrix", "Pulp Fiction",
        "Goodfellas", "Fight Club", "Interstellar", "Up"
    ]

    genre_overlaps = []
    found_count    = 0

    for movie in test_movies:
        recs = get_content_recommendations(
            movie, tmdb_df, tfidf_matrix,
            id_to_idx, search_df, n=k
        )
        if not recs:
            continue

        found_count += 1
        query_row = tmdb_df[
            tmdb_df['title'].str.lower() == movie.lower()
        ]
        if len(query_row) == 0:
            continue

        query_genres = set(
            str(query_row.iloc[0]['genre_names']).split()
        )
        overlaps = [
            1 if len(query_genres &
                     set(str(r['genre_names']).split())) > 0
            else 0
            for r in recs
        ]
        genre_overlaps.append(np.mean(overlaps))

    return {
        'Genre overlap@K': round(np.mean(genre_overlaps), 4)
                           if genre_overlaps else 0,
        'Movies found':    found_count,
        'Movies tested':   len(test_movies)
    }


# ══════════════════════════════════════════════════════════════
# SENTIMENT EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_sentiment(imdb_df, n_samples=300):
    print(f"\nEvaluating Sentiment (n={n_samples})...")

    from src.sentiment import (
        build_vader, vader_score,
        load_distilbert, distilbert_score,
        get_imdb_splits
    )
    from sklearn.metrics import (
        accuracy_score, precision_score,
        recall_score, f1_score
    )

    # sample exclusively from the held-out eval split
    # (rows [6000, …) — never seen during DistilBERT training)
    _, _, eval_df = get_imdb_splits(imdb_df)
    sample = eval_df.sample(
        n=min(n_samples, len(eval_df)), random_state=2
    )
    print(f"  Sampling {len(sample)} reviews from held-out eval split "
          f"({len(eval_df):,} rows, never used in training)")

    analyzer = build_vader()

    vader_correct = sum(
        1 for _, row in sample.iterrows()
        if (1 if vader_score(row['review'], analyzer) > 0
            else 0) == row['label']
    )

    try:
        model, tokenizer = load_distilbert()
        y_true, y_pred   = [], []
        for _, row in sample.iterrows():
            pred = (1 if distilbert_score(
                        row['review'], model, tokenizer
                    ) > 0.5 else 0)
            y_true.append(row['label'])
            y_pred.append(pred)

        db_results = {
            'DistilBERT accuracy':  round(accuracy_score(y_true, y_pred), 4),
            'DistilBERT precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
            'DistilBERT recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
            'DistilBERT F1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
        }
    except Exception as e:
        print(f"  ⚠ DistilBERT skipped: {e}")
        db_results = {
            'DistilBERT accuracy':  'N/A',
            'DistilBERT precision': 'N/A',
            'DistilBERT recall':    'N/A',
            'DistilBERT F1':        'N/A',
        }

    return {
        'VADER accuracy': round(vader_correct / len(sample), 4),
        **db_results
    }


# ══════════════════════════════════════════════════════════════
# WEIGHT OPTIMIZATION
# ══════════════════════════════════════════════════════════════

""" def optimize_weights(user_movie_matrix, train_ratings,
                     test_ratings, movies_df, tmdb_df,
                     tfidf_matrix, id_to_idx,
                     search_df, svd_data):
    print("\n" + "="*55)
    print("  WEIGHT OPTIMIZATION — Grid Search")
    print("="*55)

    cluster_data   = build_clustered_collab_model(train_ratings)
    user_cluster   = cluster_data['user_cluster']
    all_test_users = test_ratings['userId'].unique()
    known_users    = [u for u in all_test_users
                      if u in user_cluster][:20]

    candidates = [
        (a/10, b/10, (10-a-b)/10)
        for a in range(1, 9)
        for b in range(1, 9)
        if (10-a-b) >= 1
    ]
    print(f"  Testing {len(candidates)} combinations "
          f"on {len(known_users)} users...")

    best_score   = -1
    best_weights = (0.4, 0.3, 0.3)
    results      = []

    for i, (alpha, beta, gamma) in enumerate(candidates):
        ndcgs = []
        for user_id in known_users:
            relevant = set(
                test_ratings[
                    (test_ratings['userId'] == user_id) &
                    (test_ratings['rating'] >= 3.5)
                ]['movieId'].tolist()
            )
            if not relevant:
                continue
            try:
                recs = hybrid_recommend(
                    user_id           = user_id,
                    user_movie_matrix = user_movie_matrix,
                    ratings_df        = train_ratings,
                    movies_df         = movies_df,
                    tmdb_df           = tmdb_df,
                    tfidf_matrix      = tfidf_matrix,
                    id_to_idx         = id_to_idx,
                    search_df         = search_df,
                    svd_data          = svd_data,
                    alpha=alpha, beta=beta, gamma=gamma,
                    n=20
                )
                rec_ids = [r['movieId'] for r in recs]
                if rec_ids:
                    ndcgs.append(
                        ndcg_at_k(rec_ids, relevant, 10)
                    )
            except:
                continue

        score = float(np.mean(ndcgs)) if ndcgs else 0.0
        results.append({
            'alpha': alpha, 'beta': beta,
            'gamma': gamma, 'ndcg': round(score, 4)
        })

        if score > best_score:
            best_score   = score
            best_weights = (alpha, beta, gamma)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(candidates)}] Best: "
                  f"α={best_weights[0]} β={best_weights[1]} "
                  f"γ={best_weights[2]} NDCG={best_score:.4f}")

    # show top 5
    results_df = pd.DataFrame(results).sort_values(
        'ndcg', ascending=False
    )
    print(f"\n  Top 5 weight combinations:")
    print(f"  {'Alpha':<8} {'Beta':<8} {'Gamma':<8} {'NDCG'}")
    print(f"  {'-'*36}")
    for _, row in results_df.head(5).iterrows():
        marker = " ← BEST" if (
            row['alpha'] == best_weights[0] and
            row['beta']  == best_weights[1]
        ) else ""
        print(f"  {row['alpha']:<8} {row['beta']:<8} "
              f"{row['gamma']:<8} {row['ndcg']}{marker}")

    print(f"\n  ✓ Best weights: α={best_weights[0]} "
          f"β={best_weights[1]} γ={best_weights[2]}")

    import json
    with open('models/optimal_weights.json', 'w') as f:
        json.dump({
            'alpha': best_weights[0],
            'beta':  best_weights[1],
            'gamma': best_weights[2],
            'ndcg':  best_score
        }, f, indent=2)
    print("  ✓ Saved to models/optimal_weights.json")

    return best_weights, results_df  """


# ══════════════════════════════════════════════════════════════
# SANITY BASELINE — random recommender
# ══════════════════════════════════════════════════════════════

def random_recommend(movies_df, k=10):
    return list(np.random.choice(movies_df['movieId'], k, replace=False))


def evaluate_random_baseline(test_ratings, movies_df, n_users=50, k=10):
    """
    Precision@K and NDCG@K for a random recommender using guaranteed type casting.
    """
    print(f"\nEvaluating Random Baseline (n_users={n_users}, k={k})...")
    rng = np.random.default_rng(42)
    test_users = rng.choice(
        test_ratings['userId'].unique(),
        size=min(n_users, test_ratings['userId'].nunique()),
        replace=False
    )
    precisions, ndcgs = [], []
    
    # Cast target database to guaranteed pure Python standard integers
    all_movie_ids = [int(x) for x in movies_df['movieId'].unique()]

    for user_id in test_users:
        relevant = set(
            int(x) for x in test_ratings[
                (test_ratings['userId'] == user_id) &
                (test_ratings['rating'] >= 3.5)
            ]['movieId'].tolist()
        )
        if not relevant:
            continue
            
        # Sample directly from the standard int array
        rec_ids = [int(x) for x in rng.choice(all_movie_ids, k, replace=False)]
        
        precisions.append(precision_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))
        
    return {
        f'Random Precision@{k}': round(np.mean(precisions), 4) if precisions else 0,
        f'Random NDCG@{k}':      round(np.mean(ndcgs), 4)      if ndcgs      else 0,
        'Users evaluated':        len(precisions)
    }


# ══════════════════════════════════════════════════════════════
# LEAKAGE CHECK
# ══════════════════════════════════════════════════════════════

def check_train_test_leakage(train_ratings, test_ratings):
    """
    Verify zero (user, movie) overlap between train and test splits.
    Any overlap directly inflates every offline metric.
    Should always print: Leakage check: 0 overlaps
    """
    train_pairs = set(zip(train_ratings['userId'], train_ratings['movieId']))
    test_pairs  = set(zip(test_ratings['userId'],  test_ratings['movieId']))
    overlap     = train_pairs & test_pairs
    print(f"Leakage check: {len(overlap)} overlaps")
    if overlap:
        print(f"  ⚠ WARNING: {len(overlap)} (user, movie) pairs appear in both splits!")
    else:
        print("  ✓ No leakage detected")
    return len(overlap)


def print_report(title, results):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    for k, v in results.items():
        print(f"  {k:<30} {v}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  RECOMMENDER SYSTEM — FULL EVALUATION")
    print("=" * 55)

    import scipy.sparse

    # ── load data ─────────────────────────────────────────────
    ratings = load_ratings(sample=True)
    movies  = load_movies()

    from src.sentiment import load_imdb
    imdb = load_imdb()
    imdb['label'] = (imdb['sentiment'] == 'positive').astype(int)

    # ── load pretrained TF-IDF if available ───────────────────
    tmdb_pkl = Path("models/tmdb_clean.pkl")
    if tmdb_pkl.exists():
        print("\nLoading pre-trained TF-IDF from disk...")
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
            ['id','title','release_date',
             'genre_names','vote_count']
        ].copy()
        search_df['title_lower'] = (
            search_df['title'].fillna('').str.lower().str.strip()
        )
        print("✓ Pre-trained TF-IDF loaded")
    else:
        print("\nBuilding TF-IDF from scratch...")
        tmdb = load_tmdb()
        tmdb_clean, tfidf_matrix, tfidf, id_to_idx, search_df = \
            build_content_model(tmdb)

   

    # ── train/test split ──────────────────────────────────────
    print("\nSplitting ratings into train/test...")
    train_ratings, test_ratings = split_user_ratings(ratings)
     # ── load pretrained SVD ───────────────────────────────────
    svd_data = build_svd_model(train_ratings)  # loads from disk if exists
    # ── leakage check ─────────────────────────────────────────
    check_train_test_leakage(train_ratings, test_ratings)

    # ── build collab matrix on TRAIN data ─────────────────────
    user_movie_matrix, _ = build_collab_model(train_ratings)

    # ── build cluster model on TRAIN data ─────────────────────
    cluster_data = build_clustered_collab_model(train_ratings)
    print(f"  Cluster coverage of test users: "
          f"{sum(1 for u in test_ratings['userId'].unique() if u in cluster_data['user_cluster'])} "
          f"/ {test_ratings['userId'].nunique()}")

    # ── 1. clustered CF evaluation ────────────────────────────
    collab_results = evaluate_clustered_collab(
        train_ratings, test_ratings, movies,
        n_users=50, k=10
    )
    print_report("CLUSTERED COLLABORATIVE FILTERING", collab_results)

    # ── 2. content evaluation ─────────────────────────────────
    content_results = evaluate_content(
        tmdb_clean, tfidf_matrix,
        id_to_idx, search_df, k=10
    )
    print_report("CONTENT-BASED FILTERING", content_results)

    # ── 3. sentiment evaluation ───────────────────────────────
    sentiment_results = evaluate_sentiment(imdb, n_samples=300)
    print_report("SENTIMENT MODELS", sentiment_results)

    # ── 4. full hybrid evaluation ─────────────────────────────
    hybrid_results = evaluate_hybrid(
        train_ratings, test_ratings,
        movies, tmdb_clean, tfidf_matrix,
        id_to_idx, search_df, svd_data,
        user_movie_matrix,
        n_users=30, k=10
    )
    print_report("FULL HYBRID SYSTEM", hybrid_results)

    # ── 5. random baseline ────────────────────────────────────
    random_results = evaluate_random_baseline(
        test_ratings, movies, n_users=50, k=10
    )
    print_report("RANDOM BASELINE", random_results)

    # ── 6. final summary ──────────────────────────────────────
    print("\n" + "="*55)
    print("  FINAL SUMMARY")
    print("="*55)
    print(f"  Clustered CF Precision@10  : {collab_results['Precision@10']}")
    print(f"  Clustered CF NDCG@10       : {collab_results['NDCG@10']}")
    print(f"  Hybrid Precision@10        : {hybrid_results['Hybrid Precision@10']}")
    print(f"  Hybrid NDCG@10             : {hybrid_results['Hybrid NDCG@10']}")
    print(f"  Hybrid Diversity           : {hybrid_results['Avg Diversity']}")
    print(f"  Hybrid Coverage            : {hybrid_results['Coverage']}")
    print(f"  Random Precision@10        : {random_results['Random Precision@10']}")
    print(f"  Random NDCG@10             : {random_results['Random NDCG@10']}")
    print(f"  Content Genre Overlap      : {content_results['Genre overlap@K']}")
    print(f"  VADER Accuracy             : {sentiment_results['VADER accuracy']}")
    print(f"  DistilBERT Accuracy        : {sentiment_results['DistilBERT accuracy']}")
    print(f"  DistilBERT F1              : {sentiment_results['DistilBERT F1']}")
    print("="*55)
    print("\n✓ Evaluation complete!")