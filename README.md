# 🎬 CineIQ — AI-Powered Movie Recommendation System


CineIQ is a hybrid movie recommendation engine that combines **clustered collaborative filtering**, **SVD matrix factorisation**, **TF-IDF content-based filtering**, and **DistilBERT sentiment re-ranking** into a single personalised recommendation pipeline — served via a FastAPI backend and an interactive Streamlit dashboard.

---

## 📽️ Demo Video

> 🎥 **[Watch Demo Video](https://drive.google.com/file/d/1AuxYRQlRcnUFtecNqSMHEhcQjCWMYCuF/view?usp=sharing)

---

## 📊 Results at a Glance

| Component | Metric | Score |
|---|---|---|
| Clustered CF | Precision@10 | 0.0063 |
| Clustered CF | Hit Rate@10 | 6.25% |
| Content-Based | Genre Overlap@10 | 87.9% |
| DistilBERT Sentiment | F1 Score | 93.67% |
| Hybrid System | Diversity | 0.765 |
| Random Baseline | Precision@10 | 0.0 |

> Evaluated on a temporal 80/20 train-test split (most recent 20% of each user's ratings held out). Random baseline scores exactly zero — confirming the system generates meaningful signal above chance across a 62,000-film catalogue.

---

## 🏗️ System Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────┐
│              HYBRID PIPELINE                    │
│                                                 │
│  ┌──────────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Clustered CF │  │   SVD    │  │ TF-IDF   │  │
│  │ (α = 0.40)   │  │ (γ=0.30) │  │ (β=0.30) │  │
│  │ 100 clusters │  │ k = 100  │  │ 15K feat │  │
│  │ 26M ratings  │  │ 26M full │  │ 45K films│  │
│  └──────┬───────┘  └────┬─────┘  └────┬─────┘  │
│         └───────────────┴─────────────┘         │
│                         │                       │
│              Weighted Ensemble Score             │
│                         │                       │
│              ┌──────────▼──────────┐            │
│              │  DistilBERT + TMDB  │            │
│              │  Re-ranking (+15%   │            │
│              │  each)              │            │
│              └──────────┬──────────┘            │
└─────────────────────────┼───────────────────────┘
                          │
                          ▼
                 Top-N Recommendations
```

**Final scoring formula:**
```
final_score = 0.70 × hybrid_score + 0.15 × tmdb_score + 0.15 × distilbert_score
```



> **Note:** `models/` and `data/` are excluded from the repo via `.gitignore` due to file size. See [Dataset Setup](#-dataset-setup) and [Model Setup](#-model-setup) below.

---

## 🧠 Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Collaborative Filtering | scikit-learn MiniBatchKMeans |
| Matrix Factorisation | cupyx / scipy sparse SVD (k=100) |
| Content-Based | TF-IDF (scikit-learn, 15K features) |
| Sentiment Analysis | DistilBERT (HuggingFace Transformers) + VADER |
| Fuzzy Title Matching | rapidfuzz WRatio (threshold 88) |
| Backend API | FastAPI |
| Frontend | Streamlit |
| GPU Training | Google Colab T4 |

---

## 📦 Dataset Setup

This project uses three datasets. Download and place them in the `data/raw` folder:

| Dataset | Source | File |
|---|---|---|
| MovieLens 25M | [grouplens.org/datasets/movielens/25m](https://grouplens.org/datasets/movielens/25m/) | `ml_ratings.csv` |
| TMDB Movies Metadata | [Kaggle — TMDB 5000](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) | `tmdb_metadata.csv` |
| IMDB 50K Reviews | [Kaggle — IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | `imdb_reviews.csv` |

---

## 🤖 Model Setup

### Option A — Download pre-trained models (recommended)

> Pre-trained models are stored on Google Drive.  
> 📂 **[Download models folder](https://drive.google.com/drive/folders/1gxcAzSPJtb1x_JCeARDB8KSPjFT8A_Y2?usp=sharing)** 

Place the downloaded `models/` folder in the project root.

### Option B — Train from scratch


1. Open `colab_notebook_models/g_colab_models.ipynb` given in the repo, on Google Colab
2. Mount your Drive and set paths
3. Run all cells — takes ~2 minutes on T4
4. Download the saved models



---

## 🚀 Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI backend

```bash
uvicorn src.api:app --reload
```

API available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### 3. Launch the Streamlit dashboard

```bash
streamlit run src/app.py
```

Dashboard available at `http://localhost:8501`

---

## 📈 Running Evaluation

```bash
python src/evaluate.py
```

This runs the full offline evaluation suite:
- Clustered CF (Precision, Recall, Hit Rate, NDCG, MAP, MRR)
- Content-based genre overlap
- Sentiment model accuracy (VADER + DistilBERT)
- Full hybrid system metrics + popularity bias analysis
- Random baseline

Expected runtime: ~10–15 minutes on CPU.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/recommend/{user_id}?n=10` | Hybrid recommendations for a known user |
| GET | `/similar/{title}?n=10` | Content-based similar movies |
| GET | `/sentiment?text=...` | DistilBERT sentiment score for a review |

---

## 🔍 Key Design Decisions

- **Item-based CF over user-based** — avoids the O(n²) memory wall of a 162K × 162K user similarity matrix
- **GPU SVD on full dataset** — cupyx sparse SVD on Colab T4 trains k=100 factors on 26M ratings in ~2 minutes; scipy CPU would take hours
- **Fixed ensemble weights (α=0.4, β=0.3, γ=0.3)** — chosen by data-richness reasoning; CF has the densest signal so receives highest weight. Grid search tuning is listed as future work
- **rapidfuzz WRatio at threshold 88** — handles MovieLens "Title, The (YYYY)" → TMDB "The Title" inversions without false-match risk
- **Temporal train/test split** — most recent 20% of each user's ratings held out; simulates real deployment conditions and prevents data leakage

---

## 🔮 Future Work

- Neural collaborative filtering (LightGCN / NeuMF)
- Ensemble weight grid search over α ∈ [0.3, 0.5]
- Session-based recommendations using transformer attention
- A/B testing framework for real click-through rate measurement
- Multi-modal enrichment via CLIP (poster images, trailer embeddings)

---

## 📄 Report

📑 **[Project Report (PDF/Word)](https://docs.google.com/document/d/1ZP9-2Myeg_Jd1kZaMkT-24mhXPo0uxPN/edit?usp=sharing&ouid=116307968164985639604&rtpof=true&sd=true)** 

---

 