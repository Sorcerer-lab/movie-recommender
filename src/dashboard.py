import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CineIQ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background:#0a0a0f; }
    [data-testid="stSidebar"] {
        background:#111118;
        border-right:1px solid #1e1e2e;
    }
    .block-container { padding:2rem 2.5rem 4rem; }
    h1 { font-size:2rem !important; font-weight:700 !important;
         color:#ffffff !important; letter-spacing:-0.5px; }
    h2 { font-size:0.95rem !important; font-weight:600 !important;
         color:#606080 !important; text-transform:uppercase;
         letter-spacing:1.5px; margin-bottom:1rem !important; }
    .stButton > button {
        background:#2979ff !important; color:#fff !important;
        border:none !important; border-radius:50px !important;
        font-weight:700 !important; width:100% !important;
        padding:0.6rem 1.5rem !important;
    }
    .stButton > button:hover {
        background:#448aff !important;
        transform:translateY(-1px) !important;
    }
    .metric-box {
        background:#111118; border:1px solid #1e1e2e;
        border-radius:16px; padding:1.2rem; text-align:center;
    }
    .metric-val {
        font-size:1.6rem; font-weight:800;
        color:#2979ff; display:block;
    }
    .metric-lbl {
        font-size:0.68rem; color:#404060;
        text-transform:uppercase; letter-spacing:1px;
        margin-top:3px; display:block;
    }
    .divider {
        border:none; border-top:1px solid #1e1e2e; margin:2rem 0;
    }
    .section-label {
        font-size:0.7rem; color:#404060; text-transform:uppercase;
        letter-spacing:1.5px; margin-bottom:8px;
        font-weight:600; display:block;
    }

    /* ── recommendation tiles ── */
    .tile-grid {
        display:grid;
        grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
        gap:14px; margin-top:1rem;
    }
    .tile {
        background:#111118; border:1px solid #1e1e2e;
        border-radius:14px; overflow:hidden;
        transition:transform 0.2s,border-color 0.2s;
        position:relative;
    }
    .tile:hover { transform:translateY(-4px); border-color:#2979ff; }
    .tile-rank {
        position:absolute; top:8px; left:8px;
        background:#2979ff; color:#fff;
        font-size:0.65rem; font-weight:800;
        border-radius:50px; padding:2px 8px; z-index:2;
    }
    .tile-poster {
        width:100%; height:200px;
        display:flex; align-items:center;
        justify-content:center; font-size:3.2rem;
        position:relative; overflow:hidden;
    }
    .tile-body { padding:10px 12px 14px; }
    .tile-title {
        font-size:0.8rem; font-weight:700; color:#fff;
        line-height:1.3; margin-bottom:5px;
        display:-webkit-box; -webkit-line-clamp:2;
        -webkit-box-orient:vertical; overflow:hidden;
    }
    .tile-score { font-size:0.72rem; color:#2979ff; font-weight:700; }
    .tile-audience { font-size:0.68rem; color:#505070; margin-top:2px; }
    .tile-bar-bg {
        height:3px; background:#1e1e2e;
        border-radius:2px; margin-top:7px;
    }
    .tile-bar-fg { height:3px; background:#2979ff; border-radius:2px; }
    .tile-expl {
        font-size:0.69rem; color:#7D7DA1;
        margin-top:5px; line-height:1.35; font-style:italic;
    }

    /* ── top rated rows ── */
    .rated-row {
        display:flex; align-items:center; gap:14px;
        background:#111118; border:1px solid #1e1e2e;
        border-radius:12px; padding:12px 16px; margin-bottom:8px;
        transition:border-color 0.2s;
    }
    .rated-row:hover { border-color:#2979ff; }
    .rated-poster {
        width:38px; height:54px; border-radius:8px;
        display:flex; align-items:center; justify-content:center;
        font-size:1.4rem; flex-shrink:0;
    }
    .rated-info { flex:1; min-width:0; }
    .rated-title {
        font-size:0.85rem; font-weight:600; color:#fff;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    }
    .rated-genre { font-size:0.72rem; color:#505070; margin-top:2px; }
    .rated-stars {
        font-size:0.9rem; font-weight:800;
        color:#2979ff; flex-shrink:0;
    }

    /* ── similar movie rows ── */
    .sim-row {
        display:flex; align-items:center; gap:14px;
        background:#111118; border:1px solid #1e1e2e;
        border-radius:12px; padding:12px 16px; margin-bottom:8px;
        transition:border-color 0.2s;
    }
    .sim-row:hover { border-color:#2979ff; }
    .sim-num {
        font-size:1rem; font-weight:800;
        color:#2979ff; min-width:26px; flex-shrink:0;
    }
    .sim-icon {
        width:38px; height:54px; border-radius:8px;
        display:flex; align-items:center;
        justify-content:center; font-size:1.4rem; flex-shrink:0;
    }
    .sim-info { flex:1; min-width:0; }
    .sim-title {
        font-size:0.88rem; font-weight:600; color:#fff;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    }
    .sim-genre { font-size:0.72rem; color:#505070; margin-top:2px; }
    .sim-badge {
        background:#1a2a4e; color:#2979ff;
        font-size:0.7rem; font-weight:700;
        border-radius:50px; padding:3px 10px; flex-shrink:0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def fetch(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=60)
        return r.json() if r.status_code == 200 else None
    except Exception:
        st.error("Cannot connect to API. "
                 "Run uvicorn in another terminal.")
        return None


def genre_emoji(genres):
    g = str(genres).lower()
    if 'action' in g or 'thriller' in g: return '💥'
    if 'comedy' in g:                    return '😂'
    if 'romance' in g:                   return '💕'
    if 'horror' in g:                    return '👻'
    if 'animation' in g or 'family' in g:return '🎠'
    if 'sci' in g or 'fantasy' in g:     return '🚀'
    if 'drama' in g:                     return '🎭'
    if 'crime' in g:                     return '🔍'
    if 'documentary' in g:               return '🎞'
    if 'music' in g:                     return '🎵'
    return '🎬'


# colour palettes for gradient tiles — cycles by index
TILE_GRADIENTS = [
    ('#0d1b3e', '#1a3a7e'),
    ('#1a0d3e', '#3a1a7e'),
    ('#0d2b1e', '#1a5e3a'),
    ('#2b0d1a', '#6e1a30'),
    ('#2b1a0d', '#6e3a0d'),
    ('#0d2b2b', '#0d5e5e'),
    ('#1e0d2b', '#4a0d6e'),
    ('#2b2b0d', '#5e5e0d'),
]


def tile_gradient(idx):
    c1, c2 = TILE_GRADIENTS[idx % len(TILE_GRADIENTS)]
    return f"background:linear-gradient(135deg,{c1},{c2});"


def strip_year(title):
    return re.sub(r'\s*\(\d{4}\)\s*$', '', str(title)).strip()


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;'>
        <span style='font-size:1.7rem;font-weight:900;
                     color:#fff;letter-spacing:-1px;'>
            Cine<span style='color:#2979ff;'>IQ</span>
        </span>
        <p style='font-size:0.7rem;color:gray;
                  margin:4px 0 0;letter-spacing:1px;'>
            AI MOVIE RECOMMENDATIONS
        </p>
    </div>
    <hr style='border-color:#1e1e2e;margin:1rem 0;'>
    """, unsafe_allow_html=True)

    mode = st.radio("", ["For You", "Similar Movies"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e1e2e;margin:1rem 0;'>",
                unsafe_allow_html=True)

    if mode == "For You":
        st.markdown("<span class='section-label'>User ID</span>",
                    unsafe_allow_html=True)
        user_id = st.number_input(
            "", min_value=1, value=187, step=1,
            label_visibility="collapsed"
        )
        st.markdown("<span class='section-label' "
                    "style='margin-top:12px;display:block;'>"
                    "Results</span>", unsafe_allow_html=True)
        n_recs = st.slider("", 5, 20, 10,
                           label_visibility="collapsed")
        st.markdown("<span class='section-label' "
                    "style='margin-top:12px;display:block;'>"
                    "Sentiment weight</span>",
                    unsafe_allow_html=True)
        sw = st.slider("", 0.0, 0.5, 0.2, 0.05,
                       label_visibility="collapsed")
        go_btn = st.button("Get Recommendations")

    else:
        st.markdown("<span class='section-label'>Movie title</span>",
                    unsafe_allow_html=True)
        movie_title = st.text_input(
            "", value="Inception",
            label_visibility="collapsed"
        )

        st.markdown("<span class='section-label' "
                    "style='margin-top:12px;display:block;'>"
                    "Genre (optional)</span>",
                    unsafe_allow_html=True)
        genre_options = [
            "Any", "Action", "Adventure", "Animation",
            "Comedy", "Crime", "Drama", "Family",
            "Fantasy", "Horror", "Romance",
            "Science Fiction", "Thriller"
        ]
        selected_genre = st.selectbox(
            "", genre_options,
            label_visibility="collapsed"
        )

        st.markdown("<span class='section-label' "
                    "style='margin-top:12px;display:block;'>"
                    "Language priority</span>",
                    unsafe_allow_html=True)
        lang_options = {
            "Auto detect": None,
            "English":     "en",
            "Hindi":       "hi",
            "Korean":      "ko",
            "French":      "fr",
            "Spanish":     "es",
            "Japanese":    "ja",
        }
        selected_lang_label = st.selectbox(
            "", list(lang_options.keys()),
            label_visibility="collapsed"
        )
        selected_lang = lang_options[selected_lang_label]

        st.markdown("<span class='section-label' "
                    "style='margin-top:12px;display:block;'>"
                    "Results</span>", unsafe_allow_html=True)
        n_similar = st.slider("", 5, 15, 8,
                              label_visibility="collapsed")
        go_btn = st.button("Find Similar")
    st.markdown("""
    <hr style='border-color:#1e1e2e;margin:1.5rem 0 1rem;'>
    <p style='font-size:0.65rem;color:#252540;line-height:1.6;'>
        Collaborative filtering · Content-based TF-IDF<br>
        SVD matrix factorisation · DistilBERT sentiment
    </p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOR YOU MODE
# ══════════════════════════════════════════════════════════════

if mode == "For You":
    st.markdown("<h1>Your picks tonight</h1>",
                unsafe_allow_html=True)

    if go_btn:

        with st.spinner("Loading profile and generating recommendations..."):
            profile = fetch(f"/user/{user_id}/profile")
            data    = fetch(
                f"/recommend?user_id={user_id}"
                f"&n={n_recs}&sentiment_weight={sw}"
            )

        # ── metrics ───────────────────────────────────────────
        if profile:
            top_genre = max(
                profile['genre_counts'],
                key=profile['genre_counts'].get,
                default="—"
            ) if profile['genre_counts'] else "—"

            top_decade = (
                max(profile['decade_counts'],
                    key=profile['decade_counts'].get,
                    default="—")
                if profile['decade_counts'] else "—"
            )

            c1, c2, c3, c4 = st.columns(4)
            for col, val, lbl in [
                (c1, profile['total_ratings'], "movies rated"),
                (c2, f"{profile['avg_rating']} ★", "avg rating"),
                (c3, top_genre,  "top genre"),
                (c4, top_decade, "fav decade"),
            ]:
                col.markdown(
                    f"<div class='metric-box'>"
                    f"<span class='metric-val'>{val}</span>"
                    f"<span class='metric-lbl'>{lbl}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<hr class='divider'>",
                        unsafe_allow_html=True)

            # ── charts ────────────────────────────────────────
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("<h2>Genre profile</h2>",
                            unsafe_allow_html=True)
                if profile['genre_counts']:
                    top8 = dict(
                        sorted(profile['genre_counts'].items(),
                               key=lambda x: x[1],
                               reverse=True)[:8]
                    )
                    fig = go.Figure(go.Scatterpolar(
                        r=list(top8.values()),
                        theta=list(top8.keys()),
                        fill='toself',
                        fillcolor='rgba(41,121,255,0.12)',
                        line=dict(color='#2979ff', width=2)
                    ))
                    fig.update_layout(
                        polar=dict(
                            bgcolor='#111118',
                            radialaxis=dict(
                                visible=True,
                                showticklabels=False,
                                gridcolor='#1e1e2e',
                                linecolor='#1e1e2e'
                            ),
                            angularaxis=dict(
                                gridcolor='#1e1e2e',
                                linecolor='#1e1e2e',
                                tickfont=dict(
                                    color='#6060a0', size=11
                                )
                            )
                        ),
                        paper_bgcolor='#0a0a0f',
                        margin=dict(l=40,r=40,t=10,b=10),
                        height=270
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col_r:
                st.markdown("<h2>Decade taste</h2>",
                            unsafe_allow_html=True)
                if profile['decade_counts']:
                    dddf = pd.DataFrame(
                        list(profile['decade_counts'].items()),
                        columns=['Decade', 'Count']
                    ).sort_values('Decade')
                    fig2 = px.bar(
                        dddf, x='Decade', y='Count',
                        color='Count',
                        color_continuous_scale=['#1a2a4e','#2979ff']
                    )
                    fig2.update_layout(
                        paper_bgcolor='#0a0a0f',
                        plot_bgcolor='#111118',
                        font=dict(color='#6060a0', size=11),
                        coloraxis_showscale=False,
                        margin=dict(l=10,r=10,t=10,b=10),
                        height=270,
                        xaxis=dict(gridcolor='#1e1e2e',
                                   linecolor='#1e1e2e'),
                        yaxis=dict(gridcolor='#1e1e2e',
                                   linecolor='#1e1e2e')
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # ── top rated by user ─────────────────────────────
            st.markdown("<hr class='divider'>",
                        unsafe_allow_html=True)
            st.markdown("<h2>This user's top rated movies</h2>",
                        unsafe_allow_html=True)

            top_rated = profile.get('top_rated', [])
            if top_rated:
                for idx, m in enumerate(top_rated[:8]):
                    clean  = strip_year(m['title'])
                    emoji  = genre_emoji(m['genres'])
                    grad   = tile_gradient(idx)
                    genres = str(m['genres']).replace('|', ' · ')
                    st.markdown(f"""
                    <div class='rated-row'>
                        <div class='rated-poster'
                             style='{grad}border-radius:8px;'>
                            {emoji}
                        </div>
                        <div class='rated-info'>
                            <div class='rated-title'>{clean}</div>
                            <div class='rated-genre'>{genres}</div>
                        </div>
                        <div class='rated-stars'>{m['rating']} ★</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    "<p style='color:#404060;font-size:0.85rem;'>"
                    "No rating history found for this user.</p>",
                    unsafe_allow_html=True
                )

        # ── recommendations grid ──────────────────────────────
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<h2>Recommended for you</h2>",
                    unsafe_allow_html=True)

        if data and data.get('recommendations'):
            recs      = data['recommendations']
            max_score = max(r['final_score'] for r in recs)

            tile_html = "<div class='tile-grid'>"
            for i, r in enumerate(recs, 1):
                title    = r['title']
                clean    = strip_year(title)
                pct      = int(r['final_score'] / max_score * 100)
                sent     = r.get('sentiment_score', 0.5)
                sent_lbl = ("Loved" if sent > 0.7
                             else "Liked" if sent > 0.4
                             else "Mixed")
                expl     = r.get('explanation', '')
                emoji    = genre_emoji(r.get('genres', ''))
                grad     = tile_gradient(i)

                tile_html += f"""
                <div class='tile'>
                    <div class='tile-rank'>#{i}</div>
                    <div class='tile-poster' style='{grad}'>
                        {emoji}
                    </div>
                    <div class='tile-body'>
                        <div class='tile-title'>{clean}</div>
                        <div class='tile-score'>
                            {r['final_score']:.3f} match
                        </div>
                        <div class='tile-audience'>
                            Audience: {sent_lbl} ({sent:.2f})
                        </div>
                        <div class='tile-bar-bg'>
                            <div class='tile-bar-fg'
                                 style='width:{pct}%'></div>
                        </div>
                        <div class='tile-expl'>{expl}</div>
                    </div>
                </div>"""

            tile_html += "</div>"
            st.markdown(tile_html, unsafe_allow_html=True)

        else:
            st.info("No recommendations found. Try User ID 187.")

    else:
        # ── welcome placeholder ───────────────────────────────
        st.markdown("""
        <div style='background:#111118;border:1px solid #1e1e2e;
                    border-radius:16px;padding:3rem 2rem;
                    text-align:center;margin-top:1rem;'>
            <div style='font-size:4rem;margin-bottom:1rem;'>🎬</div>
            <p style='color:#fff;font-size:1.15rem;font-weight:700;
                      margin-bottom:0.5rem;'>
                Enter a user ID and hit Get Recommendations
            </p>
            <p style='color:#303050;font-size:0.85rem;'>
                Try user ID 187 to see it in action
            </p>
            <div style='margin-top:2rem;display:flex;
                        justify-content:center;gap:12px;
                        flex-wrap:wrap;'>
                <span style='background:#1a1a2e;color:#2979ff;
                             border-radius:50px;padding:6px 16px;
                             font-size:0.78rem;font-weight:600;'>
                    Collaborative filtering
                </span>
                <span style='background:#1a1a2e;color:#2979ff;
                             border-radius:50px;padding:6px 16px;
                             font-size:0.78rem;font-weight:600;'>
                    TF-IDF content
                </span>
                <span style='background:#1a1a2e;color:#2979ff;
                             border-radius:50px;padding:6px 16px;
                             font-size:0.78rem;font-weight:600;'>
                    SVD latent factors
                </span>
                <span style='background:#1a1a2e;color:#2979ff;
                             border-radius:50px;padding:6px 16px;
                             font-size:0.78rem;font-weight:600;'>
                    DistilBERT sentiment
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIMILAR MOVIES MODE
# ══════════════════════════════════════════════════════════════

else:
    st.markdown("<h1>Find similar movies</h1>",
                unsafe_allow_html=True)

    if go_btn:
        # ── CHANGED: build query params ───────────────────────
        genre_param = (
            f"&genre={selected_genre}"
            if selected_genre != "Any" else ""
        )
        lang_param = (
            f"&lang={selected_lang}"
            if selected_lang else ""
        )

        with st.spinner(
            f"Finding movies similar to '{movie_title}'..."
        ):
            data = fetch(
                f"/similar?title={movie_title}"
                f"&n={n_similar}"
                f"{genre_param}{lang_param}"
            )

        # ── UNCHANGED: everything below stays exactly as is ───
        if data and data.get('similar'):
            recs    = data['similar']
            max_sim = max(r['similarity_score'] for r in recs)

            st.markdown(
                f"<p style='color:#404060;font-size:0.85rem;"
                f"margin-bottom:1.5rem;'>Showing {len(recs)} results "
                f"for <strong style='color:#2979ff;'>"
                f"{data['query_title']}</strong></p>",
                unsafe_allow_html=True
            )

            # bar chart
            sim_df = pd.DataFrame(recs)
            fig = px.bar(
                sim_df,
                x='similarity_score', y='title',
                orientation='h', color='similarity_score',
                color_continuous_scale=['#1a2a4e','#2979ff']
            )
            fig.update_layout(
                paper_bgcolor='#0a0a0f',
                plot_bgcolor='#111118',
                font=dict(color='#6060a0', size=11),
                coloraxis_showscale=False,
                margin=dict(l=10,r=10,t=10,b=10),
                height=max(280, len(recs) * 35),
                yaxis=dict(autorange='reversed',
                           gridcolor='#1e1e2e'),
                xaxis=dict(gridcolor='#1e1e2e')
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<hr class='divider'>",
                        unsafe_allow_html=True)
            st.markdown("<h2>Results</h2>",
                        unsafe_allow_html=True)

            for i, r in enumerate(recs, 1):
                emoji = genre_emoji(r['genre_names'])
                grad  = tile_gradient(i)
                st.markdown(f"""
                <div class='sim-row'>
                    <div class='sim-num'>#{i}</div>
                    <div class='sim-icon' style='{grad}
                         border-radius:8px;'>
                        {emoji}
                    </div>
                    <div class='sim-info'>
                        <div class='sim-title'>{r['title']}</div>
                        <div class='sim-genre'>{r['genre_names']}</div>
                    </div>
                    <div class='sim-badge'>
                        {r['similarity_score']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info(
                f"'{movie_title}' not found. "
                "Try: Inception · Toy Story · The Dark Knight"
            )
    else:
        st.markdown("""
        <div style='background:#111118;border:1px solid #1e1e2e;
                    border-radius:16px;padding:3rem 2rem;
                    text-align:center;margin-top:1rem;'>
            <div style='font-size:4rem;margin-bottom:1rem;'>🔍</div>
            <p style='color:#fff;font-size:1.15rem;font-weight:700;
                      margin-bottom:0.5rem;'>
                Find movies similar to one you already love
            </p>
            <p style='color:#303050;font-size:0.85rem;
                      margin-bottom:1.5rem;'>
                Powered by TF-IDF content similarity
            </p>
            <div style='display:flex;justify-content:center;
                        gap:10px;flex-wrap:wrap;'>
                <span style='background:#1a1a2e;color:#2979ff;
                             border-radius:50px;padding:5px 14px;
                             font-size:0.78rem;font-weight:600;'>
                    Inception
                </span>
                <span style='background:#1a1a2e;color:#2979ff;
                             border-radius:50px;padding:5px 14px;
                             font-size:0.78rem;font-weight:600;'>
                    The Dark Knight
                </span>
                <span style='background:#1a1a2e;color:#2979ff;
                             border-radius:50px;padding:5px 14px;
                             font-size:0.78rem;font-weight:600;'>
                    Toy Story
                </span>
                <span style='background:#1a1a2e;color:#2979ff;
                             border-radius:50px;padding:5px 14px;
                             font-size:0.78rem;font-weight:600;'>
                    Pulp Fiction
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)