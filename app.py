import textwrap

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="MOVIE RECOMMENDER",
    page_icon=":sparkles:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_movies_data():
    movies = pd.DataFrame(
        [
            {"item_id": 1, "title": "Inception", "category": "Sci-Fi", "rating": 4.8, "year": 2010},
            {"item_id": 2, "title": "The Dark Knight", "category": "Action", "rating": 4.9, "year": 2008},
            {"item_id": 3, "title": "Interstellar", "category": "Sci-Fi", "rating": 4.7, "year": 2014},
            {"item_id": 4, "title": "Parasite", "category": "Thriller", "rating": 4.6, "year": 2019},
            {"item_id": 5, "title": "Whiplash", "category": "Drama", "rating": 4.5, "year": 2014},
            {"item_id": 6, "title": "The Social Network", "category": "Drama", "rating": 4.3, "year": 2010},
            {"item_id": 7, "title": "Mad Max: Fury Road", "category": "Action", "rating": 4.4, "year": 2015},
            {"item_id": 8, "title": "La La Land", "category": "Romance", "rating": 4.2, "year": 2016},
            {"item_id": 9, "title": "Blade Runner 2049", "category": "Sci-Fi", "rating": 4.4, "year": 2017},
            {"item_id": 10, "title": "Spider-Man: Into the Spider-Verse", "category": "Animation", "rating": 4.6, "year": 2018},
            {"item_id": 11, "title": "Dune", "category": "Sci-Fi", "rating": 4.5, "year": 2021},
            {"item_id": 12, "title": "Top Gun: Maverick", "category": "Action", "rating": 4.5, "year": 2022},
        ]
    )

    ratings = pd.DataFrame(
        [
            ("User 1", 1, 5), ("User 1", 2, 5), ("User 1", 3, 4), ("User 1", 9, 4), ("User 1", 11, 5),
            ("User 2", 2, 5), ("User 2", 7, 4), ("User 2", 12, 5), ("User 2", 10, 4), ("User 2", 1, 4),
            ("User 3", 4, 5), ("User 3", 5, 4), ("User 3", 6, 4), ("User 3", 8, 3), ("User 3", 2, 4),
            ("User 4", 3, 5), ("User 4", 9, 5), ("User 4", 11, 4), ("User 4", 1, 4), ("User 4", 7, 3),
            ("User 5", 8, 5), ("User 5", 5, 4), ("User 5", 10, 4), ("User 5", 4, 4), ("User 5", 6, 3),
            ("User 6", 12, 5), ("User 6", 7, 5), ("User 6", 2, 4), ("User 6", 10, 4), ("User 6", 11, 3),
            ("User 7", 1, 4), ("User 7", 3, 5), ("User 7", 9, 4), ("User 7", 11, 5), ("User 7", 4, 3),
            ("User 8", 5, 5), ("User 8", 6, 4), ("User 8", 8, 4), ("User 8", 4, 4), ("User 8", 10, 3),
        ],
        columns=["user", "item_id", "rating_score"],
    )
    return movies, ratings


def load_songs_data():
    songs = pd.DataFrame(
        [
            {"item_id": 101, "title": "Blinding Lights", "category": "Pop", "rating": 4.8, "artist": "The Weeknd"},
            {"item_id": 102, "title": "Levitating", "category": "Pop", "rating": 4.6, "artist": "Dua Lipa"},
            {"item_id": 103, "title": "As It Was", "category": "Pop", "rating": 4.5, "artist": "Harry Styles"},
            {"item_id": 104, "title": "Bad Guy", "category": "Alt Pop", "rating": 4.4, "artist": "Billie Eilish"},
            {"item_id": 105, "title": "Watermelon Sugar", "category": "Pop", "rating": 4.3, "artist": "Harry Styles"},
            {"item_id": 106, "title": "Stay", "category": "Pop", "rating": 4.5, "artist": "The Kid LAROI"},
            {"item_id": 107, "title": "Peaches", "category": "R&B", "rating": 4.2, "artist": "Justin Bieber"},
            {"item_id": 108, "title": "Good 4 U", "category": "Pop Rock", "rating": 4.4, "artist": "Olivia Rodrigo"},
            {"item_id": 109, "title": "Easy On Me", "category": "Soul", "rating": 4.6, "artist": "Adele"},
            {"item_id": 110, "title": "Anti-Hero", "category": "Pop", "rating": 4.5, "artist": "Taylor Swift"},
            {"item_id": 111, "title": "Calm Down", "category": "Afrobeats", "rating": 4.7, "artist": "Rema"},
            {"item_id": 112, "title": "Flowers", "category": "Pop", "rating": 4.4, "artist": "Miley Cyrus"},
        ]
    )

    ratings = pd.DataFrame(
        [
            ("Listener 1", 101, 5), ("Listener 1", 102, 4), ("Listener 1", 103, 4), ("Listener 1", 106, 5), ("Listener 1", 110, 4),
            ("Listener 2", 104, 5), ("Listener 2", 108, 5), ("Listener 2", 102, 4), ("Listener 2", 112, 3), ("Listener 2", 103, 3),
            ("Listener 3", 109, 5), ("Listener 3", 107, 4), ("Listener 3", 111, 5), ("Listener 3", 101, 4), ("Listener 3", 110, 4),
            ("Listener 4", 111, 5), ("Listener 4", 101, 4), ("Listener 4", 106, 4), ("Listener 4", 107, 3), ("Listener 4", 109, 4),
            ("Listener 5", 110, 5), ("Listener 5", 112, 4), ("Listener 5", 103, 4), ("Listener 5", 102, 4), ("Listener 5", 105, 3),
            ("Listener 6", 108, 5), ("Listener 6", 104, 4), ("Listener 6", 106, 4), ("Listener 6", 101, 3), ("Listener 6", 112, 4),
            ("Listener 7", 105, 5), ("Listener 7", 103, 4), ("Listener 7", 102, 4), ("Listener 7", 110, 3), ("Listener 7", 109, 3),
            ("Listener 8", 111, 5), ("Listener 8", 101, 5), ("Listener 8", 109, 4), ("Listener 8", 106, 4), ("Listener 8", 107, 4),
        ],
        columns=["user", "item_id", "rating_score"],
    )
    return songs, ratings


def build_similarity_matrix(items_df, ratings_df):
    interaction_matrix = ratings_df.pivot_table(
        index="user",
        columns="item_id",
        values="rating_score",
        fill_value=0,
    )
    similarity = cosine_similarity(interaction_matrix.T)
    similarity_df = pd.DataFrame(
        similarity,
        index=interaction_matrix.columns,
        columns=interaction_matrix.columns,
    )
    return similarity_df


def recommend_items(item_name, items_df, similarity_df, top_n=6):
    matched = items_df.loc[items_df["title"].str.lower() == item_name.lower()]
    if matched.empty:
        return pd.DataFrame()

    item_id = matched.iloc[0]["item_id"]
    similarity_scores = similarity_df[item_id].sort_values(ascending=False).drop(item_id)

    recommendations = (
        items_df.set_index("item_id")
        .join(similarity_scores.rename("similarity_score"))
        .dropna(subset=["similarity_score"])
        .sort_values(by="similarity_score", ascending=False)
        .head(top_n)
        .reset_index()
    )
    recommendations["similarity_score"] = recommendations["similarity_score"].round(3)
    return recommendations


def apply_theme(theme_name):
    is_dark = theme_name == "Dark"
    palette = {
        "bg": "#09111f" if is_dark else "#f4f7fb",
        "bg_secondary": "#0f172a" if is_dark else "#ffffff",
        "text": "#e5eefc" if is_dark else "#0f172a",
        "muted": "#94a3b8" if is_dark else "#475569",
        "border": "rgba(148, 163, 184, 0.22)" if is_dark else "rgba(15, 23, 42, 0.08)",
        "glass": "rgba(15, 23, 42, 0.42)" if is_dark else "rgba(255, 255, 255, 0.62)",
        "glass_strong": "rgba(15, 23, 42, 0.62)" if is_dark else "rgba(255, 255, 255, 0.82)",
        "shadow": "0 24px 80px rgba(2, 6, 23, 0.38)" if is_dark else "0 24px 80px rgba(15, 23, 42, 0.10)",
        "accent": "#8b5cf6" if is_dark else "#4f46e5",
        "accent_soft": "rgba(139, 92, 246, 0.16)" if is_dark else "rgba(79, 70, 229, 0.12)",
        "gradient_1": "#0f0c29" if is_dark else "#eef2ff",
        "gradient_2": "#302b63" if is_dark else "#dbeafe",
        "gradient_3": "#24243e" if is_dark else "#ecfeff",
    }

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --bg: {palette["bg"]};
            --bg-secondary: {palette["bg_secondary"]};
            --text: {palette["text"]};
            --muted: {palette["muted"]};
            --border: {palette["border"]};
            --glass: {palette["glass"]};
            --glass-strong: {palette["glass_strong"]};
            --shadow: {palette["shadow"]};
            --accent: {palette["accent"]};
            --accent-soft: {palette["accent_soft"]};
            --gradient-1: {palette["gradient_1"]};
            --gradient-2: {palette["gradient_2"]};
            --gradient-3: {palette["gradient_3"]};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(79, 70, 229, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 28%),
                linear-gradient(135deg, var(--gradient-1), var(--gradient-2) 48%, var(--gradient-3));
            color: var(--text);
        }}

        .block-container {{
            padding-top: 2.3rem;
            padding-bottom: 1.25rem;
            max-width: 1180px;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            display: none;
        }}

        .hero-card, .panel-card, .result-card, .footer-card {{
            background: var(--glass);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border-radius: 24px;
        }}

        .hero-card {{
            padding: 2rem;
            margin-bottom: 1.2rem;
        }}

        .eyebrow {{
            display: inline-flex;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 1rem;
        }}

        .hero-title {{
            font-size: clamp(2rem, 4vw, 3.6rem);
            line-height: 1.02;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0;
            color: var(--text);
        }}

        .hero-text {{
            margin-top: 1rem;
            max-width: 760px;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.7;
        }}

        .stats-row {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }}

        .stat-pill {{
            background: var(--glass-strong);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
        }}

        .stat-label {{
            font-size: 0.8rem;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }}

        .stat-value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text);
        }}

        .panel-card {{
            padding: 1.25rem;
            margin-bottom: 1rem;
        }}

        .section-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.3rem;
        }}

        .section-text {{
            color: var(--muted);
            font-size: 0.95rem;
            margin-bottom: 0.6rem;
        }}

        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {{
            background: rgba(255, 255, 255, 0.08);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 14px;
        }}

        div[data-testid="stTextInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stRadio"] label,
        div[data-testid="stToggle"] label {{
            color: var(--text) !important;
            font-weight: 600;
        }}

        .stButton > button {{
            width: 100%;
            border: none;
            border-radius: 16px;
            padding: 0.9rem 1.1rem;
            background: linear-gradient(135deg, var(--accent), #06b6d4);
            color: white;
            font-weight: 700;
            font-size: 0.98rem;
            box-shadow: 0 12px 30px rgba(79, 70, 229, 0.28);
        }}

        .stButton > button:hover {{
            filter: brightness(1.04);
        }}

        .result-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}

        .result-card {{
            padding: 1.1rem;
            min-height: 184px;
            position: relative;
            overflow: hidden;
        }}

        .result-card::after {{
            content: "";
            position: absolute;
            inset: auto -20% -45% auto;
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.08);
            filter: blur(10px);
        }}

        .item-type {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.85rem;
        }}

        .item-title {{
            font-size: 1.1rem;
            line-height: 1.35;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.55rem;
        }}

        .item-meta {{
            color: var(--muted);
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }}

        .badge-row {{
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: auto;
        }}

        .metric-badge {{
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: var(--glass-strong);
            color: var(--text);
            border: 1px solid var(--border);
            font-size: 0.78rem;
            font-weight: 600;
        }}

        .footer-card {{
            padding: 1rem 1.1rem;
            margin-top: 1.2rem;
            text-align: center;
            color: var(--muted);
            font-size: 0.92rem;
        }}

        @media (max-width: 900px) {{
            .stats-row {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(mode, item_count):
    st.markdown(
        f"""
        <section class="hero-card">
            <div class="eyebrow">Recommendation System</div>
            <h1 class="hero-title">Discover your next favorite {mode.lower()} with collaborative filtering.</h1>
            <p class="hero-text">
                Aura Recommend blends item-based collaborative filtering and cosine similarity to surface
                high-confidence suggestions from a curated sample catalog. Search directly, browse from a selector,
                switch modes instantly, and explore recommendations in a modern SaaS-style interface.
            </p>
            <div class="stats-row">
                <div class="stat-pill">
                    <div class="stat-label">Recommendation Mode</div>
                    <div class="stat-value">{mode}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-label">Catalog Size</div>
                    <div class="stat-value">{item_count} items</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-label">Recommendation Method</div>
                    <div class="stat-value">Cosine Similarity</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_result_cards(recommendations, mode):
    cards = []
    for _, row in recommendations.iterrows():
        meta_line = f"{row['category']}"
        if mode == "Movies":
            meta_line += f" • {int(row['year'])}"
        else:
            meta_line += f" • {row['artist']}"

        card_html = textwrap.dedent(f"""
        <div class="result-card">
            <div class="item-type">{mode[:-1] if mode.endswith('s') else mode}</div>
            <div class="item-title">{row['title']}</div>
            <div class="item-meta">{meta_line}</div>
            <div class="badge-row">
                <span class="metric-badge">Similarity {row['similarity_score']:.3f}</span>
                <span class="metric-badge">Rating {row['rating']:.1f}/5</span>
            </div>
        </div>
        """).strip()

        cards.append(card_html)

    grid_html = '<div class="result-grid">' + "".join(cards) + "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)



def main():
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"

    top_bar_left, top_bar_right = st.columns([0.75, 0.25])
    with top_bar_left:
        mode = st.radio(
            "Choose a recommendation mode",
            ["Movies", "Songs"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with top_bar_right:
        theme_is_dark = st.toggle("Dark mode", value=st.session_state.theme == "Dark")
        st.session_state.theme = "Dark" if theme_is_dark else "Light"

    apply_theme(st.session_state.theme)

    if mode == "Movies":
        items_df, ratings_df = load_movies_data()
    else:
        items_df, ratings_df = load_songs_data()

    similarity_df = build_similarity_matrix(items_df, ratings_df)
    render_hero(mode, len(items_df))

    st.markdown(
        """
        <section class="panel-card">
            <div class="section-title">Find an item</div>
            <div class="section-text">
                Use search and dropdown together. Search narrows the list, while the selector lets you confirm the exact title.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    search_col, select_col, count_col = st.columns([1.4, 1.2, 0.6])
    with search_col:
        search_term = st.text_input(
            "Search by name",
            placeholder=f"Search {mode.lower()} by title...",
        )

    filtered_df = items_df.copy()
    if search_term.strip():
        filtered_df = items_df.loc[
            items_df["title"].str.contains(search_term.strip(), case=False, na=False)
        ]

    option_titles = filtered_df["title"].tolist() if not filtered_df.empty else items_df["title"].tolist()

    default_title = option_titles[0]
    if search_term.strip():
        exact_match = items_df.loc[
            items_df["title"].str.lower() == search_term.strip().lower(), "title"
        ]
        if not exact_match.empty:
            default_title = exact_match.iloc[0]

    with select_col:
        selected_title = st.selectbox(
            "Select from dropdown",
            options=option_titles,
            index=option_titles.index(default_title) if default_title in option_titles else 0,
        )

    with count_col:
        top_n = st.number_input("Results", min_value=5, max_value=10, value=6, step=1)

    selected_label = (
        f"Search matched {search_term.strip()}" if search_term.strip() and not filtered_df.empty else f"Selected {selected_title}"
    )

    st.caption(f"{len(option_titles)} item(s) available in the current filtered list. {selected_label}.")

    generate = st.button("Generate Recommendations", type="primary")

    if generate:
        recommendations = recommend_items(selected_title, items_df, similarity_df, top_n=int(top_n))
        if recommendations.empty:
            st.warning("No recommendations found for the selected item. Try another title.")
        else:
            st.markdown(
                f"""
                <section class="panel-card">
                    <div class="section-title">Top recommendations for {selected_title}</div>
                </section>
                """,
                unsafe_allow_html=True,
            )
            render_result_cards(recommendations, mode)
    else:
        st.markdown(
            f"""
            <section class="panel-card">
                <div class="section-title">Ready when you are</div>
                <div class="section-text">
                    Pick a title and generate recommendations to see polished result cards with similarity scores and ratings.
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        textwrap.dedent(
            """
            <section class="footer-card">
                Built with Python, Pandas, NumPy, scikit-learn, and Streamlit.
            </section>
            """
        ),
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
