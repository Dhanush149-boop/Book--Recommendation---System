import streamlit as st
import pandas as pd
from pathlib import Path
import math
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="🪷",
    layout="wide"
)

# ------------------ CSS ------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    padding: 0;
    margin: 0;
}

header {visibility: hidden;}

.sticky-header {
    position: fixed;
    top: 0;
    width: 100%;
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 12px 24px;
    z-index: 999;
    display: flex;
    align-items: center;
}

.logo {
    font-size: 32px;
}

.title {
    flex: 1;
    text-align: center;
    font-size: 28px;
    color: white;
    font-weight: 600;
}

.main {
    margin-top: 80px;
    margin-bottom: 120px;
    padding: 0 20px;
}

.card {
    height: 430px;
    border-radius: 14px;
    padding: 14px;
    background: white;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.15);
}

.card img {
    height: 190px;
}

.card-title {
    font-size: 15px;
    font-weight: 600;
    margin: 6px 0;
}

.card-text {
    font-size: 13px;
    color: #555;
}

.pagination {
    position: fixed;
    bottom: 52px;
    width: 100%;
    background: #f8fafc;
    padding: 10px;
    text-align: center;
    z-index: 998;
}

.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background: #1e293b;
    color: white;
    padding: 10px;
    text-align: center;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div class="sticky-header">
    <div class="logo">🪷</div>
    <div class="title">Book Recommendation System</div>
</div>
""", unsafe_allow_html=True)

# ------------------ DATA ------------------
DATA_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    books = pd.read_csv(DATA_DIR / "Books.csv", encoding="latin-1")
    ratings = pd.read_csv(DATA_DIR / "Ratings.csv", encoding="latin-1")
    users = pd.read_csv(DATA_DIR / "Users.csv", encoding="latin-1")
    return books, ratings, users

books, ratings, users = load_data()

# ------------------ PREPARE DATA ------------------
rating_stats = ratings.groupby("ISBN").agg(
    avg_rating=("Book-Rating", "mean"),
    rating_count=("Book-Rating", "count")
).reset_index()

books = books.merge(rating_stats, on="ISBN", how="left")
books["avg_rating"] = books["avg_rating"].fillna(0).round(2)
books["rating_count"] = books["rating_count"].fillna(0).astype(int)

# ------------------ PAGINATION ------------------
PER_PAGE = 12
total_pages = math.ceil(len(books) / PER_PAGE)

if "page" not in st.session_state:
    st.session_state.page = 1

if "expanded" not in st.session_state:
    st.session_state.expanded = None

start = (st.session_state.page - 1) * PER_PAGE
end = start + PER_PAGE
page_books = books.iloc[start:end]

# ------------------ MAIN ------------------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.subheader("📚 Explore Books")

cols = st.columns(4)

for i, row in page_books.iterrows():
    with cols[i % 4]:
        st.markdown(f"""
        <div class="card">
            <img src="{row['Image-URL-M']}"><br>
            <div class="card-title">{row['Book-Title']}</div>
            <div class="card-text">{row['Book-Author']}</div>
            <div class="card-text">📅 {row['Year-Of-Publication']}</div>
            <div class="card-text">⭐ {row['avg_rating']} ({row['rating_count']})</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.expanded == row["ISBN"]:
            if st.button("🔽 Show Less", key=f"less{row['ISBN']}"):
                st.session_state.expanded = None
                st.rerun()

            user_data = ratings[ratings["ISBN"] == row["ISBN"]].merge(
                users, on="User-ID", how="left"
            ).head(5)

            st.dataframe(
                user_data[["User-ID", "Location", "Age", "Book-Rating"]],
                use_container_width=True
            )
        else:
            if st.button("🔼 Show More", key=f"more{row['ISBN']}"):
                st.session_state.expanded = row["ISBN"]
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PAGINATION BAR ------------------
st.markdown('<div class="pagination">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col1:
    if st.button("⬅ Previous", disabled=st.session_state.page == 1):
        with st.spinner("Loading books..."):
            time.sleep(0.5)
            st.session_state.page -= 1
            st.session_state.expanded = None
            st.rerun()

with col3:
    if st.button("Next ➡", disabled=st.session_state.page == total_pages):
        with st.spinner("Loading books..."):
            time.sleep(0.5)
            st.session_state.page += 1
            st.session_state.expanded = None
            st.rerun()

st.markdown(f"Page {st.session_state.page} of {total_pages}", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footer">
    © 2026 Book Recommendation System | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
