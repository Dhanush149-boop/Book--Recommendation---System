import streamlit as st
import pandas as pd
from pathlib import Path
import math

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
header {visibility: hidden;}

.sticky-header {
    position: fixed;
    top: 0;
    width: 100%;
    background-color: #0f172a;
    padding: 10px 30px;
    z-index: 999;
    display: flex;
    align-items: center;
}

.logo {
    font-size: 26px;
}

.title {
    flex: 1;
    text-align: center;
    font-size: 26px;
    color: white;
    font-weight: bold;
}

.main-content {
    margin-top: 90px;
    margin-bottom: 90px;
}

.book-card {
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    background: white;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
}

.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: #0f172a;
    color: white;
    text-align: center;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div class="sticky-header">
    <div class="logo">📚</div>
    <div class="title">Book Recommendation System</div>
</div>
""", unsafe_allow_html=True)

# ------------------ DATA LOADING ------------------
DATA_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    books = pd.read_csv(DATA_DIR / "Books.csv", encoding="latin-1")
    ratings = pd.read_csv(DATA_DIR / "Ratings.csv", encoding="latin-1")
    users = pd.read_csv(DATA_DIR / "Users.csv", encoding="latin-1")
    return books, ratings, users

books, ratings, users = load_data()

# ------------------ MERGE & PREPARE DATA ------------------
ratings_summary = ratings.groupby("ISBN").agg(
    avg_rating=("Book-Rating", "mean"),
    rating_count=("Book-Rating", "count")
).reset_index()

books = books.merge(ratings_summary, on="ISBN", how="left")
books["avg_rating"] = books["avg_rating"].fillna(0).round(2)
books["rating_count"] = books["rating_count"].fillna(0).astype(int)

# ------------------ PAGINATION ------------------
BOOKS_PER_PAGE = 12
total_pages = math.ceil(len(books) / BOOKS_PER_PAGE)

if "page" not in st.session_state:
    st.session_state.page = 1

start = (st.session_state.page - 1) * BOOKS_PER_PAGE
end = start + BOOKS_PER_PAGE
books_page = books.iloc[start:end]

# ------------------ MAIN CONTENT ------------------
st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.subheader("📖 Explore Books")

cols = st.columns(4)

for idx, row in books_page.iterrows():
    with cols[idx % 4]:
        st.markdown(f"""
        <div class="book-card">
            <img src="{row['Image-URL-M']}" height="180"><br><br>
            <b>{row['Book-Title']}</b><br>
            <small>{row['Book-Author']}</small><br>
            📅 {row['Year-Of-Publication']}<br>
            🏢 {row['Publisher']}<br><br>
            ⭐ {row['avg_rating']} ({row['rating_count']} ratings)
        </div>
        """, unsafe_allow_html=True)

        if st.button("Show More", key=row["ISBN"]):
            st.markdown("#### 👤 User Ratings")

            book_ratings = ratings[ratings["ISBN"] == row["ISBN"]].merge(
                users, on="User-ID", how="left"
            ).head(5)

            st.dataframe(
                book_ratings[["User-ID", "Location", "Age", "Book-Rating"]],
                use_container_width=True
            )

# ------------------ PAGINATION CONTROLS ------------------
col1, col2, col3 = st.columns([1,2,1])

with col1:
    if st.button("⬅ Previous") and st.session_state.page > 1:
        st.session_state.page -= 1
        st.rerun()

with col3:
    if st.button("Next ➡") and st.session_state.page < total_pages:
        st.session_state.page += 1
        st.rerun()

st.markdown(f"<p style='text-align:center;'>Page {st.session_state.page} of {total_pages}</p>",
            unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit | Book Recommendation System
</div>
""", unsafe_allow_html=True)
