import streamlit as st
import pandas as pd
from pathlib import Path
import math
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# ------------------ REMOVE DEFAULT PADDING ------------------
st.markdown("""
<style>
.block-container {
    padding: 0rem !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

body {
    background-color: #f6f8fc;
}

header, footer {visibility: hidden;}

.sticky-header {
    position: fixed;
    top: 0;
    width: 100%;
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 12px 30px;
    z-index: 1000;
    display: flex;
    align-items: center;
}

.logo {
    height: 45px;
}

.header-title {
    flex: 1;
    text-align: center;
    font-size: 26px;
    color: white;
    font-weight: 600;
}

.page-spacer {
    height: 80px;
}

/* BOOK CARD */
.book-card {
    background: white;
    border-radius: 14px;
    padding: 12px;
    height: 420px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    text-align: center;
}

.book-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 10px 26px rgba(0,0,0,0.18);
}

.book-img {
    height: 210px;
    object-fit: contain;
    margin-bottom: 10px;
}

.book-title {
    font-size: 15px;
    font-weight: 600;
    min-height: 45px;
}

.book-author {
    font-size: 13px;
    color: #555;
}

.rating {
    color: #f4b400;
    font-weight: 600;
    margin-top: 4px;
}

/* BUTTONS */
.btn {
    margin-top: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    border: none;
    background: #2a5298;
    color: white;
    cursor: pointer;
    font-size: 13px;
}

.btn:hover {
    background: #1e3c72;
}

/* PAGINATION */
.pagination {
    position: fixed;
    bottom: 55px;
    width: 100%;
    background: #f6f8fc;
    padding: 10px;
    text-align: center;
}

/* FOOTER */
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    color: white;
    text-align: center;
    padding: 8px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
LOGO_BASE64 = """PUT_YOUR_BASE64_IMAGE_HERE"""

st.markdown(f"""
<div class="sticky-header">
    <img class="logo" src="data:image/jpeg;base64,{LOGO_BASE64}">
    <div class="header-title">Book Recommendation System</div>
</div>
<div class="page-spacer"></div>
""", unsafe_allow_html=True)

# ------------------ DATA LOADING ------------------
DATA_DIR = Path(__file__).parent

@st.cache_data(show_spinner=False)
def load_data():
    time.sleep(1.5)  # loader effect
    books = pd.read_csv(DATA_DIR / "Books.csv", encoding="latin-1")
    ratings = pd.read_csv(DATA_DIR / "Ratings.csv", encoding="latin-1")
    users = pd.read_csv(DATA_DIR / "Users.csv", encoding="latin-1")
    return books, ratings, users

with st.spinner("✨ Loading books..."):
    books, ratings, users = load_data()

# ------------------ PREP DATA ------------------
avg_ratings = ratings.groupby("ISBN")["Book-Rating"].mean().round(2)
rating_count = ratings.groupby("ISBN")["Book-Rating"].count()

books["avg_rating"] = books["ISBN"].map(avg_ratings).fillna(0)
books["rating_count"] = books["ISBN"].map(rating_count).fillna(0).astype(int)

# ------------------ PAGINATION ------------------
PER_PAGE = 12
total_pages = math.ceil(len(books) / PER_PAGE)

if "page" not in st.session_state:
    st.session_state.page = 1

start = (st.session_state.page - 1) * PER_PAGE
end = start + PER_PAGE
page_books = books.iloc[start:end]

# ------------------ BOOK GRID ------------------
st.markdown("## 📚 Explore Books")

cols = st.columns(4)

for i, row in page_books.iterrows():
    with cols[i % 4]:
        st.markdown(f"""
        <div class="book-card">
            <img class="book-img"
                 src="{row['Image-URL-L']}"
                 onerror="this.src='https://via.placeholder.com/150'">
            <div class="book-title">{row['Book-Title']}</div>
            <div class="book-author">{row['Book-Author']}</div>
            <div class="rating">⭐ {row['avg_rating']} ({row['rating_count']})</div>
        </div>
        """, unsafe_allow_html=True)

# ------------------ PAGINATION CONTROLS ------------------
st.markdown('<div class="pagination">', unsafe_allow_html=True)

prev_disabled = st.session_state.page == 1
next_disabled = st.session_state.page == total_pages

col1, col2, col3 = st.columns([1,2,1])

with col1:
    if st.button("⬅ Previous", disabled=prev_disabled):
        st.session_state.page -= 1

with col3:
    if st.button("Next ➡", disabled=next_disabled):
        st.session_state.page += 1

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footer">
© 2026 Book Recommendation System | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
