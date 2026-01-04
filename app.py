import streamlit as st
import pandas as pd
from pathlib import Path
import math
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Book Recommendation System",
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
    padding: 10px 24px;
    z-index: 1000;
    display: flex;
    align-items: center;
}

.logo {
    height: 42px;
}

.header-title {
    flex: 1;
    text-align: center;
    font-size: 24px;
    color: white;
    font-weight: 700;
}

.page-spacer {
    height: 75px;
}

/* BOOK CARD */
.book-card {
    background: white;
    border-radius: 14px;
    padding: 10px;
    height: 350px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    text-align: center;
}

.book-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 26px rgba(0,0,0,0.18);
}

.book-img {
    height: 160px;
    object-fit: contain;
    margin-bottom: 6px;
}

.book-title {
    font-size: 17px;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.book-author {
    font-size: 14px;
    font-weight: 500;
    color: #444;
}

.book-meta {
    font-size: 12px;
    color: #777;
}

.rating {
    color: #f4b400;
    font-weight: 600;
    margin-top: 4px;
}

/* BUTTON */
.show-btn {
    margin-top: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    border: none;
    background: #2a5298;
    color: white;
    font-size: 12px;
    cursor: pointer;
}

.show-btn:hover {
    background: #1e3c72;
}

/* PAGINATION */
.pagination {
    position: fixed;
    bottom: 45px;
    width: 100%;
    background: #f6f8fc;
    padding: 8px;
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
    padding: 6px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div class="sticky-header">
    <img class="logo" src="https://surl.lt/tjejwe">
    <div class="header-title">Book Recommendation System</div>
</div>
<div class="page-spacer"></div>
""", unsafe_allow_html=True)

# ------------------ DATA LOADING ------------------
DATA_DIR = Path(__file__).parent

@st.cache_data(show_spinner=False)
def load_data():
    time.sleep(1.2)
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

if "expanded" not in st.session_state:
    st.session_state.expanded = None

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
                 src="{row['Image-URL-M']}"
                 onerror="this.src='https://via.placeholder.com/150'">

            <div class="book-title" title="{row['Book-Title']}">
                {row['Book-Title']}
            </div>

            <div class="book-author">
                {row['Book-Author']}
            </div>

            <div class="book-meta">
                {row['Publisher']} · {row['Year-Of-Publication']}
            </div>

            <div class="rating">
                ⭐ {row['avg_rating']} ({row['rating_count']})
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.expanded == row["ISBN"]:
            if st.button("🔽 Show Less", key=f"less_{row['ISBN']}"):
                st.session_state.expanded = None
                st.rerun()

            details = ratings[ratings["ISBN"] == row["ISBN"]] \
                .merge(users, on="User-ID", how="left") \
                .head(3)

            st.dataframe(
                details[["User-ID", "Location", "Age", "Book-Rating"]],
                use_container_width=True,
                height=150
            )
        else:
            if st.button("🔼 Show More", key=f"more_{row['ISBN']}"):
                st.session_state.expanded = row["ISBN"]
                st.rerun()

# ------------------ PAGINATION ------------------
st.markdown('<div class="pagination">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col1:
    if st.session_state.page > 1:
        if st.button("⬅ Previous"):
            st.session_state.page -= 1
            st.rerun()

with col3:
    if st.session_state.page < total_pages:
        if st.button("Next ➡"):
            st.session_state.page += 1
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footer">
© 2026 Book Recommendation System | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
