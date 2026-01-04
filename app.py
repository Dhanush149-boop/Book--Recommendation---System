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

# ------------------ REMOVE STREAMLIT PADDING ------------------
st.markdown("""
<style>
.block-container { padding: 0 !important; }
header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
/* HEADER */
.header {
    position: fixed;
    top: 0;
    width: 100%;
    height: 70px;
    background: linear-gradient(90deg, #f8fafc, #e2e8f0);
    display: flex;
    align-items: center;
    padding: 0 25px;
    z-index: 1000;
    border-bottom: 1px solid #ddd;
}

.header img {
    height: 55px;
}

.header-title {
    flex: 1;
    text-align: center;
    font-size: 26px;
    font-weight: 700;
    color: #1e293b;
}

.spacer { height: 85px; }

/* CARD */
.card {
    height: 360px;
    border-radius: 14px;
    background: #ffffff;
    padding: 12px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    transition: 0.3s ease;
}

.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 26px rgba(0,0,0,0.15);
}

.card img {
    height: 150px;
    object-fit: contain;
    margin-bottom: 8px;
}

/* TEXT */
.title {
    font-size: 17px;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.author {
    font-size: 14px;
    font-weight: 500;
    color: #475569;
}

.meta {
    font-size: 12px;
    color: #64748b;
}

/* BUTTON */
.stButton button {
    border-radius: 20px;
    font-size: 12px;
    padding: 4px 14px;
}

/* PAGINATION */
.pagination {
    position: fixed;
    bottom: 48px;
    width: 100%;
    background: #f8fafc;
    padding: 8px;
    text-align: center;
    border-top: 1px solid #ddd;
}

/* FOOTER */
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    height: 45px;
    background: #1e293b;
    color: white;
    text-align: center;
    padding-top: 12px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div class="header">
    <img src="assets/saraswati.png">
    <div class="header-title">Book Recommendation System</div>
</div>
<div class="spacer"></div>
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

# ------------------ RATINGS ------------------
stats = ratings.groupby("ISBN").agg(
    avg=("Book-Rating", "mean"),
    cnt=("Book-Rating", "count")
).reset_index()

books = books.merge(stats, on="ISBN", how="left")
books[["avg","cnt"]] = books[["avg","cnt"]].fillna(0)

# ------------------ SESSION ------------------
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
cols = st.columns(4)

for i, row in page_books.iterrows():
    with cols[i % 4]:
        st.markdown(f"""
        <div class="card">
            <img src="{row['Image-URL-M']}">
            <div class="title">{row['Book-Title']}</div>
            <div class="author">{row['Book-Author']}</div>
            <div class="meta">{row['Publisher']} · {row['Year-Of-Publication']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.expanded == row["ISBN"]:
            if st.button("🔽 Show Less", key=f"less{row['ISBN']}"):
                st.session_state.expanded = None
                st.rerun()

            details = ratings[ratings["ISBN"] == row["ISBN"]] \
                .merge(users, on="User-ID", how="left") \
                .head(5)

            st.dataframe(details[["User-ID","Location","Age","Book-Rating"]],
                         use_container_width=True)

        else:
            if st.button("🔼 Show More", key=f"more{row['ISBN']}"):
                st.session_state.expanded = row["ISBN"]
                st.rerun()

# ------------------ PAGINATION ------------------
st.markdown('<div class="pagination">', unsafe_allow_html=True)

if st.session_state.page > 1:
    if st.button("⬅ Previous"):
        with st.spinner("Loading..."):
            time.sleep(0.4)
            st.session_state.page -= 1
            st.session_state.expanded = None
            st.rerun()

if st.session_state.page < total_pages:
    if st.button("Next ➡"):
        with st.spinner("Loading..."):
            time.sleep(0.4)
            st.session_state.page += 1
            st.session_state.expanded = None
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footer">
© 2026 Book Recommendation System | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
