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

# ------------------ CONSTANTS ------------------
FALLBACK_IMAGE = "http://images.amazon.com/images/P/0671870432.01.MZZZZZZZ.jpg"


# ------------------ REMOVE DEFAULT PADDING ------------------
st.markdown("""
<style>
.block-container {
    padding-top: 0rem !important;
}
header, footer {
    visibility: hidden;
    height: 0;
}
</style>
""", unsafe_allow_html=True)


# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #f6f8fc;
}

/* HEADER */
.sticky-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 64px;
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    z-index: 999999;
    display: flex;
    align-items: center;
    justify-content: center;
}

.logo {
    position: absolute;
    left: 24px;
    height: 40px;
}

.header-title {
    font-size: 22px;
    color: white;
    font-weight: 700;
}

.page-spacer {
    height: 64px;
}

            
/* BOOK CARD */
.book-card {
    background: white;
    border-radius: 14px;
    padding: 10px;
    height: 320px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    text-align: center;
}

.book-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 26px rgba(0,0,0,0.18);
}

.book-img {
    height: 150px;
    object-fit: contain;
    margin-bottom: 6px;
}

/* TITLE */
.book-title {
    font-size: 16px;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.book-author {
    font-size: 14px;
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
    left:0;
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
    time.sleep(1)
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

# ------------------ SESSION STATE ------------------
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

        # ✅ SAFE IMAGE LOGIC
        img_url = row["Image-URL-M"]
        if pd.isna(img_url) or str(img_url).strip() == "":
            img_url = FALLBACK_IMAGE

        st.markdown(f"""
        <div class="book-card">
            <img class="book-img"
                 src="{img_url}"
                 onerror="this.src='{FALLBACK_IMAGE}'">
            <div class="book-title" title="{row['Book-Title']}">
                {row['Book-Title']}
            </div>
            <div class="book-author">{row['Book-Author']}</div>
            <div class="book-meta">
                {row['Publisher']} · {row['Year-Of-Publication']}
            </div>
            <div class="rating">
                ⭐ {row['avg_rating']} ({row['rating_count']})
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SHOW MORE / LESS
        if st.session_state.expanded == row["ISBN"]:
            if st.button("🔽 Show Less", key=f"less_{row['ISBN']}"):
                st.session_state.expanded = None
                st.rerun()

            details = ratings[ratings["ISBN"] == row["ISBN"]] \
                .merge(users, on="User-ID", how="left") \
                .head(5)

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

c1, _, c3 = st.columns([1,2,1])

with c1:
    if st.session_state.page > 1:
        if st.button("⬅ Previous"):
            st.session_state.page -= 1
            st.rerun()

with c3:
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
