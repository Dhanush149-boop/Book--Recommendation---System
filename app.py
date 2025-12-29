import streamlit as st
import pandas as pd
from pathlib import Path

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Recommendation System")

# ------------------ DATA LOADING ------------------
DATA_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    required_files = ["Books.csv", "Ratings.csv", "Users.csv"]

    for file in required_files:
        if not (DATA_DIR / file).exists():
            st.error(f"❌ Missing file: {file}")
            return None, None, None

    books = pd.read_csv(DATA_DIR / "Books.csv")
    ratings = pd.read_csv(DATA_DIR / "Ratings.csv")
    users = pd.read_csv(DATA_DIR / "Users.csv")

    return books, ratings, users

books, ratings, users = load_data()

if books is None:
    st.stop()

st.success("✅ Data loaded successfully")

# ------------------ DATA CLEANING ------------------
books = books.dropna(subset=["Book-Title"])
ratings = ratings.dropna(subset=["Book-Rating"])

# ------------------ DATA OVERVIEW ------------------
st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Books", books.shape[0])
col2.metric("Total Ratings", ratings.shape[0])
col3.metric("Total Users", users.shape[0])

# ------------------ DATA PREVIEW ------------------
st.subheader("🔍 Sample Data Preview")

with st.expander("📘 Books Dataset"):
    st.dataframe(books.head(10))

with st.expander("⭐ Ratings Dataset"):
    st.dataframe(ratings.head(10))

with st.expander("👤 Users Dataset"):
    st.dataframe(users.head(10))

# ------------------ POPULARITY-BASED RECOMMENDER ------------------
st.subheader("🔥 Popular Books Recommendation")

def get_popular_books(n=10):
    popular = (
        ratings.groupby("ISBN")["Book-Rating"]
        .count()
        .reset_index()
        .rename(columns={"Book-Rating": "Number of Ratings"})
    )

    popular = popular.merge(books, on="ISBN")
    popular = popular.sort_values(
        by="Number of Ratings", ascending=False
    ).head(n)

    return popular

num_books = st.slider("Select number of books", 5, 20, 10)

popular_books = get_popular_books(num_books)

cols = st.columns(5)
for i, (_, row) in enumerate(popular_books.iterrows()):
    with cols[i % 5]:
        st.markdown(f"**{row['Book-Title']}**")
        st.caption(row.get("Book-Author", "Unknown Author"))
        if "Image-URL-M" in row:
            st.image(row["Image-URL-M"], use_container_width=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("📚 *Book Recommendation System built with Streamlit*")
