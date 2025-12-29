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
    try:
        books = pd.read_csv(DATA_DIR / "Books_compressed.csv.gz")
        ratings = pd.read_csv(DATA_DIR / "Ratings.csv")
        users = pd.read_csv(DATA_DIR / "Users.csv")
        return books, ratings, users
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

books, ratings, users = load_data()

if books is None:
    st.stop()

st.success("✅ Data loaded successfully")

# ------------------ DATA CLEANING ------------------
books = books.dropna(subset=["Book-Title"])
ratings = ratings.dropna(subset=["Book-Rating"])

# ------------------ POPULARITY-BASED RECOMMENDER ------------------
def get_popular_books(n=10):
    popular = (
        ratings.groupby("ISBN")["Book-Rating"]
        .count()
        .reset_index()
        .rename(columns={"Book-Rating": "num_ratings"})
    )

    popular = popular.merge(books, on="ISBN")

    popular = popular.sort_values(
        by="num_ratings", ascending=False
    ).head(n)

    return popular

# ------------------ STREAMLIT UI ------------------
st.subheader("🔥 Popular Books")

num_books = st.slider("Select number of recommendations", 5, 20, 10)

popular_books = get_popular_books(num_books)

cols = st.columns(5)
for i, (_, row) in enumerate(popular_books.iterrows()):
    with cols[i % 5]:
        st.markdown(f"**{row['Book-Title']}**")
        st.caption(row.get("Book-Author", "Unknown"))
        if "Image-URL-M" in row:
            st.image(row["Image-URL-M"], use_container_width=True)

# ------------------ DATA PREVIEW ------------------
with st.expander("📊 Preview Dataset"):
    st.write(books.head())
