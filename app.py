import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Recommendation System")

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

# Dataset overview
st.subheader("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Books", books.shape[0])
col2.metric("Ratings", ratings.shape[0])
col3.metric("Users", users.shape[0])

# Preview data
st.subheader("🔍 Sample Data Preview")
with st.expander("📘 Books Dataset"):
    st.dataframe(books.head(10))

with st.expander("⭐ Ratings Dataset"):
    st.dataframe(ratings.head(10))

with st.expander("👤 Users Dataset"):
    st.dataframe(users.head(10))
