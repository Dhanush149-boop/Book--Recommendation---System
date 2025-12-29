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
    required_files = ["Books", "Ratings", "Users"]

    for file in required_files:
        if not (DATA_DIR / file).exists():
            st.error(f"❌ Missing file: {file}")
            return None, None, None

    books = pd.read_csv(DATA_DIR / "Books")
    ratings = pd.read_csv(DATA_DIR / "Ratings")
    users = pd.read_csv(DATA_DIR / "Users")

    return books, ratings, users
