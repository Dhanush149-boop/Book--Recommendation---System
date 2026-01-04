import streamlit as st
import pandas as pd
from pathlib import Path

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# ------------------ HEADER ------------------
st.markdown(
    """
    <h1 style='text-align:center;'>📚 Book Recommendation System</h1>
    <p style='text-align:center; color:gray;'>
    Discover books you’ll love
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ------------------ DATA LOADING ------------------
DATA_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    books = pd.read_csv(DATA_DIR / "Books.csv", encoding="latin-1")
    ratings = pd.read_csv(DATA_DIR / "Ratings.csv", encoding="latin-1")
    users = pd.read_csv(DATA_DIR / "Users.csv", encoding="latin-1")
    return books, ratings, users

books, ratings, users = load_data()

st.success("✅ Data loaded successfully!")

# ------------------ BOOK CARDS ------------------
st.subheader("📖 Popular Books")

# Limit books for UI performance
books_display = books[['Book-Title', 'Year-Of-Publication', 'Publisher', 'Image-URL-M']].head(12)

cols = st.columns(4)  # 4 cards per row

for idx, row in books_display.iterrows():
    with cols[idx % 4]:
        st.markdown(
            f"""
            <div style="
                border:1px solid #ddd;
                border-radius:10px;
                padding:10px;
                margin-bottom:20px;
                text-align:center;
                box-shadow:2px 2px 8px rgba(0,0,0,0.05);
            ">
                <img src="{row['Image-URL-M']}" style="height:200px; margin-bottom:10px;">
                <h4>{row['Book-Title']}</h4>
                <p style="font-size:14px;">
                    📅 {row['Year-Of-Publication']}<br>
                    🏢 {row['Publisher']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------ FOOTER ------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray; font-size:14px;'>
    Built with ❤️ using Streamlit | Book Recommendation System
    </p>
    """,
    unsafe_allow_html=True
)
