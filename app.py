import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os

# Title
st.title("Book Recommendation System")

# Data Upload
st.sidebar.header("Upload Your Data")
books_file = st.sidebar.file_uploader("Upload Books.csv", type=["csv"])
ratings_file = st.sidebar.file_uploader("Upload Ratings.csv", type=["csv"])
users_file = st.sidebar.file_uploader("Upload Users.csv", type=["csv"])

# Helper Functions
@st.cache_data
def load_data(books_file, ratings_file, users_file):
    books = pd.read_csv(books_file, encoding="latin1")
    ratings = pd.read_csv(ratings_file, encoding="latin1")
    users = pd.read_csv(users_file, encoding="latin1", engine="python", on_bad_lines="skip")
    return books, ratings, users

def build_matrices(books, ratings):
    ratings_up = ratings.merge(books[['ISBN', 'Book-Title']], on='ISBN', how='left')
    ratings_up.dropna(subset=['Book-Title'], inplace=True)
    ratings_up['user_index'] = ratings_up['User-ID'].astype('category').cat.codes
    ratings_up['book_index'] = ratings_up['Book-Title'].astype('category').cat.codes
    user_book_matrix = csr_matrix(
        (ratings_up['Book-Rating'], (ratings_up['user_index'], ratings_up['book_index']))
    )
    book_user_matrix = csr_matrix(
        (ratings_up['Book-Rating'], (ratings_up['book_index'], ratings_up['user_index']))
    )
    return ratings_up, user_book_matrix, book_user_matrix

def get_mappings(ratings_up):
    user_id_mapping = dict(enumerate(ratings_up['User-ID'].astype('category').cat.categories))
    book_title_mapping = dict(enumerate(ratings_up['Book-Title'].astype('category').cat.categories))
    user_index_mapping = {v: k for k, v in user_id_mapping.items()}
    book_index_mapping = {v: k for k, v in book_title_mapping.items()}
    return user_id_mapping, book_title_mapping, user_index_mapping, book_index_mapping

# Main
if books_file and ratings_file and users_file:
    books, ratings, users = load_data(books_file, ratings_file, users_file)
    st.success("Files uploaded and loaded successfully!")
    
    ratings_up, user_book_matrix, book_user_matrix = build_matrices(books, ratings)
    user_id_mapping, book_title_mapping, user_index_mapping, book_index_mapping = get_mappings(ratings_up)
    
    st.sidebar.subheader("Choose Recommendation Type")
    rec_type = st.sidebar.selectbox("Type", ["User-based", "Item-based", "Hybrid"])
    
    st.subheader("Select User")
    selected_user_id = st.selectbox("User-ID", sorted(ratings_up['User-ID'].unique()), index=0)
    top_n = st.slider("Number of Recommendations", 1, 20, 10)

    # Models
    user_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    user_knn.fit(user_book_matrix)
    book_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
    book_knn.fit(book_user_matrix)

    def recommend_user_based(target_user_id, ratings_up, user_book_matrix, user_knn, user_id_mapping, user_index_mapping, top_n):
        if target_user_id not in user_index_mapping:
            st.warning("User not found in dataset.")
            return []
        target_user_index = user_index_mapping[target_user_id]
        distances, indices = user_knn.kneighbors(user_book_matrix[target_user_index], n_neighbors=6)
        similar_users = [user_id_mapping[i] for i in indices.flatten() if i != target_user_index]
        target_user_books = set(ratings_up[ratings_up['User-ID'] == target_user_id]['Book-Title'])
        similar_users_books = ratings_up[ratings_up['User-ID'].isin(similar_users)]
        rec_books = (
            similar_users_books[similar_users_books['Book-Rating'] >= 5]
            .loc[~similar_users_books['Book-Title'].isin(target_user_books)]
            .groupby('Book-Title')['Book-Rating']
            .mean().sort_values(ascending=False).head(top_n)
        )
        return rec_books

    def recommend_item_based(target_user_id, ratings_up, book_knn, book_index_mapping, book_title_mapping, book_user_matrix, top_n):
        target_user_books = set(ratings_up[ratings_up['User-ID'] == target_user_id]['Book-Title'])
        similar_books_set = set()
        for book in target_user_books:
            if book in book_index_mapping:
                book_id = book_index_mapping[book]
                distances, indices = book_knn.kneighbors(book_user_matrix[book_id], n_neighbors=4)
                for idx in indices.flatten()[1:]:
                    similar_books_set.add(book_title_mapping[idx])
        similar_books_ratings = (
            ratings_up[ratings_up['Book-Title'].isin(similar_books_set)]
            .groupby('Book-Title')['Book-Rating']
            .mean()
            .sort_values(ascending=False)
        )
        return similar_books_ratings.head(top_n)

    def recommend_hybrid(target_user_id, ratings_up, user_book_matrix, user_knn, user_id_mapping, user_index_mapping, book_knn, book_index_mapping, book_title_mapping, book_user_matrix, top_n):
        user_recs = recommend_user_based(target_user_id, ratings_up, user_book_matrix, user_knn, user_id_mapping, user_index_mapping, top_n)
        item_recs = recommend_item_based(target_user_id, ratings_up, book_knn, book_index_mapping, book_title_mapping, book_user_matrix, top_n)
        final_set = set(user_recs.index).union(set(item_recs.index))
        hybrid_books = []
        for book in list(final_set)[:top_n]:
            avg_rating = ratings_up[ratings_up['Book-Title'] == book]['Book-Rating'].mean()
            hybrid_books.append((book, avg_rating))
        return hybrid_books

    st.subheader("Recommended Books")
    if rec_type == "User-based":
        recs = recommend_user_based(selected_user_id, ratings_up, user_book_matrix, user_knn, user_id_mapping, user_index_mapping, top_n)
        if not recs.empty:
            st.write("Top recommendations from similar users:")
            st.dataframe(recs)
        else:
            st.write("No recommendations found.")
    elif rec_type == "Item-based":
        recs = recommend_item_based(selected_user_id, ratings_up, book_knn, book_index_mapping, book_title_mapping, book_user_matrix, top_n)
        if not recs.empty:
            st.write("Top books similar to ones you've read:")
            st.dataframe(recs)
        else:
            st.write("No recommendations found.")
    else:
        recs = recommend_hybrid(
            selected_user_id, ratings_up, user_book_matrix, user_knn, user_id_mapping, user_index_mapping,
            book_knn, book_index_mapping, book_title_mapping, book_user_matrix, top_n
        )
        if recs:
            st.write("Hybrid recommendations:")
            st.table(pd.DataFrame(recs, columns=['Book-Title', 'Avg. Rating']))
        else:
            st.write("No recommendations found.")
else:
    st.info("Please upload Books.csv, Ratings.csv, and Users.csv files to begin.")

st.sidebar.markdown("---")
st.sidebar.info("App by [Your Name]. Upload the dataset and choose recommendation type to get started!")
