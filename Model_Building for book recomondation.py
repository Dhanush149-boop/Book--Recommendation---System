#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


user = pd.read_csv("data/Users.csv", encoding="latin1", engine="python", on_bad_lines="skip")
book = pd.read_csv("data/Books.csv", encoding='latin1')
rating=pd.read_csv("data/Ratings.csv",encoding='latin1')


# In[3]:


user.head()


# In[4]:


book.head()


# In[5]:


rating.head()


# In[6]:


rating_up = rating.merge(book[['ISBN', 'Book-Title']], on='ISBN', how='left')
rating_up


# In[7]:


null_values1 = rating_up['Book-Title'].isnull().sum()
null_values2 = len(rating_up)


# In[8]:


print(f'These are the number of non null records : {null_values2 - null_values1}')


# In[9]:


rating_up.dropna(subset=['Book-Title'],inplace=True)
rating_up.head()


# In[10]:


len(rating_up)


# In[11]:


from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# In[12]:


rating_up['user_index'] = rating_up['User-ID'].astype('category').cat.codes
rating_up['book_index'] = rating_up['Book-Title'].astype('category').cat.codes


# In[13]:


user_book_matrix = csr_matrix(
    (rating_up['Book-Rating'], (rating_up['user_index'], rating_up['book_index']))
)
print(f"User-Book Matrix Shape: {user_book_matrix.shape}")


# In[15]:


from sklearn.neighbors import NearestNeighbors 

user_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1) 
user_knn.fit(user_book_matrix) 

distances, indices = user_knn.kneighbors(user_book_matrix[3], n_neighbors=5) 
print(indices) 
print(distances) 


# In[17]:


user_id_mapping = dict(enumerate(rating_up['User-ID'].astype('category').cat.categories)) 
book_title_mapping = dict(enumerate(rating_up['Book-Title'].astype('category').cat.categories))

similar_users = [user_id_mapping[i] for i in indices.flatten()]


# In[18]:


# Example: get 10 most similar users to user_index 0 
target_user_index = 0
target_user_id = user_id_mapping[target_user_index]

similar_users_books = rating_up[rating_up['User-ID'].isin(similar_users)] 
target_user_books = set(rating_up[rating_up['User-ID'] == target_user_id]['Book-Title']) 


# In[19]:


# Step 6: Recommend books that similar users liked (say rating ≥ 8) 
recommended_books = ( similar_users_books[
                     similar_users_books['Book-Rating'] >= 5]
                     .loc[~similar_users_books['Book-Title']
                     .isin(target_user_books)]
                     .groupby('Book-Title')['Book-Rating']
                     .mean().sort_values(ascending=False).head(10)) 
print(f"Top recommendations for User {target_user_id}:") 
print(recommended_books) 


# In[20]:


book_user_matrix = csr_matrix( (rating_up['Book-Rating'], (rating_up['book_index'], rating_up['user_index'])) ) 
print(book_user_matrix.shape)


# In[21]:


book_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1) 
book_knn.fit(book_user_matrix) 

# Example: get similar books to the first book (index 0) 
distances, indices = book_knn.kneighbors(book_user_matrix[0], n_neighbors=6) 
print(indices) 
print(distances) 


# In[22]:


book_mapping = dict( zip(rating_up['book_index'], rating_up['Book-Title']) ) 
book_id = 0 # example book index 

similar_books = indices[0][1:] 
# exclude itself 

print("Target Book:", book_mapping[book_id]) 
print("Similar Books:") 
for idx in similar_books: 
    print("-", book_mapping[idx])


# In[23]:


user_index_mapping = {v: k for k, v in user_id_mapping.items()}
book_index_mapping = {v: k for k, v in book_mapping.items()}


# In[29]:


def hybrid_recommend(target_user_id, rating_df, top_n=10):
    # --- Step A: Find target user index ---
    if target_user_id not in user_index_mapping:
        print("User not found.")
        return

    target_user_index = user_index_mapping[target_user_id]

    # --- Step B: Find similar users ---
    distances, indices = user_knn.kneighbors(user_book_matrix[target_user_index], n_neighbors=6)
    similar_users = [user_id_mapping[i] for i in indices.flatten() if i != target_user_index]
    print(f"\n👥 Similar Users to {target_user_id}: {similar_users}")

    # --- Step C: Books read and rated by target user ---
    target_user_data = rating_df[rating_df['User-ID'] == target_user_id][['Book-Title', 'Book-Rating']]
    target_user_books = set(target_user_data['Book-Title'])

    print(f"\n📖 Books read by User {target_user_id}:")
    if target_user_data.empty:
        print("This user has not rated any books yet.")
    else:
        for _, row in target_user_data.iterrows():
            print(f"- {row['Book-Title']} (Rating: {row['Book-Rating']})")

    # --- Step D: Books liked by similar users (User–User based CF) ---
    similar_users_books = rating_df[rating_df['User-ID'].isin(similar_users)]
    recommended_books_user = (
        similar_users_books[similar_users_books['Book-Rating'] >= 5]
        .loc[~similar_users_books['Book-Title'].isin(target_user_books)]
        .groupby('Book-Title')['Book-Rating']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    print("\n📘 Top recommendations from similar users (User–User CF):")
    if recommended_books_user.empty:
        print("No new books found from similar users.")
    else:
        for book, rating in recommended_books_user.items():
            print(f"- {book} (avg rating: {rating:.2f})")

    # --- Step E: Books similar to what target user read (Item–Item based CF) ---
    similar_books_set = set()
    for book in target_user_books:
        if book in book_index_mapping:
            book_id = book_index_mapping[book]
            distances, indices = book_knn.kneighbors(book_user_matrix[book_id], n_neighbors=4)
            for idx in indices.flatten()[1:]:
                similar_books_set.add(book_mapping[idx])

    print("\n📙 Books similar to what the user has read (Item–Item CF):")
    if not similar_books_set:
        print("No similar books found.")
    else:
        similar_books_ratings = (
            rating_df[rating_df['Book-Title'].isin(similar_books_set)]
            .groupby('Book-Title')['Book-Rating']
            .mean()
            .sort_values(ascending=False)
        )

        for book, avg_rating in similar_books_ratings.head(top_n).items():
            print(f"- {book} (avg rating: {avg_rating:.2f})")

    # --- Step F: Combine both sources ---
    final_recommendations = set(recommended_books_user.index).union(similar_books_set)

    print(f"\n📚 Final Combined Recommendations for User {target_user_id}:")
    if not final_recommendations:
        print("No recommendations available.")
    else:
        for book in list(final_recommendations)[:top_n]:
            print("-", book)


# In[31]:


hybrid_recommend(target_user_id=276709, rating_df=rating_up, top_n=10)


# In[ ]:




