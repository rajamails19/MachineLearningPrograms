import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset: Movies and their genres
movies = pd.DataFrame({
    'Movie': ["Inception", "Interstellar", "Titanic", "Avatar", "The Dark Knight"],
    'Genres': ["Sci-Fi Action", "Sci-Fi Adventure", "Romance Drama", "Sci-Fi Action", "Action Thriller"]
})

# Convert genres into one-hot encoding
genres_encoded = pd.get_dummies(movies['Genres'])

# Compute similarity matrix
similarity_matrix = cosine_similarity(genres_encoded)

# Function to get similar movies
def get_recommendations(movie_name):
    idx = movies[movies["Movie"] == movie_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    print(f"Movies similar to {movie_name}:")
    for i in scores[1:]:
        print("-", movies.iloc[i[0]]["Movie"])

get_recommendations("Inception")
