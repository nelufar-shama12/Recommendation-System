import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
movies = pd.DataFrame({
    "movie_id": [1,2,3,4,5],
    "title": ["Inception", "Interstellar", "The Matrix", "Avatar", "Titanic"],
    "genre": ["Sci-Fi Action", "Sci-Fi Drama", "Sci-Fi Action",
              "Fantasy Adventure", "Romance Drama"]
})

# User ratings dataset (Collaborative Filtering)
ratings = pd.DataFrame({
    "user": ["A","A","B","B","C","C","D"],
    "movie_id": [1,2,2,3,1,4,5],
    "rating": [5,4,5,4,4,5,5]
})



tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies["genre"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_content(movie_title, top_n=3):
    idx = movies[movies["title"] == movie_title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = [movies.iloc[i]["title"] for i,_ in scores[1:top_n+1]]
    return recommendations

user_movie_matrix = ratings.pivot_table(
    index="user", columns="movie_id", values="rating"
).fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)

def recommend_collaborative(user, top_n=3):
    user_idx = user_movie_matrix.index.tolist().index(user)
    sim_users = list(enumerate(user_similarity[user_idx]))
    sim_users = sorted(sim_users, key=lambda x: x[1], reverse=True)[1:]

    scores = {}
    for idx, sim in sim_users:
        sim_user = user_movie_matrix.index[idx]
        for movie_id, rating in user_movie_matrix.loc[sim_user].items():
            if user_movie_matrix.loc[user, movie_id] == 0:
                scores[movie_id] = scores.get(movie_id, 0) + sim * rating

    recommended_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return movies[movies["movie_id"].isin(recommended_ids)]["title"].tolist()
print("Content-Based Recommendation:")
print(recommend_content("Inception"))

print("\nCollaborative Recommendation:")
print(recommend_collaborative("A"))
