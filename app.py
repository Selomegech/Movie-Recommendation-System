from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
import requests

with open('model.pkl', 'rb') as file:
    svd = pickle.load(file)
ratings_df = pd.read_csv('ratings.csv')
ratings_pivot = ratings_df.groupby(['userId', 'movieId'])['rating'].mean().reset_index().pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Create ratings matrix
ratings_matrix = ratings_pivot.values

user_factors = svd.fit_transform(ratings_matrix)
movie_factors = svd.components_
predicted_ratings = user_factors.dot(movie_factors)
movies_df = pd.read_csv("movies.csv")
split_genres_df = movies_df['genres'].str.split('|', expand=True)

# Stack the columns to create a Series of genres
stacked_genres = split_genres_df.stack()

# Reset index and rename columns
stacked_genres_df = stacked_genres.reset_index()
stacked_genres_df.columns = ['movieId', 'genre_index', 'genre']
movies_with_genres_df = pd.merge(movies_df[['movieId', 'title']], stacked_genres_df, on='movieId')

# Drop the unnecessary column 'genre_index'
movies_with_genres_df = movies_with_genres_df.drop('genre_index', axis=1)
print(predicted_ratings)

# Load links.csv for TMDb IDs
links_df = pd.read_csv("links.csv")


def get_movie_recommendations(user_id, N=40):
    user_idx = ratings_pivot.index.get_loc(user_id)
    unrated_movies = ratings_pivot.loc[user_id][ratings_pivot.loc[user_id] == 0].index - 1
    valid_unrated_movies = list(set(unrated_movies).intersection(set(range(len(predicted_ratings[user_idx])))))
    predicted_ratings_for_user = predicted_ratings[user_idx][valid_unrated_movies]
    top_movie_indices = predicted_ratings_for_user.argsort()[::-1][:N]
    recommended_movie_ids = [valid_unrated_movies[i] + 1 for i in top_movie_indices]

    # Create DataFrame of recommendations
    recommendations_df = pd.DataFrame({'movieId': recommended_movie_ids,
                                       'predicted_rating': [predicted_ratings_for_user[i] for i in top_movie_indices]})

    # Merge with movies_df to get titles and genres
    recommendations_df = pd.merge(recommendations_df, movies_df[['movieId', 'title']], on='movieId')

    # Group by movieId and predicted_rating, aggregate genres
    recommendations_df = recommendations_df.groupby(['movieId', 'predicted_rating'])['title'].apply(lambda x: '|'.join(x)).reset_index()

    # Merge again with movies_df to get all genres for each movie
    recommendations_df = pd.merge(recommendations_df, movies_df[['movieId', 'genres']], on='movieId' )

    # Sort by predicted rating
    recommendations_df = recommendations_df.sort_values(by='predicted_rating', ascending=False)
    # recommendations_df = recommendations_df.to_dict(orient="records")
    print(recommendations_df)

    # recommendations_df = pd.DataFrame(recommendations_df)  # This line is crucial
    # recommendations_df.info()
    recommendations_df = pd.merge(recommendations_df, movies_df[['movieId', 'genres']], on='movieId' )

    # Convert recommendations to DataFrame (if it's not already)
    recommendations_df = pd.merge(recommendations_df, links_df[['movieId', 'tmdbId']], on='movieId')
    recommendations = recommendations_df.to_dict(orient="records")

    # Fetch poster paths from TMDb API
    for index, row in recommendations_df.iterrows():
        if pd.notna(row['tmdbId']):
            api_url = f"https://api.themoviedb.org/3/movie/{int(row['tmdbId'])}?api_key=ebfb7c60cae7f24a39b56dc15cbdce27"
            response = requests.get(api_url)
            if response.status_code == 200:
                data = response.json()
                recommendations_df.at[index, 'poster_path'] = data.get('poster_path', '')
            else:
                recommendations_df.at[index, 'poster_path'] = ''
        recommendations = recommendations_df.to_dict(orient="records")
    return recommendations





app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form['user_id']
        return redirect(url_for('result', user_id=user_id))
    return render_template('index.html')

@app.route('/result')
def result():
    user_id = request.args.get('user_id')
    print(user_id)
    user_id = int(user_id)
      # Get user_id from URL
    # Perform logic to fetch user data based on user_id (replace with your own logic)
    recommendations = get_movie_recommendations(user_id=user_id)
    if len(recommendations) <= 0:
        return render_template('index.html')
    print(recommendations, "recommendation")
    return render_template('result.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
