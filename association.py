import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import random

# Load the MovieLens Small Latest Dataset
# Replace 'path/to/movies.csv' and 'path/to/ratings.csv' with the actual paths to your downloaded files
movies_path = './movies_input/movies.csv'
ratings_path = './movies_input/ratings.csv'

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Merge movies and ratings data
data = pd.merge(ratings, movies, on='movieId')

# Filter movies rated higher than a certain threshold (e.g., 4)
threshold = 4
data = data[data['rating'] >= threshold]

# Sample a random subset of users
random.seed(42)  # Set a seed for reproducibility
user_subset = random.sample(data['userId'].unique().tolist(), k=min(1000, len(data['userId'].unique()))) #mudar o 1000
data_subset = data[data['userId'].isin(user_subset)]

# Sample a random subset of movies
movie_subset = random.sample(data_subset['movieId'].unique().tolist(), k=min(100, len(data_subset['movieId'].unique()))) #mudar o 100
data_subset = data_subset[data_subset['movieId'].isin(movie_subset)]

# Aggregate ratings for duplicate entries
data_subset = data_subset.groupby(['userId', 'title'])['rating'].max().reset_index()

# Create a binary matrix representing user-movie preferences
user_movie_matrix = data_subset.pivot(index='userId', columns='title', values='rating').fillna(0)

# Convert ratings to binary values (1 or 0)
basket_sets = user_movie_matrix.applymap(lambda x: 1 if x >= threshold else 0)

# Generate frequent itemsets using Apriori
min_support = 0.01
frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Print the association rules
for index, rule in rules.iterrows():
    antecedent_movies = rule['antecedents']
    consequent_movies = rule['consequents']

    print(f"Rule {index + 1}:")
    print(f"  If the user likes movies: {', '.join(antecedent_movies)}")
    print(f"  Then the user might also like movies: {', '.join(consequent_movies)}")
    print(f"  Confidence: {rule['confidence']:.2f}")
    print(f"  Lift: {rule['lift']:.2f}")
    print("-" * 50)