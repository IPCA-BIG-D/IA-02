import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import random

# Path para os dados do dataset - MovieLens Small Latest Dataset (https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset)
movies_path = './movies_input/movies.csv'
ratings_path = './movies_input/ratings.csv'

# Lê os dados
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Junta os ratings com os filmes
data = pd.merge(ratings, movies, on='movieId')

# Filtra os filmes com um rating superior ou igual a "threshold"
threshold = 4
data = data[data['rating'] >= threshold]

# Definir uma semente para reprodutibilidade
random.seed(42)  

# Amostragem de utilizadores aleatória
user_subset = random.sample(data['userId'].unique().tolist(), k=min(1000, len(data['userId'].unique()))) # 1000 - sub conjunto de dados a utilizar
data_subset = data[data['userId'].isin(user_subset)]

# Amostragem de filmes aleatória
movie_subset = random.sample(data_subset['movieId'].unique().tolist(), k=min(100, len(data_subset['movieId'].unique()))) # 100 - sub conjunto de dados a utilizar
data_subset = data_subset[data_subset['movieId'].isin(movie_subset)]

# Agregar ratings de entradas duplicadas
data_subset = data_subset.groupby(['userId', 'title'])['rating'].max().reset_index()

# Cria uma matriz de preferência utilizador-filme
user_movie_matrix = data_subset.pivot(index='userId', columns='title', values='rating').fillna(0)

# Converter ratings para valores binários
basket_sets = user_movie_matrix.applymap(lambda x: 1 if x >= threshold else 0)

# Gerar conjuntos frequentes usando o algoritmo Apriori
min_support = 0.01
frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

# Gerar regras de associação
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Apresenta as regras
for index, rule in rules.iterrows():
    antecedent_movies = rule['antecedents']
    consequent_movies = rule['consequents']

    print(f"Rule {index + 1}:")
    print(f"  If the user likes movies: {', '.join(antecedent_movies)}")
    print(f"  Then the user might also like movies: {', '.join(consequent_movies)}")
    print(f"  Confidence: {rule['confidence']:.2f}")
    print(f"  Lift: {rule['lift']:.2f}")
    print("-" * 50)

# If Lift = 1: The antecedent and consequent are independent, and knowing one doesn't provide any information about the other.
# If Lift > 1: The antecedent and consequent are positively correlated. The higher the lift, the stronger the correlation.
# If Lift < 1: The antecedent and consequent are negatively correlated. A lift less than 1 indicates that the items are less likely to occur together than if they were independent.