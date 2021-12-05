# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Metricas
# - [Referencia](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)
#
# Lo primero que tenemos que pensar es como medir: mas allá del modelo, se entrena y evalúa del mismo modo que lo que ya hemos visto?
#
# ## Precision@K
#
# Tomamos las top-N recomendaciones y medimos cuantas de esas son _relevantes_.
#
# ## Recall@K
# Tomamos las top-N recomendaciones y medimos que proporcion de las _relevantes_ son.
#
# Pero estas metricas no consideran el orden. Quisieramos recomendar primero las cosas mas relevantes. Estas metricas son __rank aware__.
#
# ## MRR: Mean reciprocal rank
# Encontrar la ubicacion del primer item relevante. $$MRR = \frac{1}{\mid U \mid} \sum_{u \in U} k_u$$ donde $k_u$ es la ubicacion del primer elemento relevante entre las recomendaciones para el usuario $u$.
#
# Pero no consideramos el resto de la lista de recomendaciones. Quizas para un servicio de streaming o un ecommerce no sea una buena metrica, pero para un buscador si lo sea.
#
# ## MAP@N: Mean Average Precision
# Para cada usuario recomendamos una lista ordenada de largo N. A diferencia de Precision@N, queremos considerar el hecho de que es una lista ordenada. Queremos equivocarnos **poco en las primeras recomendaciones**.
#
# Calculamos el promedio del **AP@N** de todos los usuarios. Para cada elemento relevante recomendado, calculamos la precision hasta ese elemento. Luego, dividimos por la cantidad de elementos relevantes recomendados.
#
# ## NDCG@N: Ganancia acumulada de descuento normalizado
# La idea es usar la graduacion de la ganancia y moverse del esquema "es/no es relevante"
#
# $$DCG_p = \sum_{i=1}^p \frac{rel_i}{\log_2 (i+1)}$$
#
# Donde $p$ es la cantidad de elementos en la lista recomendada ordenada. $rel_i$ es la ganancia en la posicion $i$.
#
# $$IDCG_p = \sum_{i=1}^{\mid REL_p \mid} \frac{rel_i}{\log_2 (i+1)}$$
#
# Donde $\mid REL_p \mid$ es la lista ordenada de documentos relevantes hasta la posicion p.
#
# $$nDCG_p = \frac{DCG_p}{IDCG_p}$$
#
# Luego promediamos entre todos los usuarios.
#
#
#
#
#
#
#

# + tags=[] jupyter={"source_hidden": true}
from typing import Sequence, Tuple
import numpy as np

# source: https://github.com/raminqaf/Metrics/blob/fix/average-percision-calculation/Python/ml_metrics/average_precision.py
def apk(actual: list, predicted: list, k=10) -> float:
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted
    predicted : list
             A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : float
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    sum_precision = 0.0
    num_hits = 0.0

    for i, prediction in enumerate(predicted):
        if prediction in actual[:k] and prediction not in predicted[:i]:
            num_hits += 1.0
            precision_at_i = num_hits / (i + 1.0)
            sum_precision += precision_at_i

    if num_hits == 0.0:
        return 0.0

    return sum_precision / num_hits


def mapk(actual: list, predicted: list, k=10) -> float:
    """
    Computes the mean average precision at k.
    This function computes the mean average precision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# -

# # Entrenamiento
#
# Tenemos que considerar algunas cosas diferentes:
#
# ## Censura de reviews
# Si queremos predecir un score, solo podemos medir contra lo que los usuarios ya han visto. No podemos medir el contrafactico. Entonces, enmascaramos items a los que los usuarios le hayan dado feedback explicito y predecimos que puntaje le darian.
#
# ## Corte temporal
# Nos quedamos con la actividad hasta un tiempo $t$ y evaluamos las recomendaciones contra la actividad posterior. Hay que considerar el caso donde no tenga actividad el usuario, por ejemplo.
#
# ## Cold start
# Como incorporamos o predecimos para usuarios que nunca vimos hasta el momento?

# # Ejemplo de dataset para recsys
#
# Exploremos un poco el dataset [movielens 100k](https://grouplens.org/datasets/movielens/100k/)

# +
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
import pandas as pd

# Load the movielens-100k dataset (download it if needed),
_ = Dataset.load_builtin('ml-100k')

# + tags=[] jupyter={"source_hidden": true}
from pathlib import Path

file_path = Path('~/.surprise_data/ml-100k/ml-100k/')

# + tags=[] jupyter={"source_hidden": true}
ratings = pd.read_csv(
    file_path / 'u.data',
    names=["user_id", "item_id", "rating", "timestamp"],
    sep='\t',
    parse_dates=['timestamp'],
)

# + tags=[] jupyter={"source_hidden": true}
movies = pd.read_csv(
    file_path / 'u.item',
    names=[
        'movie_id',
        'movie_title',
        'release_date',
        'video_release_date',
        'IMDb_URL',
        'unknown',
        'Action',
        'Adventure',
        'Animation',
        "Childrens",
        'Comedy',
        'Crime',
        'Documentary',
        'Drama',
        'Fantasy',
        'FilmNoir',
        'Horror',
        'Musical',
        'Mystery',
        'Romance',
        'Sci-Fi',
        'Thriller',
        'War',
        'Western',
    ],
    sep="|",
    encoding="latin1",
    parse_dates=['release_date', 'video_release_date'],
)

# + tags=[] jupyter={"source_hidden": true}
users = pd.read_csv(
    file_path / 'u.user',
    names=["user_id", "age", "gender", "occupation", "zip_code"],
    sep='|',
)
# -

ratings.head()

movies.head()

users.head()

# Vemos que los ratings estan en _long form_. Lo pasamos a _wide form_ con `pivot`.

ratings.pivot(index="user_id", columns="item_id", values="rating")

f'Peliculas en promedio que vio cada usuario: {ratings.groupby("user_id").item_id.agg("count").values.mean()}'

f"De entre {ratings.item_id.nunique()} peliculas"

ratings.pivot(index="user_id", columns="item_id", values="rating").info(
    memory_usage='deep'
)

(943 * 1682 * 8 / 1024) / 1024

# Que porcentaje de None tenemos?

ratings.pivot(
    index="user_id", columns="item_id", values="rating"
).isna().values.flatten().mean()


# Podriamos:
# - Eliminar usuarios con poca actividad
# - Eliminar películas con pocas vistas

# # Algoritmos base

# ## Recomendar peliculas del mismo genero
# Al dar un puntaje a una pelicula, podemos recomendar peliculas del mismo genero, que el usuario no haya visto aun, ordenadas por su score promedio.


def recommend(user_id, movie_id):
    # movies of the same categories
    sim_movies = movies[
        (
            movies[movies.columns[6:]].values
            == movies[movies.movie_id == movie_id][movies.columns[6:]].values
        ).all(axis=1)
    ]
    # movies seen by the user
    user_movies = ratings[ratings.user_id == user_id]
    # filter movies seen by the user
    sim_movies = sim_movies[
        ~sim_movies.movie_id.isin([*user_movies.item_id.tolist(), movie_id])
    ]
    # add score
    sim_movies = sim_movies.merge(
        ratings.groupby('item_id').rating.mean(), left_on='movie_id', right_index=True
    )
    # return by rating ordered descending
    return sim_movies.sort_values('rating', ascending=False)


sample_rating = ratings.sample(1)
display(movies[movies.movie_id == sample_rating.item_id.iloc[0]].iloc[0])
recommend(user_id=sample_rating.user_id.iloc[0], movie_id=sample_rating.item_id.iloc[0])

# ## Buscar usuarios similares!
# Podriamos pensar en buscar los top _k_ usuarios mas similares y recomendar en base a los puntajes que le han dado a cosas que yo no haya visto.
# Similares como? Que hayan tenido interacciones/reseñas similares!

wfratings = ratings.pivot(index="user_id", columns="item_id", values="rating")
wfratings = wfratings.fillna(0)  # Queremos esto?

# +
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

similarities = cosine_similarity(wfratings, wfratings)
# -

user_id = 1
k = 5
a = similarities[user_id - 1, :]
most_similar_users = np.argpartition(a, -k - 1)[-k - 1 : -1][::-1]

ratings[ratings.user_id.isin([*most_similar_users, user_id])].pivot(
    index="user_id", columns="item_id", values="rating"
)

# Y teniendo los usuarios mas similares, como los usamos? Que hacemos?
# - Con un threshold, majority voting
# - Promedio de los que han visto?
# - Podemos considerar scores promedio de cada pelicula y las tendencias de cada user a scorear (siempre muy alto? siempre muy bajo?)

# # Algoritmos interesantes

# ## SVDpp
#
# Un algoritmo popularizado durante el [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_prize) fue el de SVD++. Lo que queremos es "llenar" todos esos `None` que vimos recién. Para esto descomponemos $S = Q \cdot P$, donde $S \in R^{u \cdot m}$, $Q \in R^{u \cdot f}$, $P \in R^{f \cdot m}$ y $f$ es la cantidad de _factores latentes_ que querramos usar.
#
# Esta matriz $U$ queremos que sea lo mas parecida a nuestra matriz original de score de review por usuario e item. Para llenar los `None`, tenemos que tener en cuenta que $Q_i$ nos da un "embedding" de las preferencias del usuario $i$ y $P_j$ nos da un "embedding" de preferencias de la pelicula $j$. Entonces, llenamos $U_{i,j}$ con $Q_i \cdot P_j$.
#
# Hasta aca tenemos un `SVD` silvestre de jardín, en `SVDpp` consideramos también los _ratings implicitos_ de usuarios y peliculas.

# ## NMF
# Muuuuuy parecido a `SVD`, pero los factores de $Q$ y $P$ se mantienen positivos.

# # Algunas bibliotecas

# ## [Surprise](https://surprise.readthedocs.io/en/stable/index.html)
# > Surprise is an easy-to-use Python scikit for recommender systems.

# +
from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k')

algo = SVDpp()

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# -

# Incluye [varios](https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html) algoritmos:
#
# - `random_pred.NormalPredictor`: Algorithm predicting a random rating based on the distribution of the training set, which is assumed to be normal.
# - `baseline_only.BaselineOnly`: Algorithm predicting the baseline estimate for given user and item.
# - `knns.KNNBasic`: A basic collaborative filtering algorithm.
# - `knns.KNNWithMeans`: A basic collaborative filtering algorithm, taking into account the mean ratings of each user.
# - `knns.KNNWithZScore`: A basic collaborative filtering algorithm, taking into account the z-score normalization of each user.
# - `knns.KNNBaseline`: A basic collaborative filtering algorithm taking into account a baseline rating.
# - `matrix_factorization.SVD`: The famous SVD algorithm, as popularized by Simon Funk during the Netflix Prize. When baselines are not used, this is equivalent to Probabilistic Matrix Factorization.
# - `matrix_factorization.SVDpp`: The SVD++ algorithm, an extension of SVD taking into account implicit ratings.
# - `matrix_factorization.NMF`: A collaborative filtering algorithm based on Non-negative Matrix Factorization.
# - `slope_one.SlopeOne`: A simple yet accurate collaborative filtering algorithm.
# - `co_clustering.CoClustering`: A collaborative filtering algorithm based on co-clustering.

# La [guia de inicio](https://surprise.readthedocs.io/en/stable/getting_started.html#getting-started) es muy comprensiva y nos da idea de sus capacidades para facilitar CV, GridSearch, etc

# ## [lightFM](https://github.com/lyst/lightfm)
#
# > LightFM is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback, including efficient implementation of BPR and WARP ranking losses. It's easy to use, fast (via multithreaded model estimation), and produces high quality results.

# +
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

# Load the MovieLens 100k dataset. Only five
# star ratings are treated as positive.
data = fetch_movielens(min_rating=5.0)

# Instantiate and train the model
model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

# Evaluate the trained model
test_precision = precision_at_k(model, data['test'], k=5).mean()
print(test_precision)
# -

# ## [Vowpal Wabbit](https://vowpalwabbit.org)
# En la clase de RL mencionamos brevemente que MAB se usan en sistemas de recomendacion de contenido.
#
# - [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/pdf/1003.0146.pdf)
#

# ## [W2V](https://radimrehurek.com/gensim/models/word2vec.html)
# Podemos usar el historial de reseñas para armar una secuencia y tratar de predecir los siguientes items.

sequences = (
    ratings.sort_values("timestamp").groupby("user_id").item_id.agg(list)
)  # les juro que esto queda bien ordenado al hacer groupby
sequences

# +
K = 10

train = sequences.apply(lambda x: x[:-K])
test = sequences.apply(lambda x: x[-K:])
# -

from gensim.models import Word2Vec
from IPython.display import display

model = Word2Vec(
    sentences=train.values,
    vector_size=10,
    window=100,
    min_count=1,
    workers=32,
    epochs=50,
)

# +
predictions = []
for i in train.index:
    predictions.append([x[0] for x in model.predict_output_word(train.loc[i], K)])

PatKs = [len(set(train_) & set(test_)) / K for train_, test_ in zip(test, predictions)]

print(f"MAP: {mapk(test.values.tolist(), predictions, k=K)}")
print("Precision at K:")
display(pd.Series(PatKs).value_counts())
print(f"Mean: {pd.Series(PatKs).mean()}")
# -

# # Links
# - [competitive-recsys: A collection of resources for Recommender Systems (RecSys)](https://github.com/chihming/competitive-recsys)
