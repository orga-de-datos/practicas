# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm

sns.set()
# -

# # Aprendizaje no supervisado

# - Problemas de reconocimiento de patrones donde el grupo de puntos de entrenamineto no tienen una variable target
# - Quiero descubrir grupos de datos con características similares
# - No tengo una medida de éxito directa
# - Muchos métodos son heurísticas: estrategias, reglas, silogismos.

# Generamos puntos en un espacio 2D que correspondan a 4 clusters, para ello usamos make_blobs de sklearn:

X, y_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.figure(figsize=(20, 10))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")

# ## K-means

# Usamos la herramienta KMeans, que busca los clusteres de puntos, retornando los labels de cada punto (a qué clase pertenece) y los centroides de dichos clusteres.

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# +
plt.figure(figsize=(20, 10))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=90, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")

# -

clusters = np.unique(y_kmeans)
clusters

# Pero, ¿cómo funciona el algoritmo?

# +
from sklearn.metrics import pairwise_distances_argmin


def kmeans_distorsion(X, centers):
    distortion = 0
    labels = pairwise_distances_argmin(X, centers)
    for i in range(len(labels)):
        distortion += np.linalg.norm(X[i] - centers[labels[i]])
    return distortion


def find_clusters(X, n_clusters, rseed=2):
    # 1. Seleccionamos de manera aleatoria los primeros valores de los centroides
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    j = 0
    while True:
        """2a. Asignamos a qué cluster pertenece según su cercanía al centroide
           pairwise_distances_argmin retorna array de indices, cada índice corresponde
           con el índice del controide más cercano para ese punto"""

        labels = pairwise_distances_argmin(X, centers)

        """2b. Buscamos los nuevo centroides, calculados como el promedio
           (en cada dimension) de los puntos de cada cluster"""
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        """ 2c. si los centroides no cambiaron respecto al paso anterior, paro"""
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels, kmeans_distorsion(X, centers)


plt.figure(figsize=(20, 10))
centers, labels, error = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=90, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")

# -

# Observamos ahora como se mueven los centroides en las iteraciones del código anterior:

# +
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from functools import partial


def find_clusters_limited_iters(X, n_clusters, iters, rseed=2):
    """1. Seleccionamos de manera aleatoria los primeros valores de los centroides"""
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    j = 0
    for i in range(iters):
        """2a. Asignamos a qué cluster pertenece según su cercanía al centroide
           pairwise_distances_argmin retorna array de indices, cada índice corresponde con el índice del controide
           más cercano para ese punto"""

        labels = pairwise_distances_argmin(X, centers)

        """2b. Buscamos los nuevo centroides, calculados como el promedio (en cada dimension) de los puntos de
           cada cluster"""
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        # 2c. si los centroides no cambiaron respecto al paso anterior, stop
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels, kmeans_distorsion(X, centers)


def plot_clusters(X, n_clusters, iters, rseed=2):
    plt.figure(figsize=(20, 10))
    centers, labels, error = find_clusters_limited_iters(
        X, n_clusters, iters, rseed=rseed
    )
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=90, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.ylabel("X2", fontsize=20, weight="bold")
    plt.xlabel("X1", fontsize=20, weight="bold")
    plt.title("Distortion: %f" % error, fontsize=20)
    plt.show()


interact(
    plot_clusters, X=fixed(X), n_clusters=fixed(4), rseed=fixed(2), iters=(1, 10, 1)
)
# -

# ### Matriz de similaridad

ordered_points = X[np.where(y_kmeans == clusters[0])]
for i in clusters[1:]:
    points_cluster_i_index = np.where(y_kmeans == i)
    points_cluster_i = X[points_cluster_i_index]
    ordered_points = np.vstack((ordered_points, points_cluster_i))

distance_matrix = euclidean_distances(ordered_points, ordered_points)

fig, ax = plt.subplots(figsize=(13, 13))
cax = ax.matshow(
    distance_matrix, interpolation='nearest', cmap=matplotlib.cm.Spectral_r
)
fig.colorbar(cax)
ax.grid(True)
plt.title("Matriz de similaridad: Distancia euclideana", fontsize=24, weight="bold")
plt.show()

# ¿Qué pasa cuando la inicialización de los centroides no es muy feliz?

centers, labels, error = find_clusters(X, 4, rseed=0)
plt.figure(figsize=(20, 10))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=90, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")

interact(
    plot_clusters, X=fixed(X), n_clusters=fixed(4), rseed=fixed(0), iters=(1, 10, 1)
)

clusters = np.unique(labels)
ordered_points = X[np.where(y_kmeans == clusters[0])]
for i in clusters[1:]:
    points_cluster_i_index = np.where(labels == i)
    points_cluster_i = X[points_cluster_i_index]
    ordered_points = np.vstack((ordered_points, points_cluster_i))

distance_matrix = euclidean_distances(ordered_points, ordered_points)

fig, ax = plt.subplots(figsize=(13, 13))
cax = ax.matshow(
    distance_matrix, interpolation='nearest', cmap=matplotlib.cm.Spectral_r
)
fig.colorbar(cax)
ax.grid(True)
plt.title("Distancia euclideana", fontsize=24, weight="bold")
plt.show()

# ## Clustering aglomerativo

# - No tengo que explícitamente dar número de clusters k* de antemano
# - Se produce una representación jerárquica (dendrograma).
# - Clústeres de un nivel superior se forman por la unión de clústeres de niveles inferiores.
# - En el nivel más inferior tengo clusteres formados por 1 punto dato, en el nivel más superior tendo 1 cluster formado por todos los puntos dato.

# Usamos [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering) de sklearn: "Recursively merges the pair of clusters that minimally increases a given linkage distance"

from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(n_clusters=4).fit(X)

clustering.labels_

plt.figure(figsize=(20, 10))
plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, s=90, cmap='viridis')
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")


# +
def plot_agglometive(X, n_clusters):
    plt.figure(figsize=(20, 10))
    clustering = AgglomerativeClustering(n_clusters).fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, s=90, cmap='viridis')
    plt.ylabel("X2", fontsize=20, weight="bold")
    plt.xlabel("X1", fontsize=20, weight="bold")
    plt.show()


interact(plot_agglometive, X=fixed(X), n_clusters=(1, 20, 1))
# -

# Este método nos permite tener un ordenamiento jerárquico de las observaciones en lo que llamamos un dendrograma: nos indica qué grupo de observaciones es más parecida a otra. Observaciones que se unen más abajo en el dendrograma son más similares.

from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X)
plt.figure(figsize=(20, 10))
plot_dendrogram(model, truncate_mode='level', p=3)
plt.title('Hierarchical Clustering Dendrogram', fontsize=24, weight="bold")
plt.xlabel(
    "Number of points in node (or index of point if no parenthesis).",
    fontsize=16,
    weight="bold",
)

from sklearn.datasets import make_moons

X, y = make_moons(200, noise=0.05, random_state=0)

# Ahora vemos que pasa cuando se complican un poco más los datos:

plt.figure(figsize=(20, 10))
plt.scatter(X[:, 0], X[:, 1], edgecolor='black', s=90, cmap='viridis')
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")

# Qué pasa si aplicamos kmeans a esto?

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.predict(X)
centers = kmeans.cluster_centers_
plt.figure(figsize=(20, 10))
plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolor='black', s=90, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")

cluster_0_id = np.where(labels == 0)
cluster_1_id = np.where(labels == 1)
cluster_0 = X[cluster_0_id]
cluster_1 = X[cluster_1_id]

ordered_points = np.append(cluster_0, cluster_1, axis=0)

distance_matrix = euclidean_distances(ordered_points, ordered_points)

fig, ax = plt.subplots(figsize=(13, 13))
cax = ax.matshow(
    distance_matrix, interpolation='nearest', cmap=matplotlib.cm.Spectral_r
)
fig.colorbar(cax)
ax.grid(True)
plt.title("Matriz de similaridad: Distancia euclideana", fontsize=24, weight="bold")
plt.show()

# Vemos que los clusters que encuentra el algoritmo no son los correctos. :(
# Que nos da Agglomerative clustering?

from sklearn.neighbors import kneighbors_graph


# +
def plot_agglometive(X, n_clusters):
    plt.figure(figsize=(20, 10))
    clustering = AgglomerativeClustering(n_clusters).fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, s=90, cmap='viridis')
    plt.ylabel("X2", fontsize=20, weight="bold")
    plt.xlabel("X1", fontsize=20, weight="bold")
    plt.show()


interact(plot_agglometive, X=fixed(X), n_clusters=(1, 20, 1))
# -

# ## DBSCAN
# - No tengo que especificar la cantidad de clusters.
# - Para este método los clusters son zonas de alta densidad de puntos, separados por zonas con baja densidad de puntos.
# - Este método clasifica los puntos en: puntos borde, puntos core o puntos ruido(noise)

# Que pasa con DBSCAN?
#
# Usamos la herramienta DBSCAN de sklearn

from sklearn.cluster import DBSCAN

# Los parámetros más importantes son:
#  - **eps**: da la noción de cercanía, es la máxima distancia dentro de la cual 2 puntos son considerados vecinos.
#  - **min_samples**: cantidad de vecinos que tiene que tener un punto para se considerado como **core point**.

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# DBSCAN nos da los *core points* y los *noise points*, además de lac antidad de cluster encontrados y los labels para cada punto (a qué cluster pertenece)

db.core_sample_indices_

# Los noise points tendrán índice -1 al acceder a labels_ (ojo!)

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

db.labels_

n_clusters_

# +
unique_labels = set(labels)
plt.figure(figsize=(20, 10))
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        'o',
        markerfacecolor=tuple(col),
        markeredgecolor='k',
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        'o',
        markerfacecolor=tuple(col),
        markeredgecolor='k',
        markersize=6,
    )

plt.title("DBSCAN, eps=0.3, min_samples=10", fontsize=20, weight="bold")
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")
# -

# Cambio **eps**, noción de cernanía.

db = DBSCAN(eps=0.2, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

n_clusters_

# +
unique_labels = set(labels)
plt.figure(figsize=(20, 10))
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        'o',
        markerfacecolor=tuple(col),
        markeredgecolor='k',
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        'o',
        markerfacecolor=tuple(col),
        markeredgecolor='k',
        markersize=6,
    )

plt.title("DBSCAN, eps=0.2, min_samples=5", fontsize=20, weight="bold")
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")
# -

db.labels_

cluster_0_id = np.where(db.labels_ == 0)
cluster_1_id = np.where(db.labels_ == 1)

cluster_0 = X[cluster_0_id]

cluster_1 = X[cluster_1_id]

ordered_points = np.append(cluster_0, cluster_1, axis=0)

distance_matrix = euclidean_distances(ordered_points, ordered_points)

fig, ax = plt.subplots(figsize=(13, 13))
cax = ax.matshow(
    distance_matrix, interpolation='nearest', cmap=matplotlib.cm.Spectral_r
)
fig.colorbar(cax)
ax.grid(True)
plt.title("Distancia euclideana", fontsize=24, weight="bold")
plt.show()

# Aumento **min_samples**

db = DBSCAN(eps=0.2, min_samples=13).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

n_clusters_

# +
unique_labels = set(labels)
plt.figure(figsize=(20, 10))
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        'o',
        markerfacecolor=tuple(col),
        markeredgecolor='k',
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        'o',
        markerfacecolor=tuple(col),
        markeredgecolor='k',
        markersize=6,
    )

plt.title("DBSCAN, eps=0.2, min_samples=5", fontsize=20, weight="bold")
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")
# -

# Muchos menos core points, más clusters, más noise points

# ## Robustez

X, y_labels = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
plt.figure(figsize=(20, 10))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")

# ¿Qué pasa si tenemos outliers?

X_out = np.vstack((X, ([-3, 6], [-4, 5], [-3.5, 4.5])))

plt.figure(figsize=(20, 10))
plt.scatter(X_out[:, 0], X_out[:, 1], s=50)
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")

# +
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.figure(figsize=(20, 10))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=90, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")


# +
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_out)
y_kmeans = kmeans.predict(X_out)
plt.figure(figsize=(20, 10))
plt.scatter(X_out[:, 0], X_out[:, 1], c=y_kmeans, s=90, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")

# -

X_out = np.vstack((X_out, ([-25, 3.5])))

# +
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_out)
y_kmeans = kmeans.predict(X_out)
plt.figure(figsize=(20, 10))
plt.scatter(X_out[:, 0], X_out[:, 1], c=y_kmeans, s=90, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")

# -

# Vemos que el outlier hizo que uno de los clusters, por lo que este método es bastante sensible a outliers. Pensar que con mayor cantidad de dimensiones es más complicado ver outliers. Usando algoritmos jerárquicos se tiene el mismo problema:

clustering = AgglomerativeClustering(3).fit(X_out)
plt.figure(figsize=(20, 10))
plt.scatter(X_out[:, 0], X_out[:, 1], c=clustering.labels_, s=90, cmap='viridis')
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")
plt.show()

# En cambio, BDSCAN al poder clasificar algunos puntos como outliers, puede manejar este tipo de situaciones:

db = DBSCAN(eps=0.7, min_samples=10).fit(X_out)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# +
unique_labels = set(labels)
plt.figure(figsize=(20, 10))
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X_out[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        'o',
        markerfacecolor=tuple(col),
        markeredgecolor='k',
        markersize=14,
    )

    xy = X_out[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        'o',
        markerfacecolor=tuple(col),
        markeredgecolor='k',
        markersize=6,
    )

plt.title("DBSCAN, eps=0.2, min_samples=5", fontsize=20, weight="bold")
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")
# -

# Finalmente, notar que los centroides de kmeans son puntos representativos del cluster. Por ejemplo, si corremos kmeans en el dataset de MNIST:

from sklearn.datasets import load_digits

digits = load_digits()
digits.data.shape

digits.data

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = digits.data[:10].reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
