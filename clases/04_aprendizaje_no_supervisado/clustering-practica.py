# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm

sns.set()
# -

from PIL import Image

# Generamos puntos en un espacio 2D que correspondan a 4 clusters, para ello usamos make_blobs de sklearn:

X, y_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.figure(figsize=(20, 10))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.ylabel("X2", fontsize=20, weight="bold")
plt.xlabel("X1", fontsize=20, weight="bold")

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
plt.title("Distancia euclideana", fontsize=24, weight="bold")
plt.show()

# Pero, ¿cómo funciona el algoritmo?

# +
from sklearn.metrics import pairwise_distances_argmin


def find_clusters(X, n_clusters, rseed=2):
    # 1. Seleccionamos de manera aleatoria los primeros valores de los centroides
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    j = 0
    while True:
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

    return centers, labels


plt.figure(figsize=(20, 10))
centers, labels = find_clusters(X, 4)
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
    # 1. Seleccionamos de manera aleatoria los primeros valores de los centroides
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

    return centers, labels


def plot_clusters(X, n_clusters, iters, rseed=2):
    plt.figure(figsize=(20, 10))
    centers, labels = find_clusters_limited_iters(X, n_clusters, iters, rseed=rseed)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=90, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.ylabel("X2", fontsize=20, weight="bold")
    plt.xlabel("X1", fontsize=20, weight="bold")
    plt.show()


interact(
    plot_clusters, X=fixed(X), n_clusters=fixed(4), rseed=fixed(2), iters=(1, 10, 1)
)
# -

# ¿Qué pasa cuando la inicialización de los centroides no es muy feliz?

centers, labels = find_clusters(X, 4, rseed=0)
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
plt.title("Distancia euclideana", fontsize=24, weight="bold")
plt.show()

# Vemos que los clusters que encuentra el algoritmo no son los correctos. :(
#
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

# Finalmente, notar que los centroides de kmeans son puntos representativos del cluster. Por ejemplo, si corremos kmeans en el dataset de MNIST:

from sklearn.datasets import load_digits

digits = load_digits()
digits.data.shape

digits.data

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

# Vemos como en promedio es escrito cada número.
