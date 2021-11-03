# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords

# ## Datos
#
# En esta notebook vamos a utilizar datos de Instagram. Los mismos fueron descargados de páginas públicas de Instagram. Tendremos 3 "tipos" de cuentas:
#
# - Fit
# - RecetasFit
# - Recetas
# - Jugadores de fútbol argentinos
# - Periodismo deportivo
# - Otros deportistas argentinos
#
# Descargaremos todos los posts que las cuentas hayan hecho en los últimos 12 meses.
#
# La idea será ver si estos tipos de cuentas comparten una misma semántica entre sí y si a la vez son muy distintas entre los grupos.

dataFit = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=1wEN85LBolVxFKKpNwWZwxb90do60okyN"
)
print(dataFit.columns)
print(dataFit.shape)
dataFit.head()

dataFit["User Name"].unique()

dataRecetasFit = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=13FL4Am8VRVPulISyobQf41IbgCgk2Egn"
)
print(dataRecetasFit.columns)
print(dataRecetasFit.shape)
dataRecetasFit.head()

dataRecetasFit["User Name"].unique()

dataRecetas = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=1k0rSIpL9ycPtGSjZoIDhy6wQG3l6KxuE"
)
print(dataRecetas.shape)
dataRecetas.head()

dataRecetas["User Name"].unique()

dataJugadoresArg = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=1YR1uT4USWgXzemIDWYaLwhSEmulwsUP6"
)
print(dataJugadoresArg.shape)
dataJugadoresArg.head()

dataJugadoresArg["User Name"].unique()

dataPeriodistmoDep = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=1szz4vhaIIi5QBxZM1ZpIrL0plkuyy6Ek"
)
dataPeriodistmoDep["User Name"].unique()

dataPeriodistmoDep.shape

dataotrosDeportistas = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=17-k6vXfQ34T02Mb5-BK-DaezStkj5aRB"
)
print(dataotrosDeportistas.shape)
dataotrosDeportistas["User Name"].unique()

# Unimos todos los dataframe en uno sólo en el siguiente orden: Fit, recetasFit, Recetas, FutbolArg, PeriodismoDep, otrosDeportistasArg

# +

data = pd.concat(
    [
        dataFit,
        dataRecetasFit,
        dataRecetas,
        dataJugadoresArg,
        dataPeriodistmoDep,
        dataotrosDeportistas,
    ]
)

print(data.shape)
data.head()
# -

# ## Bag of Words

# En esta parte vamos a comparar la importancia de cada token según el grupo de cuentas para estimar que palabras tienen mas "importancia" en cada uno.
#
# Para eso primero estimamos el BOW de los grupos que queremos analizar, en este primero caso, recetas general vs recetas fit.

# +
data['Description'] = data['Description'].fillna('')
texts2 = list(data[(data['User Name'].isin(dataRecetas["User Name"])) | (data['User Name'].isin(dataRecetasFit["User Name"]))]['Description'])

# Cuento los terminos
count_vect = CountVectorizer(ngram_range = (1,3), max_df = 0.8, min_df = 0.01, stop_words=stopwords.words('spanish'), lowercase=True)
x_counts = count_vect.fit_transform(texts2)
# -

# Luego estimamos los coeficientes BOW de cada uno, para eso sumamos todos los coeficientes de cada grupo de cuentas y escalamos los mismos entre 1 y 0

import numpy as np
datafiltered = data[(data['User Name'].isin(dataRecetas["User Name"])) | (data['User Name'].isin(dataRecetasFit["User Name"]))]
coefficients1 = np.sum(x_counts.toarray()[datafiltered['User Name'].isin(dataRecetas["User Name"])], axis=0)
coefficients2 = np.sum(x_counts.toarray()[datafiltered['User Name'].isin(dataRecetasFit["User Name"])], axis=0)

# +

coefficients1 = np.interp(coefficients1, (coefficients1.min(), coefficients1.max()), (0, +1))
coefficients2 = np.interp(coefficients2, (coefficients2.min(), coefficients2.max()), (0, +1))

# -

# Finalmente, hacemos un scatter plot, donde cada punto es un token y esta posicionado según su score en recetas general y recetas Fit. Siendo el eje X las general y el Y las FIT.

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
fig, ax = plt.subplots(dpi=(150))
ax.scatter(coefficients1, coefficients2, s=5)
for i, txt in enumerate(count_vect.get_feature_names()):
    if(coefficients1[i] > 0.3 or coefficients2[i] > 0.3 ):
        ax.annotate(txt, (coefficients1[i], coefficients2[i]), fontsize=6)
line = mlines.Line2D([0, 1], [0, 1], color='red')
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.ylabel("Recetas Fit")
plt.xlabel("Recetas comunes")
plt.title('Importancia de cada token')
plt.show()

# Podemos ver que palabras como manteca, chocolate, dulceo azúcar tienen muchísima importancia dentro del grupo de recetas comunes mientras que muy poco en las recetas Fit y lo mismo pero al reves sucede con "saludable".
#
# Repetimos lo mismo pero comparando entre otros grupos

# +
data['Description'] = data['Description'].fillna('')
texts2 = list(data[(data['User Name'].isin(dataFit["User Name"])) | (data['User Name'].isin(dataRecetasFit["User Name"]))]['Description'])

# Cuento los terminos
count_vect = CountVectorizer(ngram_range = (1,3), max_df = 0.8, min_df = 0.01, stop_words=stopwords.words('spanish'), lowercase=True)
x_counts = count_vect.fit_transform(texts2)
# -

datafiltered = data[(data['User Name'].isin(dataFit["User Name"])) | (data['User Name'].isin(dataRecetasFit["User Name"]))]

import numpy as np
coefficients1 = np.sum(x_counts.toarray()[datafiltered['User Name'].isin(dataFit["User Name"])], axis=0)
coefficients2 = np.sum(x_counts.toarray()[datafiltered['User Name'].isin(dataRecetasFit["User Name"])], axis=0)

# +

coefficients1 = np.interp(coefficients1, (coefficients1.min(), coefficients1.max()), (0, +1))
coefficients2 = np.interp(coefficients2, (coefficients2.min(), coefficients2.max()), (0, +1))

# -

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
fig, ax = plt.subplots(dpi=(150))
ax.scatter(coefficients1, coefficients2, s=5)
for i, txt in enumerate(count_vect.get_feature_names()):
    if(coefficients1[i] > 0.3 or coefficients2[i] > 0.3 ):
        ax.annotate(txt, (coefficients1[i], coefficients2[i]), fontsize=6)
line = mlines.Line2D([0, 1], [0, 1], color='red')
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.ylabel("Recetas Fit")
plt.xlabel("Cuentas Fit")
plt.title('Importancia de cada token')
plt.show()

# +
data['Description'] = data['Description'].fillna('')
texts2 = list(data[(data['User Name'].isin(dataFit["User Name"])) | (data['User Name'].isin(dataJugadoresArg["User Name"]))]['Description'])

# Cuento los terminos
count_vect = CountVectorizer(ngram_range = (1,3), max_df = 0.8, min_df = 0.01, stop_words=stopwords.words('spanish'), lowercase=True)
x_counts = count_vect.fit_transform(texts2)
# -

datafiltered = data[(data['User Name'].isin(dataFit["User Name"])) | (data['User Name'].isin(dataJugadoresArg["User Name"]))]

import numpy as np
coefficients1 = np.sum(x_counts.toarray()[datafiltered['User Name'].isin(dataFit["User Name"])], axis=0)
coefficients2 = np.sum(x_counts.toarray()[datafiltered['User Name'].isin(dataJugadoresArg["User Name"])], axis=0)

# +

coefficients1 = np.interp(coefficients1, (coefficients1.min(), coefficients1.max()), (0, +1))
coefficients2 = np.interp(coefficients2, (coefficients2.min(), coefficients2.max()), (0, +1))

# -

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
fig, ax = plt.subplots(dpi=(150))
ax.scatter(coefficients1, coefficients2, s=5)
for i, txt in enumerate(count_vect.get_feature_names()):
    if(coefficients1[i] > 0.3 or coefficients2[i] > 0.3 ):
        ax.annotate(txt, (coefficients1[i], coefficients2[i]), fontsize=6)
line = mlines.Line2D([0, 1], [0, 1], color='red')
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.ylabel("Futbol")
plt.xlabel("Cuentas Fit")
plt.title('Importancia de cada token')
plt.show()

# ## TF - IDF

# Ahora agruparemos por cuenta todos los posts, concatenando todos los textos de los posts por cada usuario.

# +
usuarios = pd.DataFrame()
usuarios["User_Name"] = data["User Name"].unique()
usuarios["numberOfPosts"] = [
    data["User Name"].value_counts()[user] for user in usuarios["User_Name"]
]
usuarios["Description"] = [
    data[data["User Name"] == user].Description.str.cat(sep=". ")
    for user in usuarios["User_Name"]
]
usuarios["ImageText"] = [
    data[data["User Name"] == user]["Image Text"].str.cat(sep=". ")
    for user in usuarios["User_Name"]
]  # No anda tan bien
usuarios["Titles"] = [
    data[data["User Name"] == user].Title.str.cat(sep=". ")
    for user in usuarios["User_Name"]
]
usuarios["AllText"] = (
    usuarios["Description"] + usuarios["Titles"] + usuarios["ImageText"]
)
usuarios["Likes"] = [
    np.sum(data[data["User Name"] == user].Likes) for user in usuarios["User_Name"]
]
usuarios["mean_Likes"] = [
    np.mean(data[data["User Name"] == user].Likes) for user in usuarios["User_Name"]
]
usuarios["Comments"] = [
    np.sum(data[data["User Name"] == user].Comments) for user in usuarios["User_Name"]
]
usuarios["mean_Comments"] = [
    np.mean(data[data["User Name"] == user].Comments) for user in usuarios["User_Name"]
]

usuarios.head()
# -

usuarios.tail()

# Veamos cuantos usuarios obtengo

usuarios.shape

# Veamos cual es el mínimo y máximo de posts por usuario

print(usuarios.numberOfPosts.min(), usuarios.numberOfPosts.max())


# Seteo 10 posts como mínimo para tener un mínimo de información por cuenta

numberOfPosts_minimo = 10

print(len(usuarios.numberOfPosts), sum(usuarios.numberOfPosts > numberOfPosts_minimo))

usuarios_mini = usuarios[usuarios.numberOfPosts > numberOfPosts_minimo]

import nltk

nltk.download('stopwords')

# +
# Carga de datos
texts = list(usuarios_mini.AllText)

# Cuento los terminos
count_vect = CountVectorizer(
    ngram_range=(1, 3),
    max_df=0.8,
    min_df=0.01,
    stop_words=stopwords.words('spanish'),
    lowercase=True,
)
x_counts = count_vect.fit_transform(texts)

# Genero matriz con valorizacion tf-idf
tfidf_transformer = TfidfTransformer(norm='l2')
x_tfidf = tfidf_transformer.fit_transform(x_counts)
# -

np.shape(x_tfidf)

# ## Armo la red
#
#
# Ahora creo un grafo, donde cada nodo es una cuenta y una arista los une si la similitud entre sus vectores de TF-IDF es mayor a la media + 1 desvío estándar.
# La idea es modelar mediante un grafo las conexiones "semánticas" de cada cuenta.

import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Creo el grafo, poniendo un nodo por cuenta (sin aristas por el momento) y agregando el atributo "User_Name" para después saber que nodo pertenece a que cuenta.

# Inicializo el grafo
G = nx.Graph()
G.add_nodes_from(
    [
        (i, {"User_Name": usuarios_mini.User_Name.iloc[i]})
        for i in range(usuarios_mini.shape[0])
    ]
)
len(G.nodes)

# Defino la función para calcular de forma matricial todos los pares de similitudes

from numpy import dot
from numpy.linalg import norm


def calcular_similitudes(x_tfidf):
    x1 = x_tfidf.toarray()
    normx1 = np.apply_along_axis(norm, 1, x1)
    normx2 = np.apply_along_axis(norm, 0, x1.T)
    x2 = x1.T
    similitudes = dot(x1, x2) / dot(
        normx1.reshape(x1.shape[0], 1), normx2.reshape(1, x1.shape[0])
    )
    np.fill_diagonal(similitudes, 0)
    return np.round(similitudes, decimals=6)


similitudes = calcular_similitudes(x_tfidf)
similitudes

# Defino el umbral (media + 1 desvío estándar) para establecer una arista entre los nodos y agrego todas las aristas correspondientes según este criterio.

# +
# Agrego aristas
treshold = similitudes.mean() + 1 * similitudes.std()
print("treshold: {:.5f}".format(treshold))

edges = list(G.edges)
G.remove_edges_from(edges)  # borro viejas, por si habia algo


for i in range(usuarios_mini.shape[0]):
    for j in range(usuarios_mini.shape[0]):
        if similitudes[i, j] > treshold:
            G.add_edge(i, j, weight=similitudes[i, j])
len(G.edges)
# -

# Grafico la matriz de similitudes mediante un heatmap binario.

# plot matriz similitud
plt.imshow(similitudes > treshold)
plt.show()

# ## Detecta comunidades
#
# Ahora pasaré a detectar comunidades en el grafo creado. Para esto utilizó el método "Louvain", el cual es una técnica greedy de detección de clusters en grafos (no lo vimos en la materia)

from networkx.algorithms import community
import community as com

partition = com.best_partition(G)

# Me fijo cuales son las 10 principales comunidades y cuantos usuarios tiene cada una

comunidad = []
for i in range(usuarios_mini.shape[0]):
    comunidad.append(partition[i])
usuarios_mini["comunidad"] = comunidad
usuarios_mini["comunidad"].value_counts().head(10)

# Veamos ahora que usuarios componen cada uno de estas comunidades (las que tienen mas de 2 usuarios)

usuarios_mini[usuarios_mini.comunidad == 23].sort_values(
    "mean_Likes", ascending=False
).User_Name.head(10)

usuarios_mini[usuarios_mini.comunidad == 2].sort_values(
    "mean_Likes", ascending=False
).User_Name.head(10)

usuarios_mini[usuarios_mini.comunidad == 16].sort_values(
    "mean_Likes", ascending=False
).User_Name.head(10)

usuarios_mini[usuarios_mini.comunidad == 30].sort_values(
    "mean_Likes", ascending=False
).User_Name.head(10)

# # Ploteo la red a ver como agrupa

colors = ["b", "r", "y", "g"] + ["w"] * 1000
dic_colores = {}
for i, key in enumerate(usuarios_mini["comunidad"].value_counts().index):
    dic_colores[key] = colors[i]

color_map = []
for user in list(nx.get_node_attributes(G, 'User_Name').values()):
    com = int(usuarios_mini[usuarios_mini.User_Name == user].comunidad)
    color_map.append(dic_colores[com])

plt.figure(figsize=[8, 8])
# pos = nx.draw_kamada_kawai(G) #draw_kamada_kawai draw_spectral draw_circular draw_spring
pos = nx.spring_layout(G, k=0.3)
nx.draw(G, node_color=color_map, with_labels=True, pos=pos)
plt.show()

G.nodes[45]['User_Name']


