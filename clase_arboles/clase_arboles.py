# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3 (venv)
#     language: python
#     name: python3
# ---

# # Arboles de decision
#
# En esta clase entrenaremos y exploraremos arboles de decision.

from functools import lru_cache

import dtreeviz.trees as dtreeviz
import graphviz
import ipywidgets as widgets
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import SVG, display
from ipywidgets import Button, IntSlider, interactive
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport
from sklearn import preprocessing, tree
from sklearn.preprocessing import OneHotEncoder

sns.set()


# # Carga de datos
#
# Vamos a usar un dataset de [valor de seguros](https://www.kaggle.com/mirichoi0218/insurance), cargandolo desde [github](https://github.com/stedy/Machine-Learning-with-R-datasets). Nuestra variable objetivo será la columna `smoker` que indica si es fumador o no.

# +
@lru_cache()
def get_data():
    """Obtener el dataset.

    Devuelve
    --------
        pd.DataFrame: El dataset descargado desde github.
    """
    return pd.read_csv(
        'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
    )


dataset = get_data()
# -

dataset.info()

dataset.head()

# # Breve análisis exploratorio


report = ProfileReport(
    dataset, title='Dataset de seguros', explorative=True, lazy=False
)

report.to_widgets()
# -

# Podemos ver que la variable objetivo está bastante desbalanceada

# +
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[6.4 * 2, 4.8], dpi=100)

dataset.smoker.value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title("Smoker")
axes[0].set_ylabel("Cantidad")

dataset.smoker.value_counts().div(dataset.pipe(len)).mul(100).plot(
    kind='bar', ax=axes[1]
)
axes[1].set_title("Smoker")
axes[1].set_ylabel("Porcentaje")

plt.show()


# -

# El estimador más sencillo que podemos armar mirando esto es responder siempre `no`. Y aproximadamente el 80% de las veces tendremos razón!

# # Manejo de variables
#
# Si revisamos el panel de pandas profiling, podemos ver que las columnas `sex` y `region` son categóricas. Vamos a aplicar [one hot encoding](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) con pandas para estas variables. Por otro lado, usaremos [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) de sklearn para crear un mapping de los valores `yes`/`no` de la columna `smoker` a valores numericos.

# +
def feature_engineering(df):
    """Hace las transformaciones de datos necesarias."""
    df = pd.get_dummies(df, drop_first=True, columns=['sex', 'region'])

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.smoker)

    X = df.drop(columns=['smoker'])
    y = label_encoder.transform(df.smoker)

    return X, y, df, label_encoder


X, y, df, y_encoder = feature_engineering(dataset)
df.head()


# -

# # Entrenamiento
# Sklearn [propone](https://scikit-learn.org/stable/developers/develop.html) una interfaz común a todos sus estimadores, pero sin enforzarla. Se espera que tengan un método `.fit` y un método `.predict`. Toda la inicialización de hiperparametros debe estar en el `__init__` de la clase, y deben tener valores por defecto.
#
# Vamos a entrenar un modelo y jugar con la profundidad máxima del árbol. El árbol entrenado, lo visualizaremos utilizando código similar al de la documentación de sklearn encontrado [aquí](https://scikit-learn.org/stable/modules/tree.html#classification).

# +
def get_tree(X, y, max_depth=5, min_samples_leaf=10):
    """Devuelve un árbol entrenado."""
    clf = tree.DecisionTreeClassifier(
        random_state=117, max_depth=max_depth, min_samples_leaf=min_samples_leaf
    )
    clf.fit(X, y)
    return clf


def plot_tree(max_depth, min_samples_leaf):
    """Interfaz interactiva para visualizar un árbol entrenado."""
    df = get_data()
    X, y, df, y_encoder = feature_engineering(df)
    clf = get_tree(X, y, max_depth, min_samples_leaf)

    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=X.columns,
        class_names=list(y_encoder.classes_),
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    display(SVG(graph.pipe(format='svg')))


inter = interactive(
    plot_tree,
    max_depth=IntSlider(min=1, max=15),
    min_samples_leaf=IntSlider(min=1, max=25, value=10),
)
display(inter)
# -

# # Explorando las particiones
#
# [dtreeviz](https://github.com/parrt/dtreeviz) permite explorar un poco mas los árboles. Es un poco más visual que su contraparte de sklearn. Veamos las reglas aprendidas por cada nodo.

# +
X, y, df, y_encoder = feature_engineering(dataset)
clf = get_tree(X, y)

viz = dtreeviz.dtreeviz(
    clf,
    X,
    y,
    target_name='smoker',
    feature_names=list(X.columns),
    class_names=list(y_encoder.classes_),
    scale=1.5,
)

display(viz)


# -

# # Por dónde cae una predicción?
#
# Si tomamos una muestra al azar de nuestro dataset y le pedimos al árbol una predicción, como se hace?

# +
def explore_prediction():
    """Interfaz interactiva para ver como se hace una predicción al azar."""
    x_sample = df.sample()
    display(x_sample)

    viz = dtreeviz.dtreeviz(
        clf,
        X,
        y,
        target_name='smoker',
        feature_names=list(X.columns),
        class_names=list(y_encoder.classes_),
        scale=1.5,
        X=x_sample[X.columns].iloc[0].values,
    )

    display(viz)


inter = interactive(explore_prediction, {'manual': True})
display(inter)
