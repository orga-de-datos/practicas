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

# # Arboles de decision
#
# En esta clase entrenaremos y exploraremos arboles de decision.

from functools import lru_cache

import dtreeviz.trees as dtreeviz
import graphviz
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import SVG, display
from ipywidgets import Button, IntSlider, interactive
from sklearn import preprocessing, tree
from sklearn.preprocessing import OneHotEncoder

# # Carga de datos
# Vamos a usar un dataset de [valor de seguros](https://www.kaggle.com/mirichoi0218/insurance), cargandolo desde [github](https://github.com/stedy/Machine-Learning-with-R-datasets).

# +
@lru_cache()
def get_data():
    # leemos el dataset a explorar
    return pd.read_csv(
        'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
    )


dataset = get_data()
# -

dataset.info()

dataset.head()


# # Manejo de variables

# +
def fe(df):
    dummies = pd.get_dummies(df[['sex', 'region']], drop_first=True)
    df = df.drop(columns=['sex', 'region']).join(dummies)

    X = df.drop(columns=['smoker'])
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.smoker)
    y = label_encoder.transform(df.smoker)
    return X, y, df, label_encoder


X, y, df, y_encoder = fe(dataset)
df.head()


# -

# # Entrenamiento
# Sklearn [propone](https://scikit-learn.org/stable/developers/develop.html) una interfaz común a todos sus estimadores, pero sin enforzarla. Se espera que tengan un método `.fit` y un método `.predict`. Toda la inicialización de hiperparametros debe estar en el `__init__` de la clase, y deben tener valores por defecto.
#
# Vamos a entrenar un modelo y jugar con la profundidad máxima del árbol. El árbol entrenado, lo visualizaremos utilizando código similar al de la documentación de sklearn encontrado [aquí](https://scikit-learn.org/stable/modules/tree.html#classification).

# +
def get_tree(X, y, max_depth=5):
    clf = tree.DecisionTreeClassifier(
        random_state=117, max_depth=max_depth, min_samples_leaf=10
    )
    clf.fit(X, y)
    return clf


def plot_tree(max_depth):

    df = get_data()
    X, y, df, y_encoder = fe(df)
    clf = get_tree(X, y, max_depth)

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


inter = interactive(plot_tree, max_depth=IntSlider(min=1, max=15))
display(inter)
# -

# # Explorando las particiones
#
# [dtreeviz](https://github.com/parrt/dtreeviz) permite explorar un poco mas los árboles. Es un poco más visual que su contraparte de sklearn.

# +
clf = get_tree(X, y)

viz = dtreeviz.dtreeviz(
    clf,
    X,
    y,
    target_name='smoker',
    feature_names=list(X.columns),
    class_names=list(y_encoder.classes_),
    scale=2.0,
)

display(viz)
# -

# # Por dónde cae una predicción?
#
# Si tomamos una muestra al azar de nuestro dataset y le pedimos al árbol una predicción, como se hace?

df.sample(5)


# +
def explore_prediction():
    X, y, df, y_encoder = fe(get_data())
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
