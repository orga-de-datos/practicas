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
from IPython.display import SVG, display
from ipywidgets import Button, IntSlider, interactive
from sklearn import tree
from sklearn.datasets import load_iris

# # Carga de datos
# Vamos a usar el [iris dataset](http://archive.ics.uci.edu/ml/datasets/Iris/), cargandolo directamente con [un helper de sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html). Notar que devuelve un [bunch](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch), en lugar de un dataframe.

iris = load_iris(return_X_y=False)
X = iris.data
y = iris.target


# # Entrenamiento
# Sklearn [propone](https://scikit-learn.org/stable/developers/develop.html) una interfaz común a todos sus estimadores, pero sin enforzarla. Se espera que tengan un método `.fit` y un método `.predict`. Toda la inicialización de hiperparametros debe estar en el `__init__` de la clase, y deben tener valores por defecto.
#
# Vamos a entrenar un modelo y jugar con la profundidad máxima del árbol. El árbol entrenado, lo visualizaremos utilizando código similar al de la documentación de sklearn encontrado [aquí](https://scikit-learn.org/stable/modules/tree.html#classification).

# +
@lru_cache()
def get_iris_data():
    iris = load_iris(return_X_y=False)
    X = iris.data
    y = iris.target
    return X, y, iris


def plot_tree(max_depth):

    X, y, iris = get_iris_data()

    clf = tree.DecisionTreeClassifier(random_state=117, max_depth=max_depth)
    clf.fit(X, y)
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    display(SVG(graph.pipe(format='svg')))


inter = interactive(plot_tree, max_depth=IntSlider(min=1, max=15))
display(inter)
# -

# Notar que el árbol no crece a una profundida mayor a 5 aún cuando el hiperparámetro `max_depth` lo permite.

# # Explorando las particiones
#
# [dtreeviz](https://github.com/parrt/dtreeviz) permite explorar un poco mas los árboles. Es un poco más visual que su contraparte de sklearn.

X, y, iris = get_iris_data()

# +
clf = tree.DecisionTreeClassifier(random_state=117, max_depth=5)
clf.fit(X, y)

viz = dtreeviz.dtreeviz(
    clf,
    iris.data,
    iris.target,
    target_name='variety',
    feature_names=iris.feature_names,
    class_names=iris.target_names.tolist(),
    scale=2.0,
)

display(viz)
# -

# # Por dónde cae una predicción?
#
# Si tomamos una muestra al azar de nuestro dataset y le pedimos al árbol una predicción, como se hace?

iris_df = pd.DataFrame(
    data=np.c_[iris.data, pd.Series(iris.target_names.tolist())[iris.target]],
    columns=iris.feature_names + ['variety'],
)
iris_df.sample(5)


# +
def explore_prediction():
    X, y, iris = get_iris_data()
    x_sample = iris_df.sample()
    display(x_sample)

    viz = dtreeviz.dtreeviz(
        clf,
        iris.data,
        iris.target,
        target_name='variety',
        feature_names=iris.feature_names,
        class_names=iris.target_names.tolist(),
        scale=1.5,
        X=x_sample[iris.feature_names].iloc[0].values,
    )

    display(viz)


inter = interactive(explore_prediction, {'manual': True})
display(inter)
