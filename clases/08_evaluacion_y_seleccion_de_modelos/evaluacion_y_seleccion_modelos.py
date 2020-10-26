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
#     display_name: Python 3 (venv)
#     language: python
#     name: python3
# ---

# +
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import os
from sklearn.datasets import make_blobs

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()
# -

pd.options.display.max_columns = None


# # Hyperparameter tuning
#

# X, y = make_blobs(n_samples=1000, centers=2, n_features=128, cluster_std=16, random_state=117, center_box=(-2,2), )
X, y = datasets.make_classification(
    n_samples=1000,
    n_features=64,
    n_repeated=16,
    n_informative=8,
    n_redundant=16,
    weights=[0.8, 0.2],
    flip_y=0.08,
    random_state=117,
)

dimred = PCA(2).fit_transform(X)
sns.scatterplot(dimred[:, 0], dimred[:, 1], hue=y)

# + jupyter={"source_hidden": true}
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=117)

# +
max_depths = np.arange(1, 15)
min_samples_leafs = np.arange(1, 51)
data_points = []
for max_depth in max_depths:
    for min_samples_leaf in min_samples_leafs:
        clf = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=117
        )
        clf.fit(X_train, y_train)
        data_points.append(
            (
                max_depth,
                min_samples_leaf,
                f1_score(y_test, clf.predict(X_test), average="weighted"),
            )
        )

data_points = pd.DataFrame(
    data_points, columns=["max_depth", "min_samples_leaf", "score"]
)
# -

plt.figure(dpi=125, figsize=(12, 8))
g = sns.heatmap(
    data_points.pivot_table(
        index="max_depth", columns="min_samples_leaf", values="score"
    ),
    square=True,
    cbar_kws=dict(use_gridspec=False, location="bottom"),
)

# # Baseline model
#
# Entrenamos un arbol de decision con los parametros default.

clf = DecisionTreeClassifier(random_state=117)
clf.fit(X_train, y_train)

# # Cómo evaluar un modelo - Metricas
#
# Pero ahora que tenemos el modelo entrenado,
# - Como sabemos que el modelo esta performando bien?
# - Como sabemos si otro modelo esta performando mejor?
# - Que metricas podemos usar para evaluar un modelo?

# ## Metricas para clasificacion

# ### Acuracy
#
# $$\text{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)$$
#
# > [sklearn: accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

from sklearn.metrics import accuracy_score

accuracy_score(clf.predict(X_test), y_test)

# ## Precision
# $$\text{precision} = \frac{tp}{tp + fp}$$
#
# Donde $tp$ es la cantidad de verdaderos positivos: son positivos y la prediccion es positivo.
# $fp$ son los falsos positivos: son negativos y la prediccion es positivo.
#
# > [sklearn: accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

from sklearn.metrics import precision_score

precision_score(clf.predict(X), y)

# ## Recall
# $$\text{recall} = \frac{tp}{tp + fn}$$
#
# Donde $fn$ es la cantidad de falsos negativos: son positivos y predecimos negativos.
#
# > [sklearn: recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

from sklearn.metrics import recall_score

recall_score(clf.predict(X), y, pos_label=0)

# ## Score F1
# $$F_\beta = 2 \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}$$
#
# Donde $fn$ es la cantidad de falsos negativos: son positivos y predecimos negativos.
#
# > [sklearn: f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

from sklearn.metrics import f1_score

f1_score(clf.predict(X), y)

# ## Herramienta todo-en-uno de sklearn: reporte de clasificacion
#
# Muestra un reporte en texto con las principales metricas de clasificacion.
#
# > [sklearn: classificaction_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

from sklearn.metrics import classification_report

print(classification_report(clf.predict(X), y))


# + [markdown] jupyter={"source_hidden": true}
# ## ROC y AUC
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

# + jupyter={"source_hidden": true, "outputs_hidden": true}
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


def plot_roc(_fpr, _tpr, x):

    roc_auc = auc(_fpr, _tpr)

    plt.figure(figsize=(15, 10))
    plt.plot(
        _fpr, _tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc
    )
    plt.scatter(_fpr, x)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


fpr, tpr, thresholds = roc_curve(clf.predict(X), y)
plot_roc(fpr, tpr, thresholds)
# -

# ## Confusion Matrix
#
# La matriz de confusion nos muestra nuestros $tp$, $fp$, $tn$ y $fn$ en una matriz. La diagonal principal son los valores correctamente clasificados. Los otros valores indican la cantidad de puntos mal clasificados.
#
# Veamos como construirlo manualmente con [sklearn.metrics.confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) y `sns.heatmap`.

# +
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred):
    names = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, names)
    df_cm = pd.DataFrame(cm, names, names)

    plt.figure(dpi=100)
    plt.title("Matriz de confusion")
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g', square=True)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


plot_confusion_matrix(y, clf.predict(X))
# -

# Tambien tenemos el shortcut de [sklearn.metrics.plot_confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html), pero ojo al usarlo con seaborn seteando el estilo.

# +
from sklearn.metrics import plot_confusion_matrix

fig, ax = plt.subplots(figsize=(15, 7))
plt.grid(False)
plot_confusion_matrix(clf, X, y, cmap=plt.cm.Blues, display_labels=['1', '0'], ax=ax)
plt.show()
# -

# # Búsqueda de hiperparámetros
# Ahora que sabemos como medir que tan buenos son diferentes modelos, queremos probar y evaluar combinaciones de hiperparametros y de una grilla de posibles combinaciones. Hacer un `for` por cada hiperparametro como haciamos al principio del notebook es bastante engorroso. Por otro lado, el entrenamiento y evaluacion de distintos valores de la grilla son independientes entre si, lo cual nos hace pensar que podriamos parelelizarlo facilmente, pero tenemos que escribir ese codigo.

# ## Grid Search
# Grid search recorre exhaustivamente una grilla de combinaciones de hiperparametros.
#
# > [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

# ## Randomized Search
# Grid search toma aleatoriamente una cierta cantidad de combinaciones de hiperparametros de una grilla de combinaciones de hiperparametros.
#
# > [sklearn.model_selection.RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

# # Cross validation
# En la teorica se dijo algo como que usar los mismos datos para sacar las metricas que los que usamos para entrenar
# es como subir una foto y darte like a vos mismo.
#
# No sirve?
#
# Bueno, no todo es blanco y negro, podemos aplicar metricas al set de entrenamiento sirve para saber si el modelo puede aprender algo o si ni siquiera puede adaptarse a los datos de entrenamiento.
#
# Para evaluar correctamente al modelo necesitamos dividir el set en varios `folds` (como vimos en la teorica) y evaluar en los datos que NO fueron usados para entrenar.
# De esa manera podemos entender como funciona el modelo ante datos no vistos cuando entrenaba. Otros modos de partir los datos son `leave one out`, por ejemplo.
#
#

# # K Fold
#
# En base a lo que vimos en la teorica:
# - Vamos a dividir al dataset en k partes.
# - Entrenamos con k-1 y aplicamos las metricas anteriormente usadas en la parte restante.
#
# NOTA: hablamos de stratify para mantener la distribucion entre los cortes?

# +
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(X):
    pass
# + [markdown] jupyter={"source_hidden": true}
# # Evaluacion de calibracion
#
# muy interesante como explica las pecularidades de porque random forest nunca va a dar scores cercanos a 0 o a 1
# https://scikit-learn.org/stable/modules/calibration.html

# + jupyter={"source_hidden": true}

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split


# Create dataset of classification task with many redundant and few
# informative features


def plot_calibration_curve(est, X, y, name, fig_index=0):
    X, y = datasets.make_classification(
        n_samples=100000,
        n_features=20,
        n_informative=2,
        n_redundant=10,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1.0)

    fig = plt.figure(fig_index, figsize=(17, 12))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [
        (lr, 'Logistic'),
        (est, name),
        (isotonic, name + ' + Isotonic'),
        (sigmoid, name + ' + Sigmoid'),
    ]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob_pos, n_bins=10
        )

        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label="%s (%1.3f)" % (name, clf_score),
        )

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


plot_calibration_curve(
    RandomForestClassifier(max_depth=2, random_state=0), X, y, "Random Forest", 1
)

plt.show()
print(
    'The x axis represents the average predicted probability in each bin. The y axis is the fraction of positives, i.e. the proportion of samples whose class is the positive class (in each bin).'
)
