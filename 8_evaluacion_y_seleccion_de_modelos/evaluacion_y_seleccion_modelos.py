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

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


# !echo{USER}

# +
# # !pip install kaggle
# # !mkdir /Users/jcollinet/.kaggle
# # !echo '{"username":"jorgecollinet", "key":"4ea5796b3910631404b1baf65c230677"}' > '/Users/jcollinet/.kaggle/kaggle.json'
# # !kaggle competitions download -c ieee-fraud-detection
# # !unzip 'ieee-fraud-detection.zip' -d ./dataset
# # !ls -l dataset
# -

# # 1) Cargar Datos
#
# Se procede a la carda del dataset y al analisis del mismo

# +
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df_train_transactions = pd.read_csv('dataset/train_transaction.csv')
df_train_identity = pd.read_csv('dataset/train_identity.csv')

df_test_transaction = pd.read_csv('dataset/test_transaction.csv')
df_test_identity = pd.read_csv('dataset/test_identity.csv')

df_sample_submition = pd.read_csv('dataset/sample_submission.csv')

# +
# otra opcion es generar sinteticamente los datos

# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=10000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
# -

# # 2) Transformar
#
# Basandonos en la clase de feature engineering vamos a aplicar un par de transformaciones para que el modelo pueda entrenar

df_train_transactions.head()

# +
# Categorical Features - Transaction
# ProductCD
# card1 - card6
# addr1, addr2
# P_emaildomain
# R_emaildomain
# M1 - M9

df_train_transactions = df_train_transactions.fillna({
    'addr1':0, 
    'addr2':0, 
    'P_emaildomain': 'gmail.com', 
    'R_emaildomain': 'gmail.com', 
    'card1':0, 
    'card2':0, 
    'card3':0, 
    'card4':'visa', 
    'card5':0, 
    'card6':'debit',
    'M1':'T', 
    'M2':'T', 
    'M3':'T', 
    'M4':'M0', 
    'M5':'F', 
    'M6':'F', 
    'M7':'F', 
    'M8':'F', 
    'M9':'T',
    'ProductCD':'W'
})

categorical_cols = ['addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
            'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9','ProductCD']

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
cc = enc.fit_transform(df_train_transactions[categorical_cols])
    
final_df = pd.concat([df_train_transactions[df_train_transactions.columns.difference(categorical_cols)],pd.DataFrame(cc)])
final_df = final_df.fillna(0)
# -
# # 3) Entrenar Modelo
#
# Entrenamos el modelo con los parametros default.   
# Vamos a usar un Random Forest como ejemplo.

# +
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)
X = final_df[final_df.columns.difference(['isFraud'])]
y = final_df['isFraud']
clf.fit(X, y)
# -

# # 4) Metricas
#
# Como sabemos que el modelo esta performando bien?   
# Que metricas podemos usar para evaluar un modelo?
#
# metricas:
# - Accuracy
# - Precision
# - Recall
# - F1
# - Macro acurracy vs micro acurracy
# - ROC
# - Confusion matrix
# - Evaluacion de calibracion

# ## Acurracy
#
# Accuracy = Correct / Total

from sklearn.metrics import accuracy_score
accuracy_score(clf.predict(X), y)

# medio que no sirve (cambiar dataset ??)
y.value_counts(True)

# ## Recall
#
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
#
# Recall = TP / (TP + FN)  
# TP: True Positives , Cuando el modelo dice que es positivo y efectivamente es positivo  
# FN: False Negatives, Cuando el modelo dice que es negativo y le pifea porque en realidad era positivo  
#
# En criollo: cuantos consigue "agarrar"

from sklearn.metrics import recall_score
recall_score(clf.predict(X), y,  pos_label=0)

# ## Precision
#
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
#
# Precision = TP / (TP + FP)   
# FP: False Positives, Cuando el modelo dice que es positivo, pero en realidad es negativo 
#
# En criollo: de los que aggarró, cuan puros son?

from sklearn.metrics import precision_score
precision_score(clf.predict(X), y)

# ## Score F1
#
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
#
# F1 = (2 x Recall X Precision) / (Recall + Precision)   
# En criollo: un numero que me junta otras dos metricas, me es mas facil leer este (LOL)

from sklearn.metrics import f1_score
f1_score(clf.predict(X), y)

# ## Macro acurracy vs micro acurracy

# +
# TODO
# -

# ## Herramienta todo-en-uno de sklearn
#
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
#
# Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.
#

from sklearn.metrics import classification_report
print(classification_report(clf.predict(X), y))



# ## ROC y AUC
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html  
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

# +
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def plot_roc(_fpr, _tpr, x):
    
    roc_auc = auc(_fpr, _tpr)
    
    plt.figure(figsize=(15,10))
    plt.plot(_fpr, _tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.scatter(_fpr, x)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
fpr, tpr, thresholds  = roc_curve(clf.predict(X), y)
plot_roc(fpr, tpr, thresholds)
# -

# ## Confusion Matrix

# +
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_confusion_matrix(y_pred, y_true, label_mapping=None):
    ids = list(set(y_true))

    if label_mapping is None:
        names = ids
    else:
        names = [label_mapping[_id] for _id in ids]

    cm = confusion_matrix(y_true, y_pred, ids)
    df_cm = pd.DataFrame(cm, names, names)

    plt.figure(figsize=(15, 7))
    sns.set(font_scale=2)
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')
    plt.yticks(rotation=20)
    plt.xticks(rotation=20)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    
plot_confusion_matrix(clf.predict(X), y, {0: 'no fraude', 1:'te estafaron amigo'})
# -

from sklearn.metrics import plot_confusion_matrix
fig, ax = plt.subplots(figsize=(15, 7))
plt.grid(False)
plot_confusion_matrix(clf, X, y, cmap=plt.cm.Blues, display_labels=['TODO PIOLA', 'ESTAFA'], ax=ax)

# ## Evaluacion de calibracion
#
# muy interesante como explica las pecularidades de porque random forest nunca va a dar scores cercanos a 0 o a 1 
# https://scikit-learn.org/stable/modules/calibration.html

# +

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
#     X, y = datasets.make_classification(n_samples=100000, n_features=20,
#                                     n_informative=2, n_redundant=10,
#                                     random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1.)

    fig = plt.figure(fig_index, figsize=(17, 12))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
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

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


plot_calibration_curve(RandomForestClassifier(max_depth=2, random_state=0), X, y, "Random Forest", 1)

plt.show()
print('The x axis represents the average predicted probability in each bin. The y axis is the fraction of positives, i.e. the proportion of samples whose class is the positive class (in each bin).')
# -

# # Momento!!
# En la teorica se dijo algo como que usar los mismos datos para sacar las metricas que los que usamos para entrenar 
# es como subir una foto y darte like a vos mismo.  
#
#
# no sirve?  
#
#
# Bueno, no todo es blanco y negro, aplicar metricas al set de entrenamiento sirve para saber si el modelo puede aprender algo
# o si ni siquiera puede adaptarse a los datos de entrenamiento.
#
# Para evaluar correctamente al modelo necesitamos dividir el set en 2 o 3 (como vimos en la teorica) y evaluar en los datos que NO fueron usados para entrenar.  
# De esa manera podemos entender como funciona el modelo ante datos no vistos cuando entrenaba.
#
#

# # 5) K Fold
#
# En base a lo que vimos en la teorica:
# - Vamos a dividir al dataset en k partes.
# - Entrenamos con k-1 y aplicamos las metricas anteriormente usadas en la parte restante.
#
# NOTA: hablamos de stratify para mantener la distribucion entre los cortes?

# # 6) Grid Search y Random Search
#
# Vamos a buscar encontrar que combinacion de los parametros que recibe el Random Forest (Hyperparametros) son los que mejor metricas consiguen.


