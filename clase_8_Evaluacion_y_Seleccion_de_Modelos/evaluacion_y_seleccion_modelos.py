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
# Recall = TP / (TP + FN)  
# En criollo: cuantos consigue "aggarrar"

from sklearn.metrics import recall_score
recall_score(clf.predict(X), y)

# ## Precision
# Precision = TP / (TP + FP)   
# En criollo: de los que aggarró, cuan puros son?

from sklearn.metrics import precision_score
precision_score(clf.predict(X), y,  pos_label=0), precision_score(clf.predict(X), y, pos_label=1) 

# ## Score F1
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

from sklearn.metrics import classification_report
print(classification_report(clf.predict(X), y))

# ## ROC y AUC

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
fig, ax = plt.subplots(figsize=(10, 10))
plt.grid(False)
plot_confusion_matrix(clf, X, y, cmap=plt.cm.Blues, display_labels=['TODO PIOLA', 'ESTAFA'], ax=ax)

# ## Evaluacion de calibracion



# # 5) K Fold
#
# En base a lo que vimos en la teorica:
# - Vamos a dividir al dataset en k partes.
# - Entrenamos con k-1 y aplicamos las metricas anteriormente usadas en la parte restante.

# # 6) Grid Search y Random Search
#
# Vamos a buscar encontrar que combinacion de los parametros que recibe el Random Forest (Hyperparametros) son los que mejor metricas consiguen.


