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
import pandas as pd


# # 1) Cargar Datos
#
# Se procede a la carda del dataset y al analisis del mismo

# # 2) Transformar
#
# Basandonos en la clase de feature engineering vamos a aplicar un par de transformaciones para que el modelo pueda entrenar

# # 3) Entrenar Modelo
#
# Entrenamos el modelo con los parametros default.   
# Vamos a usar un Random Forest como ejemplo.

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
# - ROC
# - Confusion matrix
# - Macro acurracy vs micro acurracy
# - Evaluacion de calibracion

# # 5) K Fold
#
# En base a lo que vimos en la teorica:
# - Vamos a dividir al dataset en k partes.
# - Entrenamos con k-1 y aplicamos las metricas anteriormente usadas en la parte restante.

# # 6) Grid Search y Random Search
#
# Vamos a buscar encontrar que combinacion de los parametros que recibe el Random Forest (Hyperparametros) son los que mejor metricas consiguen.


