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
from sklearn.impute import SimpleImputer

# leemos el dataset a explorar
dataset = pd.read_csv('../datasets/superheroes.csv')

# info del dataset
dataset.info()

dataset.head()

# > Se observan nulos codificados con guión, los convertimos

dataset = dataset.replace('-', np.nan)

dataset.describe()

# > Claramente en altura y peso una forma de imputar nulos es asignar el -99. Es un claro ejemplo de ingenieria de variables, debemos convertir dichos valores a nulos para luego considerarlos a la hora de trabajar con los valores faltantes.

dataset = dataset.replace(-99, np.nan)

# # Conversion

# Vamos a ver que podemos mapear texto en enteros

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le = LabelEncoder()
le.fit_transform(dataset['Eye color'].astype(str))

# Tambien podemos Aplicarle One Hot Encoding a Texto

# +
from sklearn.preprocessing import LabelEncoder

ohe = OneHotEncoder()
ohe.fit_transform(dataset['Eye color'].astype(str).values.reshape(-1, 1))[:2].todense()
# -

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html#sklearn.preprocessing.KBinsDiscretizer   
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html



# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer   
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler   
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler   

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures   



# # Missings

# Veamos que variables contienen nulos

dataset.isnull().sum()

# Veamos algunos registros de dichas variables accediendo con [.loc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html)

# condicion sobre las columnas, cantidad de nulos > 0
dataset.loc[:, dataset.isnull().sum() > 0].head()

# Tenemos dos tipos de variables a tratar, numéricas y categóricas. <br>
# Veamos algunas soluciones generales

# eliminar filas con nulos
less_rows = dataset.dropna(axis=0)
less_rows.shape

# Nos quedarían 50 registros válidos en el set

# eliminar columnas con nulos
less_cols = dataset.dropna(axis=1)
less_cols.shape


# Nos quedaría una sola columna sin nulos

# Soluciones para variables numéricas, completar con la mediana, promedio o moda

# +
def complete_median(serie):
    '''Retorna la serie imputada por mediana'''
    return serie.fillna(serie.median())


def complete_mean(serie):
    '''Retorna la serie imputada por promedio'''
    return serie.fillna(serie.mean())


def complete_mode(serie):
    '''Retorna la serie imputada por moda'''
    return serie.fillna(serie.mode()[0])


# +
def compare_median_mean_mode(df, name_col):
    '''Devuelve el valor de imputacion de las tres estrategias
    para esa columna'''

    # se llaman las tres estrategias renombrando la serie
    median_fill = complete_median(df[name_col]).to_frame('median')
    mean_fill = complete_mean(df[name_col]).to_frame('mean')
    mode_fill = complete_mode(df[name_col]).to_frame('mode')

    # vemos los valores con los que completa en cada caso
    return pd.concat([df[name_col], median_fill, mean_fill, mode_fill], axis=1)[
        dataset[name_col].isna()
    ].head(1)


display(compare_median_mean_mode(dataset, 'Weight'))
display(compare_median_mean_mode(dataset, 'Height'))


# -

# Si implementamos el mismo ejemplo con sklearn, aparece el concepto de *imputer*, el mismo se entrena en un set de datos y puede ser aplicado en otro set de datos luego.

# +
# el mismo ejemplo con sklearn
def get_imputer(col, strategy):
    '''Devuelve el imputer de dicha columna para la estrategia indicada.
    Valores de estrategias: "median", "mean", "most_frequent"'''
    imputer = SimpleImputer(strategy=strategy)
    # fit del imputer
    values = col.values
    imputer.fit(values)
    return imputer


def compare_median_mean_mode_imputers(df, name_col):
    '''Devuelve el valor de imputacion de las tres estrategias
    para esa columna'''
    median_imputer = get_imputer(dataset[[name_col]], 'median')
    mean_imputer = get_imputer(dataset[[name_col]], 'mean')
    mode_imputer = get_imputer(dataset[[name_col]], 'most_frequent')

    # transformo
    values = df[[name_col]].values
    median_values = median_imputer.transform(values).ravel()
    mean_values = mean_imputer.transform(values).ravel()
    mode_values = mode_imputer.transform(values).ravel()

    return pd.concat(
        [
            df[name_col],
            pd.Series(median_values, name='median'),
            pd.Series(mean_values, name='mean'),
            pd.Series(mode_values, name='mode'),
        ],
        axis=1,
    )[df[name_col].isna()].head(1)


display(compare_median_mean_mode(dataset, 'Weight'))
display(compare_median_mean_mode(dataset, 'Height'))
# -


