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
import plotly.express as px
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

# leemos el dataset a explorar
dataset = pd.read_csv('../datasets/superheroes.csv')

# Usando pandas profiling
report = ProfileReport(dataset, title='superhéroes', minimal=True)
report.to_notebook_iframe()

dataset.head()

dataset.tail()

# > Se observan nulos codificados con guión, los convertimos

dataset = dataset.replace('-', np.nan)

# > Claramente en altura y peso una forma de imputar nulos es asignar el -99. Es un claro ejemplo de ingenieria de variables, debemos convertir dichos valores a nulos para luego considerarlos a la hora de trabajar con los valores faltantes.

dataset = dataset.replace(-99, np.nan)

# > Alertamos una fila duplicada, la eliminamos

dataset[dataset.duplicated()]

dataset = dataset.drop_duplicates().reset_index(drop=True)
dataset.shape

# # Conversion

# En esta sección mostraremos las principales estrategias para convertir variables según su tipo. <br>Cabe aclarar que veremos un set de posibilidades sin evaluar el algoritmo a utilizar, que es parte fundamental de la decisión final que se tome sobre el manejo de cada variable.

# ## Categóricas de baja cardinalidad

# Aparece el concepto de *Encoder*, asociado a transformaciones sobre variables categóricas. Tenemos muchos tipos de encoders veamos los principales.

# ### Label Encoder

le = LabelEncoder()
# Convertimos nulos a string 'nan', es decir un valor posible mas
int_values = le.fit_transform(dataset['Eye color'].astype(str))

# Mostramos primeros 15 valores
pd.concat(
    [
        pd.Series(int_values[:15], name='encoded'),
        # inversión de la transformación
        pd.Series(le.inverse_transform(int_values)[:15], name='reverted'),
    ],
    axis=1,
)

# >Observamos que los nulos se codifican con un valor entero propio

# >Recordar que esta transformación es conveniente en categóricas ordinales (no es el caso del ejemplo) ya que asigna un orden a los elementos

# ### One Hot Encoding

pd.options.display.max_columns = None

ohe = OneHotEncoder()
cols = ohe.fit_transform(
    dataset['Eye color'].astype(str).values.reshape(-1, 1)
).todense()
cols = pd.DataFrame(cols, columns=ohe.categories_)
print("Valores únicos: ", dataset['Eye color'].nunique() + 1)  # +1 por null value
display(cols.head(2))
print(cols.shape)

# Otra solución para OneHotEncoding implementada en pandas:

eye_color_dummies = pd.get_dummies(dataset['Eye color'].astype(str))
display(eye_color_dummies.head(2))
print(eye_color_dummies.shape)

# Para evitar problemas de colinealidad se debe excluir una categoría del set (la ausencia de todas - vector de 0s - indica la presencia de la categoría faltante) <br>
# La función de pandas ya viene con una parámetro para esto:

eye_color_dummies = pd.get_dummies(dataset['Eye color'].astype(str), drop_first=True)
display(eye_color_dummies.head(2))
print(eye_color_dummies.shape)

# La necesidad de eliminar una columna se más claramente para una categórica de dos valores, veamos el caso de *Gender*

gender_dummies = pd.get_dummies(dataset['Gender'])
display(gender_dummies.tail(5))

# >Con una sola columna tenemos toda la información necesaria

# ## Categóricas de alta cardinalidad

# Que pasa con la variable *Race* que tiene mas de 60 valores, vamos a crear 60 variables? <br>
# Veamos la distribución de los mismos

# +
unique_races = dataset['Race'].value_counts(dropna=False)
print("count : ", unique_races.shape[0])
display(unique_races.head(10))

s = sum(unique_races.values)
h = unique_races.values / s
c_sum = np.cumsum(h)
plt.plot(c_sum, label="Distribución de la suma acumulativa de razas")
plt.grid()
plt.legend()
# -

# >Con el top 10 cubrimos mas del 85% de la data

# +
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
fh = FeatureHasher(n_features=10, input_type='string')
hashed_features = fh.fit_transform(
    dataset['Race'].astype(str).values.reshape(-1, 1)
).todense()

pd.DataFrame(hashed_features).add_prefix('Race_').head(10).join(
    dataset['Race'].head(10)
)
# -

# ## Numéricas

# En el set tenemos dos variables numéricas, *Weight* y *Height* veamos su distribución

# +
df = dataset[dataset.Alignment != 'neutral'].reset_index(drop=True)


def plot_weight_vs_height(df, title=""):
    fig = px.scatter(
        df.dropna(),
        x="Weight",
        y="Height",
        color="Alignment",
        marginal_x="box",
        marginal_y="box",
        hover_name='name',
        title="Peso vs altura " + title,
    )
    fig.update_layout(autosize=False, width=1000)
    fig.show()
    display(df[['Weight', 'Height']].describe())


plot_weight_vs_height(
    df[['name', 'Weight', 'Height', 'Alignment']], "- Valores originales"
)


# -

# >Se observa una dispersión mucho mas grande de valores en el peso que en la altura.

# ### Scalers

# Aparece el concepto de *Scaler*, una transformación por la cual escalamos a un determinado rango/distribución, veamos distintas implementaciones:

# +
def get_fitted_scaler(cols, scaler_instance):
    '''Devuelve el scaler entrenado para las columnas informadas'''
    # fit del scaler
    values = cols.values
    scaler_instance.fit(values)
    return scaler_instance


def transform(df, cols_to_transform, scaler):
    scaler = get_fitted_scaler(df[cols_to_transform].dropna(), scaler)
    values = scaler.transform(df[cols_to_transform].dropna())
    return df[['name', 'Alignment']].join(
        pd.DataFrame(values, columns=cols_to_transform)
    )


cols_to_transform = ['Weight', 'Height']

scalers = [
    StandardScaler(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    MinMaxScaler(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    RobustScaler(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    PowerTransformer(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    Normalizer(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
]
for i in scalers:
    plot_weight_vs_height(transform(df, cols_to_transform, i), i.__class__.__name__)

# -


# ### Discretización

# Tranformación por la cual convertimos una variable continua en categórica

df = dataset['Weight'].dropna().reset_index(drop=True)
X = df.values.reshape(-1, 1)
enc = KBinsDiscretizer(n_bins=4, encode='ordinal')
X_binned = enc.fit_transform(X)
result = pd.concat([df, pd.DataFrame(X_binned, columns=['Weight_bins'])], axis=1)
display(result.head(5))
print("Límites bins:", enc.bin_edges_)

# mismo ejemplo con pandas
result, bins = pd.qcut(df, 4, labels=[0, 1, 2, 3], retbins=True)
result = pd.concat([df, pd.Series(result, name='Weight_bins')], axis=1)
display(result.head(5))
print("Límites bins:", bins)

# # Missings

# Veamos que variables contienen nulos

dataset.isnull().sum()

# en porcentajes
round(dataset.isnull().sum() / dataset.shape[0] * 100, 2)

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

# ## Categóricas

# Como vimos, los encoders solucionan el problema de nulos ya que imputan con la misma lógica que para los demás valores de la variable

# ## Numéricas

# Completar con la mediana, promedio, moda o constante

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


def complete_constant(serie, k):
    '''Retorna la serie imputada por una constante'''
    return serie.fillna(k)


# +
def compare_strategies(df, name_col, k=-99):
    '''Devuelve el valor de imputacion de las tres estrategias
    para esa columna'''

    # se llaman a las estrategias renombrando la serie
    median_fill = complete_median(df[name_col]).to_frame('median')
    mean_fill = complete_mean(df[name_col]).to_frame('mean')
    mode_fill = complete_mode(df[name_col]).to_frame('mode')
    constant_fill = complete_constant(df[name_col], k).to_frame('constant')

    # vemos los valores con los que completa en cada caso
    return pd.concat(
        [df[name_col], median_fill, mean_fill, mode_fill, constant_fill], axis=1
    )[dataset[name_col].isna()].head(1)


display(compare_strategies(dataset, 'Weight'))
display(compare_strategies(dataset, 'Height'))
# -

# Si implementamos el mismo ejemplo con sklearn, aparece el concepto de *imputer*, el mismo se entrena en un set de datos y puede ser aplicado en otro set de datos luego.


def get_imputer(col, strategy, k=-99):
    '''Devuelve el imputer de dicha columna para la estrategia indicada.
    Valores de estrategias: "median", "mean", "most_frequent", "constant"'''
    imputer = SimpleImputer(strategy=strategy, fill_value=k)
    # fit del imputer
    values = col.values
    imputer.fit(values)
    return imputer


def compare_imputers(df, name_col, k=-99):
    '''Devuelve el valor de imputacion de las estrategias
    para esa columna'''
    median_imputer = get_imputer(dataset[[name_col]], 'median')
    mean_imputer = get_imputer(dataset[[name_col]], 'mean')
    mode_imputer = get_imputer(dataset[[name_col]], 'most_frequent')
    constant_imputer = get_imputer(dataset[[name_col]], 'constant', k)

    # transformo
    values = df[[name_col]].values
    median_values = median_imputer.transform(values).ravel()
    mean_values = mean_imputer.transform(values).ravel()
    mode_values = mode_imputer.transform(values).ravel()
    constant_values = constant_imputer.transform(values).ravel()
    return pd.concat(
        [
            df[name_col],
            pd.Series(median_values, name='median'),
            pd.Series(mean_values, name='mean'),
            pd.Series(mode_values, name='mode'),
            pd.Series(constant_values, name='constant'),
        ],
        axis=1,
    )[df[name_col].isna()].head(1)


display(compare_imputers(dataset, 'Weight'))
display(compare_imputers(dataset, 'Height'))
