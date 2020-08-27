# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Clase 1 - Introducción a pandas

# En este notebook vamos a ver una introducción a los comandos iniciales de la biblioteca pandas para Python. [Pandas](https://pandas.pydata.org) es una biblioteca sponsoreada por [NumFOCUS](https://numfocus.org/sponsored-projects).
#
# La descripción en su página la define como:
#
# > pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
# built on top of the Python programming language.

# # Estructuras de datos

# Dos estructuras fundamentales dentro de pandas son Series y DataFrames. Los dataframes son tablas de datos. Cada columna es una Series.
#
# Para comenzar, vamos a crear nuestro primer DataFrame a partir de un archivo CSV que contiene un [dataset sobre superhéroes](https://www.kaggle.com/claudiodavi/superhero-set/home).

# +
import pandas as pd

df = pd.read_csv('../datasets/superheroes.csv')
# -

# vemos los primeros elementos
df.head()

# descripción de cada columna e información general del dataframe
df.info()

# La última columna, `Dtype` nos dice qué tipo de dato interpreta pandas que es esa columna. Las columnas de tipo `str` son interpretadas con `dtype` del tipo `object`.

# resumen estadístico de las columnas numéricas
df.describe()

# si queremos ver la cantidad de nulos en cada columna
# creamos una máscara binaria y la sumamos
df.isnull().sum()

# # Cómo obtener valores que nos interesan

# ## Por índices

df.index

df = df.set_index('name')
df

df.index

# +
hero = df.loc['Aurora']

hero

# +
hero = df.iloc[55]

hero

# +
heroes = df.loc['Aurora':'Banshee']

heroes

# +
heroes = df.iloc[55:60]

heroes
# -

# ## Obtener columnas

df.Race

df['Race']

# ## Obtener un subconjunto de columnas

df[['Race', 'Skin color']]

# ## Obtener filas

condition = df['Skin color'] == 'blue'

condition.head()

# Si ahora queremos filtrar y quedarnos solamente con los datos que cumplen la condición:

df[df['Skin color'] == 'blue']


# # Transformaciones de datos

# ## Apply

# Mediante apply podemos aplicar una función definida aparte a nuestro set de datos


def rate_height(height):
    if height >= 200:
        return "Tall"
    else:
        return "Not tall"


altos = df['Height'].apply(rate_height)
altos

df['Tallness'] = df['Height'].apply(rate_height)

df

# ## Replace

# Vemos por ejemplo que 'Skin color' usa '-' para indicar valores vacíos. Corrijamos esto.

df['Skin color'].replace({"-": None}, inplace=True)

df['Skin color']

# ## Eliminar filas con nulos

df = df.dropna(subset=['Skin color'])

# # Unir información de distintas tablas
#
# Veamos si hay razas que tengan el mismo color de piel.

df.merge(df, left_on='Skin color', right_on='Skin color')[['Race_x', 'Race_y']]

# Tenemos duplicados!

df.merge(df, left_on='Skin color', right_on='Skin color')[
    ['Race_x', 'Race_y']
].drop_duplicates()

# Tenemos que sacar los que son iguales en ambas columnas!

same_skin_color = df.merge(df, left_on='Skin color', right_on='Skin color')[
    ['Race_x', 'Race_y']
].drop_duplicates()
same_skin_color[same_skin_color.Race_x != same_skin_color.Race_y]

# Por último, para ver los pares únicos

same_skin_color = df.merge(df, left_on='Skin color', right_on='Skin color')[
    ['Race_x', 'Race_y']
].drop_duplicates()
same_skin_color[
    (same_skin_color.Race_x != same_skin_color.Race_y)
    & (same_skin_color.Race_x > same_skin_color.Race_y)
]

# # Groupby

# queremos ver los nombres
df = df.reset_index()

df.groupby("Race")

df.columns


# +
def perc_good(grouping):
    """Devuelve el porcentaje que son 'good'."""
    return (grouping == 'good').mean() * 100.0


df.groupby("Race").agg(
    {
        'name': 'count',
        'Height': ['mean', 'std'],
        'Weight': ['mean', 'std'],
        'Alignment': perc_good,
    }
).head(20)
