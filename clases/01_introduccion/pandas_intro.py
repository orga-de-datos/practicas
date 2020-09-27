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

from collections import Counter

# +
import pandas as pd

df = pd.read_csv('../../datasets/superheroes.csv')
# -

# También podemos crear dataframes desde listas, diccionarios y otras estructuras.

# ## Inspeccionando un dataframe

# vemos los primeros elementos
df.head()

df.tail()

df.sample()

df.sample(10)

# ## Información sobre un dataframe

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


# ## Múltiples condiciones

df[(df['Skin color'] == 'blue') & (df['Publisher'] == 'Marvel Comics')]

df[(df['Skin color'] == 'blue') | (df['Skin color'] == 'green')].sample(10)

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

# Aprovechamos a ver como asignar una nueva columna

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

# ### Concatenar tablas

df_1 = pd.DataFrame({'col_1': range(1, 10), 'col_2': range(1, 10)})
df_2 = pd.DataFrame({'col_1': range(11, 20), 'col_2': range(11, 20)})

df_1.pipe(len)

df_2.pipe(len)

df_1.head()

df_2.head()

df_concat = pd.concat([df_1, df_2])
df_concat

df_concat.pipe(len)

# # Agrupaciones

# ## Groupby

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
# -

# Algunas agregaciones tienen métodos para realizarlos directamente

df.Race.value_counts()

# ## Pivoting


pd.pivot_table(
    df,
    index='Race',
    columns=['Gender'],
    values=['Height', 'Weight', 'Alignment'],
    aggfunc={
        'Height': 'mean',
        'Weight': 'mean',
        'Alignment': lambda x: Counter(x).most_common(1)[0][0],
    },
)

# # Sobre vistas y columnas

df_marvel = df[df.Publisher == 'Marvel Comics']


# +
def alignment_to_numeric(alignment):
    return {'bad': -1, 'good': 1, 'neutral': 0}[alignment]


df_marvel['numeric_alineation'] = df_marvel.Alignment.apply(alignment_to_numeric)
# -

df_marvel.loc[:, 'numeric_alineation'] = df_marvel.Alignment.apply(alignment_to_numeric)

df_marvel.head()

df_marvel.numeric_alineation.mean()

# # Ordenando

df.sort_index()

df.sort_values(by=['Height', 'Weight'], ascending=False)

# # Operaciones de strings

df.name.str.lower()

# # Cómo seguir
#
# La documentación oficial es excelente. Un buen repaso es [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) y para profundizar, los links de la izquierda.
