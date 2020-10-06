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

from collections import Counter

# +
import pandas as pd

GSPREADHSEET_DOWNLOAD_URL = (
    "https://docs.google.com/spreadsheets/d/{gid}/export?format=csv&id={gid}".format
)

df = pd.read_csv(
    GSPREADHSEET_DOWNLOAD_URL(gid="1nuJAaaH_IP8Q80CsyS940EVaePkbmqhN3vlorDxYMnA")
)
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

df

df = df.dropna(subset=['Skin color'])
df

# # Unir información de distintas tablas
#
# Veamos si hay razas que tengan el mismo color de piel.

df.merge(df, left_on='Skin color', right_on='Skin color')[['Race_x', 'Race_y']]

# Tenemos duplicados!

df.merge(df, left_on='Skin color', right_on='Skin color')[['Race_x', 'Race_y']]

# Tenemos que sacar los que son iguales en ambas columnas!

same_skin_color = df.merge(df, left_on='Skin color', right_on='Skin color')[
    ['Race_x', 'Race_y']
].drop_duplicates()
same_skin_color[same_skin_color.Race_x != same_skin_color.Race_y]

# +
# pd.merge?
# -

df1 = pd.DataFrame({'col': [1, 2, 3], 'val': [10, 11, 12]})
df2 = pd.DataFrame({'col': [2, 3, 4], 'val': [13, 14, 15]})

df1

df2

pd.merge(df1, df2, how='left', left_on='col', right_on='col')

pd.merge(df1, df2, how='inner', left_on='col', right_on='col')

pd.merge(df1, df2, how='right', left_on='col', right_on='col')

pd.merge(df1, df2, how='outer', left_on='col', right_on='col')

# +
# df1.merge?

# +
# df1.join?
# -

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

df

df.groupby("Race")

df.groupby("Race").agg(list)

(df['Alignment'] == 'good').mean() * 100

(df['Alignment'] == 'good').sum() / (df['Alignment'] == 'good').size

df.groupby("Race")['Alignment'].apply(len)


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

# Y si lo queremos como porcentajes?

df.Race.value_counts() / df.Race.value_counts().sum() * 100

df.Race.value_counts(normalize=True)

# Veamos como podemos obtener las filas del dataframe original donde la columna `Race` este entre aquellos valores con mas del 5% de repeticiones.

over5 = df.Race.value_counts(normalize=True) > 0.05
mutants_over5 = df.Race.value_counts()[over5]

# Teniendo la indexacion, veamos como resolverlo con `isin`

df[df.Race.isin(mutants_over5.index)].head(5)

# Alternativamente, con `merge`

df.merge(mutants_over5, left_on='Race', right_index=True, how='inner')

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

df_marvel = df[df.Publisher == 'Marvel Comics'].copy()

df_marvel.loc[:, 'numeric_alineation'] = df_marvel.Alignment.apply(alignment_to_numeric)

df_marvel.head()

df_marvel.numeric_alineation.mean()

# Una excelente guía al respecto: [link](https://www.dataquest.io/blog/settingwithcopywarning/)

# # Ordenando

df.set_index('name').sort_index()

df.sort_values(by=['Height', 'Weight'], ascending=False)

# # Operaciones de strings

df.name.apply(lambda x: x.lower())

# Entre [otras](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html)

# # Manejo de fechas

# ### Timestamp

pd.Timestamp("2020-09-30 04:32:18 PM")

# ### DatetimeIndex

fechas = ['2020-03-20', '2020-03-18', '2020/09/30']
indice_fechas = pd.DatetimeIndex(fechas)
indice_fechas

descripciones = ['cuarentena', 'cumpleañito', 'hoy']
desc_serie = pd.Series(data=descripciones, index=indice_fechas)
desc_serie

# ### to_datetime

pd.to_datetime('2020/03/30 17:43:09')

pd.to_datetime(fechas)

serie_fechas = pd.Series(
    ['September 22nd, 2019', '22, 09, 2020', 'Una fecha', 'Oct 15th, 2020']
)
pd.to_datetime(serie_fechas, errors='coerce')

# ### Rangos

rango_de_tiempo = pd.date_range(start='25/06/2019', end='25/06/2020', freq='D')
rango_de_tiempo

rango_de_tiempo = pd.date_range(start='25/06/2019', end='25/06/2022', freq='A')
rango_de_tiempo

desc_serie

# ### Filtro por fecha
#
# Usamos un dataset que registra el clima y demás datos para distintas fechas de [alquiler de bicicletas](https://www.kaggle.com/c/bike-sharing-demand/data?select=train.csv)

bicis_df = pd.read_csv(
    GSPREADHSEET_DOWNLOAD_URL(gid="1YocUXbrd6uYpOLpU53uMS-AD9To8y_r30KbZdsSSiVQ")
).set_index('datetime')
bicis_df

bicis_df.loc['2012-12-19 20:00:00']

# + tags=["raises-exception"]
# Para poder quedarnos con un rango de fechas, debemos tener el índice ordenado
bicis_df = bicis_df.sort_index()
bicis_df.loc['2012-11-19 20:00:00':'2012-12-30 20:00:00']

bicis_df.truncate(before='2012-11-19 22:00:00', after='2012-12-01 00:00:00')
# -

# ### `dt` accessor

# Permite obtener propiedades de tipo fecha de una Series
fechas_series = pd.Series(pd.date_range('2020-09-30 00:00:41', periods=3, freq='s'))
fechas_series

fechas_series.dt.second

fechas_series = pd.date_range(
    start='22/06/2019', end='28/06/2019', freq='D'
).to_series()
fechas_series.dt.dayofweek

# # Cómo seguir
#
# La documentación oficial es excelente. Un buen repaso es [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) y para profundizar, los links de la izquierda.
