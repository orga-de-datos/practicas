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

import miceforest as mf
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
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

# En esta sección mostraremos las principales estrategias para convertir variables según su tipo. <br>
# Cabe aclarar que veremos un set de posibilidades sin evaluar el algoritmo a utilizar, que es parte fundamental de la decisión final que se tome sobre el manejo de cada variable.

# ## Categóricas de baja cardinalidad

# Aparece el concepto de *Encoder*, asociado a transformaciones sobre variables categóricas. Tenemos muchos tipos de encoders veamos los principales.

# ### Ordinal Encoder

# Por cada valor de la variable categórica asigna un valor entero

oe = OrdinalEncoder()
columns_to_encode = ['Eye color', 'Gender']
# Convertimos nulos a string 'nan', es decir un valor posible mas
int_values = oe.fit_transform(dataset[columns_to_encode].astype(str))

# Mostramos primeros 15 valores
pd.concat(
    [
        pd.DataFrame(int_values[:15], columns=columns_to_encode).add_suffix('_encoded'),
        # inversión de la transformación
        pd.DataFrame(
            oe.inverse_transform(int_values)[:15], columns=columns_to_encode
        ).add_suffix('_reverted'),
    ],
    axis=1,
)

# >Observamos que los nulos se codifican con un valor entero propio

# >Recordar que esta transformación es conveniente en categóricas ordinales (no es el caso del ejemplo) ya que asigna un orden a los elementos

# ### Label Encoder

# Es exactamente la misma idea pero esperando una sola variable ya que se usa para encodear la variable target de un modelo predictivo

le = LabelEncoder()
# Convertimos nulos a string 'nan', es decir un valor posible mas
int_values = le.fit_transform(dataset['Alignment'].astype(str))

# Mostramos primeros 15 valores
pd.concat(
    [
        pd.Series(int_values[:5], name='encoded'),
        # inversión de la transformación
        pd.Series(le.inverse_transform(int_values)[:5], name='reverted'),
    ],
    axis=1,
)

# ### One Hot Encoding

# Crea una columna binaria por cada valor de la variable

pd.options.display.max_columns = None

ohe = OneHotEncoder()
cols = ohe.fit_transform(
    dataset['Eye color'].astype(str).values.reshape(-1, 1)
).todense()
# creo dataframe con las columnas creadas
cols = pd.DataFrame(cols, columns=ohe.categories_[0]).add_prefix('Eye color_')
# agrego al dataframe y elimino variable origen
cols = pd.concat([dataset, cols], axis=1).drop(['Eye color'], axis=1)
print("Valores únicos: ", dataset['Eye color'].nunique() + 1)  # +1 por null value
display(cols.head(2))
print(cols.shape)

# Otra solución para OneHotEncoding implementada en pandas:

with_dummies = pd.get_dummies(dataset, columns=['Eye color'], dummy_na=True)
display(with_dummies.head(2))
print(with_dummies.shape)

# Para evitar problemas de colinealidad se debe excluir una categoría del set (la ausencia de todas - vector de 0s - indica la presencia de la categoría faltante) <br>
# La función de pandas ya viene con una parámetro para esto:

with_dummies = pd.get_dummies(
    dataset, columns=['Eye color'], dummy_na=True, drop_first=True
)
display(with_dummies.head(2))
print(with_dummies.shape)

# La necesidad de eliminar una columna se ve más claramente para una categórica de dos valores, veamos el caso de *Gender*

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
    display(round(df[['Weight', 'Height']].describe(), 2))


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


def transform(cols, cols_to_transform, scaler):
    values = scaler.transform(cols)
    return df[['name', 'Alignment']].join(
        pd.DataFrame(values, columns=cols_to_transform)
    )


scalers = [
    StandardScaler(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    MinMaxScaler(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    RobustScaler(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    PowerTransformer(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    Normalizer(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
]

cols_to_transform = ['Weight', 'Height']
df_to_scale = df[cols_to_transform].dropna()

for i in scalers:
    fitted_scaler = get_fitted_scaler(df_to_scale, i)
    df_transformed = transform(df_to_scale, cols_to_transform, fitted_scaler)
    plot_weight_vs_height(df_transformed, i.__class__.__name__)

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

# eliminar filas con alto porcentaje de nulos
null_max = 0.75
display(dataset[dataset.isnull().sum(axis=1) / (dataset.shape[1] - 1) > null_max])
dataset[~(dataset.isnull().sum(axis=1) / (dataset.shape[1] - 1) > null_max)].shape

# eliminar columnas con nulos
less_cols = dataset.dropna(axis=1)
less_cols.shape

# Nos quedaría una sola columna sin nulos

# eliminar columnas con alto porcentaje de nulos
null_max = 0.75
display(dataset.loc[:, dataset.isnull().sum(axis=0) / dataset.shape[0] > null_max])
dataset.loc[:, ~(dataset.isnull().sum(axis=0) / dataset.shape[0] > null_max)].shape


# ## Univariadas

# Usa sólo información de la columna en cuestión para completar nulos

# ### Categóricas

# Como vimos, los encoders solucionan el problema de nulos ya que imputan con la misma lógica que para los demás valores de la variable

# ### Numéricas

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
    )


display(
    dataset[['name']]
    .join(compare_imputers(dataset, 'Weight'))[dataset['Weight'].isna()]
    .head(5)
)
display(
    dataset[['name']]
    .join(compare_imputers(dataset, 'Height'))[dataset['Height'].isna()]
    .head(5)
)


# ## Multivariada

# Usa información de todas las variables para la imputación.<br>

# Veamos un ejemplo con KNN (lo verán en detalle en las próximas clases).

# +
def hashing_encoding(df, cols, data_percent=0.85, verbose=False):
    for i in cols:
        val_counts = df[i].value_counts(dropna=False)
        s = sum(val_counts.values)
        h = val_counts.values / s
        c_sum = np.cumsum(h)
        c_sum = pd.Series(c_sum)
        n = c_sum[c_sum > data_percent].index[0]
        if verbose:
            print("n hashing para ", i, ":", n)
        if n > 0:
            fh = FeatureHasher(n_features=n, input_type='string')
            hashed_features = fh.fit_transform(
                df[i].astype(str).values.reshape(-1, 1)
            ).todense()
            df = df.join(pd.DataFrame(hashed_features).add_prefix(i + '_'))
    return df.drop(columns=cols)


def knn_imputer(df):

    cat_cols = ['Gender', 'Eye color', 'Race', 'Hair color', 'Publisher', 'Skin color']

    # Aplicamos hashing para las categoricas
    df = hashing_encoding(df, cat_cols)
    # Eliminamos name y alignment para imputar
    df = df.drop(columns=['name', 'Alignment'])

    # definimos un n arbitrario
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df


knn_imputation = knn_imputer(dataset).add_suffix('_knn')
display(
    dataset[['name', 'Weight', 'Height']]
    .join(knn_imputation[['Weight_knn', 'Height_knn']])[
        (dataset.Weight.isna() | dataset.Height.isna())
    ]
    .head(5)
)


# -

# Otro ejemplo, esta vez con RandomForest (también entraremos en detalle próximamente).

# +
def forest_imputer(df):

    cat_cols = ['Gender', 'Eye color', 'Race', 'Hair color', 'Publisher', 'Skin color']

    # Aplicamos hashing para las categoricas
    df = hashing_encoding(df, cat_cols)
    # Eliminamos name y alignment para imputar
    df = df.drop(columns=['name', 'Alignment'])
    # df.iloc[:,2:] = df.iloc[:,2:].apply(lambda x: x.astype('category'),axis=1)

    # El ampute se puede usar para validar el imputador sobre data conocida
    # df_amp = mf.ampute_data(df, perc=0.25, random_state=0)

    kernel = mf.KernelDataSet(
        df,
        save_all_iterations=True,
        random_state=0,
        variable_schema=['Weight', 'Height'],
    )

    # Run the MICE algorithm for 20 iterations on each of the datasets
    kernel.mice(20, n_jobs=2)
    return kernel.complete_data()


forest_imputation = forest_imputer(dataset).add_suffix('_forest')
display(
    dataset[['name', 'Weight', 'Height']]
    .join(forest_imputation[['Weight_forest', 'Height_forest']])[
        (dataset.Weight.isna() | dataset.Height.isna())
    ]
    .head(5)
)
# -

# # Selección de variables

# La librería sklearn tiene un [apartado exclusivo](https://scikit-learn.org/stable/modules/feature_selection.html) con herramientas implementadas para la selección de variables <br>
# Veamos algunas implementaciones

# Por varianza, se define un umbral mínimo para considerar variables. Por defecto elimina las features de varianza 0 (sin cambios) <br>
# Como en el set no tenemos ejemplos, agreguemos variables con esas condiciones

df = dataset.copy()
df['with_zero_variance'] = 10
df['with_low_variance'] = np.random.uniform(0, 0.2, df.shape[0])

df.head()

df.var()


# +
def filter_by_variance(df, threshold):
    '''Devuelve el dataset filtrado por varianza para las columnas que corresponda'''
    # Columnas con varianza calculable
    cols = df.var().index.values

    selector = VarianceThreshold(threshold=threshold)
    # calculo varianzas
    vt = selector.fit(df[cols])

    ## vt.get_support() me da los indices de las columnas que quedaron
    result = df[cols].loc[:, vt.get_support()]
    return df.loc[:, ~df.columns.isin(cols)].join(result)


display(filter_by_variance(df, 0).head(2))
display(filter_by_variance(df, 0.5).head(2))
# -

# # Poniendo todo en práctica

# Supongamos que tenemos que predecir el bando de un superhéroe dadas las variables del set. <br> Apliquemos lo visto hasta ahora

# empezamos de cero, leyendo el dataset
df = pd.read_csv('../datasets/superheroes.csv')
df.shape

# 1- Reemplazamos valores codificados como null

df = df.replace('-', np.nan)
df = df.replace(-99, np.nan)

# 2- Eliminamos filas duplicadas

df = df.drop_duplicates().reset_index(drop=True)

# 3- Eliminamos filas con mas de 75% de nulos

df = df[~(df.isnull().sum(axis=1) / (df.shape[1] - 1) > 0.75)].reset_index(drop=True)

# 4-Eliminamos filas con bando nulo o neutral

df = df[~df.Alignment.isin(['neutral', np.nan])].reset_index(drop=True)
df.shape

# Vemos como quedo la distribución por bando

fig = px.pie(
    values=df['Alignment'].value_counts().values,
    names=df['Alignment'].value_counts().index,
    title='Distribución por bando',
)
fig.update_layout(autosize=False, width=1000)
fig.show()

# 5- Análisis sobre *Skin color*

df['Skin color'].value_counts(dropna=False)

# n de casos por color de piel / bando
skin_group = (
    df.fillna('nulls').groupby(['Skin color', 'Alignment']).size().to_frame('count')
)
# porcentaje por color de piel / bando
skin_group = skin_group.join(
    skin_group.groupby(level=0)
    .apply(lambda x: 100 * x / float(x.sum()))
    .rename(columns={'count': 'percent'})
).reset_index()
# ordeno por cantidad de casos
skin_group = skin_group.sort_values(by='count', ascending=False)
# bar chart
px.bar(
    skin_group,
    x='Skin color',
    y='percent',
    color='Alignment',
    hover_data=['percent', 'count'],
)

# >Claramente la variable tiene muy pocos datos, tal vez un relevamiento mas preciso pueda hacer que tenga peso en el futuro. Por ahora no vamos a considerarla.

# Eliminamos skin color
df.drop(columns=['Skin color'], inplace=True)
df.shape

# 6- Paso todas las categóricas a minúsculas

df.loc[:, df.dtypes == 'object'] = df.loc[:, df.dtypes == 'object'].apply(
    lambda x: x.str.lower(), axis=1
)

# 7- Análisis sobre *name*

# Nombres duplicados
df['name'].value_counts()[df['name'].value_counts() > 1]

df[df['name'] == 'spider-man']


# +
def most_common(x):
    if x.value_counts().size == 0:
        return np.nan
    return x.value_counts().index[0]


# defino la forma de agregación para cada variable
df = (
    df.groupby('name')
    .agg(
        {
            'Gender': most_common,
            'Eye color': most_common,
            'Race': most_common,
            'Hair color': most_common,
            'Height': 'mean',
            'Publisher': most_common,
            'Alignment': most_common,
            'Weight': 'mean',
        },
        axis=0,
    )
    .reset_index()
)

df.shape
# -

# 8- Creamos nuevas variables

# Investigando sobre la composición de los nombres, vemos que los que contienen *captain* son buenos

# aux = df.groupby('Alignment').apply(lambda x: x['name'].str.split().apply(pd.Series).stack()).to_frame('word').reset_index()[['Alignment','word']].groupby('Alignment')['word'].value_counts().to_frame('n').reset_index()
# aux.groupby('Alignment').head(5)
display(df.loc[df.name.str.contains('captain'), ['name', 'Alignment']])
df['is_captain'] = np.where(df['name'].str.contains('captain'), 1, 0)
df.shape

# pocas mujeres superheroes, menos aún villanas
df.groupby(['Alignment', 'Gender']).size().groupby(level=0).apply(
    lambda x: 100 * x / float(x.sum())
)

# 9- Encodeamos categóricas

cat_cols = ['Gender', 'Eye color', 'Race', 'Hair color', 'Publisher']
# Aplicamos hashing para las categoricas
df = hashing_encoding(df, cat_cols, verbose=True)
df.head()
