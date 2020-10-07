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

# +
# import miceforest as mf
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
# -

# Leemos el dataset, que está en formato CSV desde google drive.  
# Adicionalmente renombro a las columnas en un formato mas comodo de manejar

df = pd.read_csv('https://drive.google.com/uc?export=download&id=1gq-wDn_dwz_5uHSEoMYmQtnNrmnfNiXS')
df.rename(columns={c: c.lower().replace(" ","_") for c in df.columns}, inplace=True)

df.head()

# Usando pandas profiling
report = ProfileReport(df, title='superhéroes', minimal=True)
report

# # Verificando la "calidad" de los datos

# ### Chequeo de valores NULOS

# Los valores nulos pueden tener distintas formas de ser represantados:
# - nan
# - vacios ""
# - algun caracter especial "_", "?", "NULL"
# - valores que no tienen sentido dado la variable (ej: distancia recorrida: -1)



# En el caso del dataset de los superheroes, que valores nulos vemos?
# > En el head vimos que "Skin color" tiene un "-" .  
# Que otras columnas tienen "-" ?

tienen_guion = df.astype('str').eq('-').any(0)
tienen_guion

# > Vemos que las columnas:
# - gender
# - eye_color
# - race
# - hair_color
# - skin_color
# - alignment

# +
# Asi se haria para quedarnos con las filas que tienen "-" en alguna de sus columnas

df[df.astype('str').eq('-').any(1)]
# -

# > Convertimos los valores "-" en nan

df = df.replace('-', np.nan)

# > Hay alguna variable que este en "blanco" ?

df.astype('str').eq('').any()

# > Vemos que ninguna variable esta en blanco  
# > Nota: Siendo un poco más putitanos deberiamos chequear que :
# - por medio de la regex "^-*[0-9]*\$" fijarnos que las columnas de numeros contengan solo numeros
# - por medio de la regex "^ *\$" fijarnos que no haya varios valores de vacios (lo mismo para otros caracteres que sospechamos que pueden ser usados para representar un valor NULO



# #### Chequeo de variables numericas:
#
# > Ahora vamos a chequear los limites de las columnas que tengan valores numericos.  
# > las columnas edad y peso no deberian tener valores negativos

columnas_con_numeros = ['height', 'weight']
(df[columnas_con_numeros] < 0).any()

for c in columnas_con_numeros:
    print(c)
    display(df[df[c]<0][c].value_counts())
    print()

# > Pasamos los valores -99 de las columnas height y weight a nan

df = df.replace({'height':-99.0, 'weight':-99.0}, np.nan)

# > Dependiendo del dataset, a veces tenemos informacion duplicada que no queremos  
# > Alertamos una fila duplicada, la eliminamos

# >  df.duplicated() devuelve una serie de booleanos indicando se una fila es duplicada o no  
# df.drop_duplicates() devuelve un dataframe nuevo con las filas duplicadas eliminadas  
# Nota: podemos pasarle el parametro subset=\<columnas a mirar> para solo considerar algunas columnas para ver si esta duplicado o no

df[df.duplicated(keep=False)]

size_antes = len(df)
df = df.drop_duplicates()
size_despues = len(df)
print(f'se eliminaron: {size_despues-size_antes} filas duplicadas')

# > Nota: a veces es util "resetear" el indice despues de eliminar filas (ya sea por drop duplicates o por algun otro filtro)

df.reset_index(drop=True, inplace=True)



# # Conversion de Variables

# En esta sección mostraremos las principales estrategias para convertir variables según su tipo.

# ## Conversion de variables categoricas:
# Hay veces que tenemos que trabajar un poco en las variables que tenemos para que puedan ser usadas en los modelos.

# ### Preguntas antes de empezar:
# - Alta vs Baja Cardinalidad de una Variable Categorica, que significa ?
# - Que significa que una Variable contenga informacion del Orden?

# ### Categóricas de baja cardinalidad

# Sklearn eligió unos nombres un poco desafortunados para los metodos de traformacion de variables categoricas:  
# Los principales son:  
# - Ordinal Encoder
# - Label Encoder
# - One Hot Encoding
#
# No hay mucha diferencia entre Label Encoder y Ordinal Encoder ya que los dos tienen la misma logica, la principal diferencia es que Label Encoder esta pensada para trabajar con solo una serie por vez, en cambio Ordinal Encoder puede trabajar con todas las columnas al mismo tiempo.  
#
# El motivo que digo que los nombres son un poco desafortunados es que Ordinal Encoder da a entender que hay una especie de orden en el encoding que haga, pero la verdad es que no hay ningun criterio en el mismo, si queremos que la transformacion mantenga un cierto orden que nosotros queremos, entonces tenemos que hacer nosotros mismos la misma.

# #### Ordinal Encoder 
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
#

# Por cada valor de la variable categórica asigna un valor entero

oe = OrdinalEncoder()
columns_to_encode = ['eye_color', 'gender']
try:
    df[['eye_color_encoded', 'gender_encoded']] = oe.fit_transform(df[columns_to_encode])
except Exception as upa:
    print(upa)

# Convertimos nulos a string 'nan', es decir un valor posible mas para que no explote
df[['eye_color_encoded', 'gender_encoded']] = oe.fit_transform(df[columns_to_encode].astype(str))

# > Una funcionalidad MUY interesante de muchas de las clases de sklearn que ayudan en la transformacion de 
# es que tienen la transformacion INVERSA!

oe.inverse_transform(df[['eye_color_encoded', 'gender_encoded']])



# Pregunta del millon:
# - Esta todo bien con esta trasnformacion??
# - Puedo usar las columnas 'eye_color_encoded' y 'gender_encoded' ??

# #### Label Encoder
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

# Es exactamente la misma idea pero esperando una sola variable ya que se usa para encodear la variable target de un modelo predictivo

le = LabelEncoder()
# Convertimos nulos a string 'nan', es decir un valor posible mas
df['alignment_encoded'] = le.fit_transform(df['alignment'].astype(str))

df[['alignment', 'alignment_encoded']]

# > Al igual que la clase anterior, se puede usar el inverso

le.inverse_transform(df.alignment_encoded)[:10]



#
# Preguntas V/F:
# - esta bien aplicar LabelEncoder() a la columna "Alignment".
# - OrdinalEncoder() o LabelEncoder() de sklearn pueden trabajar con una supuesta columna "orden" cuyos valores son \['primero','segundo','tercero'] y van a realizar el encoding correctamente.



# > Nota:
# Hay veces que es muy util aplicar OrdinalEncoder() o LabelEncoder() a una variable NO ordinal si el modelo que va a usar los datos no va a utilizar el orden.
#

del df['alignment_encoded']
del df['eye_color_encoded']
del df['gender_encoded']

# #### One Hot Encoding

# Crea una columna binaria por cada valor de la variable

pd.options.display.max_columns = None

ohe = OneHotEncoder() #drop='first'
eye_color_encoded = ohe.fit_transform(df[['eye_color']].astype(str)).todense().astype(int)
eye_color_encoded = pd.DataFrame(eye_color_encoded).add_prefix('ec_')
df = pd.concat([df, eye_color_encoded], axis=1)

ohe.categories_

df[['eye_color'] + eye_color_encoded.columns.tolist()]

df.drop(columns=eye_color_encoded.columns.tolist(), inplace=True)

# Otra solución para OneHotEncoding implementada en pandas:

with_dummies = pd.get_dummies(df, columns=['eye_color'], dummy_na=True)
display(with_dummies.head(2))
print(with_dummies.shape)

# Para evitar problemas de colinealidad se debe excluir una categoría del set (la ausencia de todas - vector de 0s - indica la presencia de la categoría faltante) <br>
# La función de pandas ya viene con una parámetro para esto:

with_dummies = pd.get_dummies(df, columns=['eye_color'], dummy_na=True, drop_first=True)
display(with_dummies.head(2))
print(with_dummies.shape)



# > La necesidad de eliminar una columna se ve más claramente para una categórica de dos valores, veamos el caso de *Gender*

gender_dummies = pd.get_dummies(df[['gender']])
display(gender_dummies.tail(5))

# >Con una sola columna tenemos toda la información necesaria

# ## Categóricas de alta cardinalidad

# Que pasa con la variable *Race* que tiene mas de 60 valores, vamos a crear 60 variables? <br>
# Veamos la distribución de los mismos

unique_races = df['race'].value_counts(dropna=False)
display(unique_races.head(10))
unique_races.cumsum().plot(kind='bar', title="Distribución de la suma acumulativa de razas",figsize=(25,8))
plt.plot()

# >Con el top 10 cubrimos mas del 85% de la data

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
fh = FeatureHasher(n_features=10, input_type='string')
hashed_features = fh.fit_transform(df['race'].astype(str)).todense()
hashed_features = pd.DataFrame(hashed_features).add_prefix('race_')
pd.concat([df[['race']], hashed_features], ignore_index=True, axis=1)


# ## Numéricas

# En el set tenemos dos variables numéricas, *Weight* y *Height* veamos su distribución

# !pip freeze | grep pl

# +
def plot_weight_vs_height(df, title=""):
    fig = px.scatter(
        df.dropna(),
        x="weight",
        y="height",
        color="alignment",
        marginal_x="box",
        marginal_y="box",
        hover_name='name',
        title="Peso vs altura " + title,
    )
    fig.update_layout(autosize=False, width=1000)
    fig.show()
#     plt.plot()
    display(round(df[['weight', 'height']].describe(), 2))


# _df = df[df.alignment != 'neutral'].reset_index(drop=True)
plot_weight_vs_height(df , "- Valores originales")
# _df = _df[['name', 'weight', 'height', 'alignment']]
# fig = px.scatter(
#     _df.dropna(),
#     x="weight",
#     y="height",
#     color="alignment",
#     marginal_x="box",
#     marginal_y="box",
#     hover_name='name',
#     title="Peso vs altura " + 'pepe',
# )
# fig.update_layout(autosize=False, width=1000)
# fig.show()
# display(round(df[['weight', 'height']].describe(), 2))

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
    return df[['name', 'alignment']].join(
        pd.DataFrame(values, columns=cols_to_transform)
    )


scalers = [
    StandardScaler(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    MinMaxScaler(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    RobustScaler(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    PowerTransformer(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    Normalizer(),  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
]

cols_to_transform = ['weight', 'height']
df_to_scale = df[cols_to_transform].dropna()

for i in scalers:
    fitted_scaler = get_fitted_scaler(df_to_scale, i)
    df_transformed = transform(df_to_scale, cols_to_transform, fitted_scaler)
    plot_weight_vs_height(df_transformed, i.__class__.__name__)

# -


# ### Discretización

# Tranformación por la cual convertimos una variable continua en categórica

# #### Binarizer
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html
#
# Nota: me salteo el ejemplo ya que es muy simple



# ##### KBinsDiscretizer
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html

# +
enc = KBinsDiscretizer(n_bins=4, encode='ordinal')

_df = df[['weight']].dropna().reset_index(drop=True)
X_binned = enc.fit_transform(_df)
result = pd.concat([_df, pd.DataFrame(X_binned.astype(int), columns=['weight_bins'])], axis=1)

display(result.head(10))
print("Límites bins:", enc.bin_edges_)
# -

# ##### pd.qcut

# mismo ejemplo con pandas
result, bins = pd.qcut(df['weight'], 4, labels=[0, 1, 2, 3], retbins=True)
result = pd.concat([df, pd.Series(result, name='weight_bins')], axis=1)
display(result.head(5))
print("Límites bins:", bins)

# # Missings (Trabajando con valores faltantes)

# ## Opcion 1: remover los nulos del dataset

# Veamos que variables contienen nulos

df.isnull().sum()

(df.isnull().mean()*100).to_frame('porcentaje nulls')

# Veamos algunos registros de dichas variables accediendo con [.loc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html)

# +
# condicion sobre las columnas, cantidad de nulos > 0

df.loc[:, df.isnull().sum() > 0].head()
# -

# Tenemos dos tipos de variables a tratar, numéricas y categóricas. <br>
# Veamos algunas soluciones generales

# +
# eliminar filas con nulos

less_rows = df.dropna(axis=0)
less_rows.shape
# -

# Nos quedarían 50 registros válidos en el set

# +
# eliminar filas con alto porcentaje de nulos

NULL_REMOVE_PERCENT = 0.30
df[df.isnull().mean(axis=1) < NULL_REMOVE_PERCENT]

# +
# eliminar columnas con nulos

less_cols = df.dropna(axis=1)
less_cols.shape
# -

# Nos quedaría una sola columna sin nulos

# +
# eliminar columnas con alto porcentaje de nulos

NULL_REMOVE_PERCENT = 0.30
cols = df.isna().mean()
cols = cols[cols < NULL_REMOVE_PERCENT]
df[cols.index]


# -

# ## Opcion 2: completar usando info de esa columna (Univariadas)

# #### Categóricas

# Como vimos, los encoders solucionan el problema de nulos ya que imputan con la misma lógica que para los demás valores de la variable

# #### Numéricas

# Completar con la mediana, promedio, moda o constante

# +
def show_strategies(df, name_col, k=-99):
    '''Devuelve el valor de imputacion de las tres estrategias para esa columna'''
    
    _df = df[[name_col]].copy()
    s = df[name_col]
    
    _df['median'] = s.fillna(s.median())
    _df['mean'] = s.fillna(s.mean())
    _df['mode'] = s.fillna(s.mode()[0])
    _df['contant'] = k

    # vemos los valores con los que completa en cada caso
    return _df[s.isna()]


show_strategies(df, 'weight')
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
    median_imputer = get_imputer(df[[name_col]], 'median')
    mean_imputer = get_imputer(df[[name_col]], 'mean')
    mode_imputer = get_imputer(df[[name_col]], 'most_frequent')
    constant_imputer = get_imputer(df[[name_col]], 'constant', k)

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
    df[['name']]
    .join(compare_imputers(df, 'weight'))[df['weight'].isna()]
    .head(5)
)
display(
    df[['name']]
    .join(compare_imputers(df, 'height'))[df['height'].isna()]
    .head(5)
)


# ## Opcion 3: completar usando info de las demas columnas (Multivariada)

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




