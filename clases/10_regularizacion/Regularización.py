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
#     display_name: env3
#     language: python
#     name: env3
# ---

# ### Regularización
#
# Utilizaremos el dataset "Boston" que nos provee sklearn, el mismo contiene valores de propiedades en el estado de Boston.

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import Lasso,LassoCV

import warnings
warnings.filterwarnings(action='ignore')
# -

boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)

boston_df['Price'] = boston.target
boston_df.head()

boston_df.info()

boston_df.describe()

# ### Semántica de los datos
#
# Cada registro de la base de datos describe un suburbio de Boston. Los datos se obtuvieron del Área Estadística Metropolitana Estándar de Boston (SMSA) en 1970. Los atributos se definen de la siguiente manera (tomados del Repositorio de Aprendizaje Automático de UCI1):
#
#  - CRIM: tasa de delincuencia per cápita
#  - ZN: proporción de terreno residencial dividido en zonas para lotes de más de 25,000 pies cuadrados.
#  - INDUS: proporción de héctareas comerciales no minoristas por ciudad
#  - CHAS: Variable ficticia de Charles River (= 1 si el tramo limita con el río; 0 en caso contrario)
#  - NOX: concentración de óxidos nítricos (partes por 10 millones)
#  - RM: número medio de habitaciones por vivienda
#  - AGE: proporción de unidades ocupadas por sus propietarios construidas antes de 1940
#  - DIS: distancias ponderadas a cinco centros de empleo de Boston
#  - RAD: índice de accesibilidad a carreteras radiales
#  - TAX: tasa de impuesto a la propiedad de valor total por  10,000
#  - PTRATIO: proporción alumno-maestro por ciudad
#  - B: proporción de gente negra
#  - LSTAT:% de menor estatus de la población
#  - MEDV: Valor medio de las viviendas ocupadas por sus propietarios en 1000
#
# #### Observaciones
#
#  - Podemos ver en quién armó los datos una clara discriminación racial, incluyen como dato potencialmente relevante para el valor de la propiedad, la proporción de gente negra. Mas allá de si esto funciona bien como predictor nos va abriendo las puertas para uno de los temas que veremos hacia el final de la materia: Ética de la inteligencia artificial
#  - Podemos ver que los atributos de entrada tienen una mezcla de unidades.
#
# Veamos ahora como es la correlación de los features de este dataset

# +

corrmat = boston_df.corr()
k = 14  # number of variables for heatmap
cols = corrmat.nlargest(k, 'Price')['Price'].index
cm = np.corrcoef(boston_df[cols].values.T)
sns.set(font_scale=1.25)
fig, ax = plt.subplots(figsize=(15, 15))
hm = sns.heatmap(
    cm,
    cbar=True,
    annot=True,
    square=True,
    fmt='.2f',
    annot_kws={'size': 10},
    yticklabels=cols.values,
    xticklabels=cols.values,
)

plt.show()
# -

# ### Regresion OLS
#
# Separemos nuestros datos en las variables independientes (X) y dependiente (y)

# Comenzamos por hacer una regresión lineal para ver cual sería la performance sin regularización y que valor le asigna a cada feature. Para tener una buena estimación lo haremos mediante cross validation con un k-fold = 5

# +
boston_df['Price'] = boston.target
X_train = boston_df.drop("Price", axis=1)
y_train = boston_df.filter(items=["Price"])

linreg = LinearRegression()
rmse = np.sqrt(
    -cross_val_score(
        LinearRegression(), X_train, y_train, scoring="neg_mean_squared_error", cv=5
    )
)
rmse.mean()
# -

linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg.coef_[0]

# +

coeficientes = pd.DataFrame(
    {'Feature': X_train.columns, 'Beta': linreg.coef_[0]}, columns=['Feature', 'Beta']
)
coeficientes.sort_values(by=['Beta'])
# -

coef = pd.Series(linreg.coef_[0], index=X_train.columns)
imp_coef = pd.concat([coef.sort_values()])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind="barh")
plt.title("Coeficientes en el Modelo Ridge ")


# ### Ridge
# Ahora procederemos a escalar los datos mediante el cálculo visto en la teórica

# +

scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object

scaler.fit(boston_df)
scaled_df = scaler.transform(boston_df)
names = boston_df.columns
scaled_boston_df = pd.DataFrame(scaled_df, columns=names)


# -

scaled_boston_df.describe()

# Volvemos a hacer split de la data poque ahor queremos trabajar con nuestro datos escalados

X_train = scaled_boston_df.drop("Price", axis=1)
y_train = boston_df.filter(items=["Price"])


# Ahora definiremos una nueva función para calcular el RMSE de un modelo mediante cross validation con un k-fold de 5


def rmse_cv(model, X_train, y_train):
    rmse = np.sqrt(
        -cross_val_score(
            model, X_train, y_train, scoring="neg_mean_squared_error", cv=5
        )
    )
    return rmse


# Comenzaremos por ajustar nuestra regresión mediante Ridge, probaremos con distintos valores de alpha y luego graficaremos el error (RMSE) en función de alpha

alphas = [
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.3,
    1,
    3,
    5,
    10,
    30,
    50,
    55,
    75,
    100,
    120,
    150,
]
cv_ridge = [rmse_cv(Ridge(alpha=alpha), X_train, y_train).mean() for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index=alphas)
cv_ridge.plot(title="Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

cv_ridge.min()

# Podemos ver que el error alcanza un mínimo de 5.4 con un alpha alrrededor de 80, mas precisamente 75 según el array de alphas que cargamos

# +
ridge = Ridge(alpha=75)
ridge.fit(X_train, y_train)


coeficientes = pd.DataFrame(
    {'Feature': X_train.columns, 'Beta': ridge.coef_[0]}, columns=['Feature', 'Beta']
)
coeficientes.sort_values(by=['Beta'])
# -

coef = pd.Series(ridge.coef_[0], index=X_train.columns)
imp_coef = pd.concat([coef.sort_values()])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind="barh")
plt.title("Coeficientes en el Modelo Ridge ")

# ### Lasso
#
# Veamos ahora como se ajusta mediante Lasso

# +
alphas = [
    0.0000001,
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.3,
    1,
    3,
    5,
    10,
    30,
    50,
    55,
    
]
cv_lasso = [rmse_cv(Lasso(alpha=alpha), X_train, y_train).mean() for alpha in alphas]
# -

cv_lasso_serie = pd.Series(cv_lasso, index=alphas)
cv_lasso_serie.plot(title="Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

# ¿Por qué el error crece tanto más rápido que Ridge?

cv_lasso_serie.min()

model_lasso = Lasso(alpha=np.array(alphas)[cv_lasso==cv_lasso_serie.min()])
model_lasso.fit(X_train, y_train)

# Dado que Lasso tiende a anular totalmente algunos coeficientes, podemos ver cuantas y cuales de nuestros coeficientes fueron anulados y por lo tanto, cuales variables pueden ser prescindibles para este ajuste

coef = pd.Series(model_lasso.coef_, index=X_train.columns)
print(
    "Lasso seleccionó "
    + str(sum(coef != 0))
    + " variables y eliminó las otras "
    + str(sum(coef == 0))
    + " variables"
)

imp_coef = pd.concat([coef.sort_values()])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind="barh")
plt.title("Coeficientes en el Modelo Lasso ")

# Vemos que le dió un valor mucho mas chico a AGE con respecto a Ridge aunque no llegó a eliminarla del todo. 
#
# ¿A que puede deberse esto? 
#
# Ya que Lasso regulariza mucho más "rapido" que Ridge, hagamos foco en los landas mas chicos para ver que pasa

# +
alphas = [
     0.0000001,
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.005,
    0.1,
    0.3,
    1,
    
    
]
cv_lasso = [rmse_cv(Lasso(alpha=alpha), X_train, y_train).mean() for alpha in alphas]
# -

cv_lasso_serie = pd.Series(cv_lasso, index=alphas)
cv_lasso_serie.plot(title="Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

# Podemos ver que para un landa de 0.4 el error es casi el mismo que aplicar OLS, chequemoslo

# +

model_lasso = Lasso(alpha=0.4)
rmse_cv(model_lasso, X_train, y_train).mean()
# -

# El error es parecido, sin embargo es un poco mas bajo que con OLS.
#
# ¿Que habrá pasado ahora con los coeficientes de las variables?

model_lasso.fit(X_train, y_train)
coef = pd.Series(model_lasso.coef_, index=X_train.columns)
print(
    "Lasso seleccionó "
    + str(sum(coef != 0))
    + " variables y eliminó las otras "
    + str(sum(coef == 0))
    + " variables"
)

imp_coef = pd.concat([coef.sort_values()])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind="barh")
plt.title("Coeficientes en el Modelo Lasso ")

# Que sucede si artificialmente creamos una nueva columna correlacionada linealmente con otra?

X_train['new_col'] = X_train['RM'] * 2

# +
alphas = [
    0.4
    
]
cv_lasso = [rmse_cv(Lasso(alpha=alpha), X_train, y_train).mean() for alpha in alphas]
cv_lasso_serie.min()
# -

cv_lasso_serie = pd.Series(cv_lasso, index=alphas)
model_lasso = Lasso(alpha=np.array(alphas)[cv_lasso==cv_lasso_serie.min()])
model_lasso.fit(X_train, y_train)

coef = pd.Series(model_lasso.coef_, index=X_train.columns)
print(
    "Lasso seleccionó "
    + str(sum(coef != 0))
    + " variables y eliminó las otras "
    + str(sum(coef == 0))
    + " variables"
)

imp_coef = pd.concat([coef.sort_values()])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind="barh")
plt.title("Coeficientes en el Modelo Lasso ")

# Podemos ver que ahora elimina 3 variables, dado que la nueva que generamos esta correlacionada con la RM.
#
# También podemos ver que en vez de eliminar nuestra nueva  variable elimina la original.....
#
# ¿Porque podrá ser esto?

# ### Elastic Net
#
# Veamos ahora como ajusta Elastic Net

# +
alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1, 3, 5]

cv_elasticNet = [
    rmse_cv(ElasticNet(alpha=alpha, l1_ratio=0.5), X_train, y_train).mean()
    for alpha in alphas
]
# -

cv_elasticNet = pd.Series(cv_elasticNet, index=alphas)
cv_elasticNet.plot(title="Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

cv_elasticNet.min()

# +
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)

coeficientes = pd.DataFrame(
    {'Feature': X_train.columns, 'Beta': elastic.coef_}, columns=['Feature', 'Beta']
)
coeficientes.sort_values(by=['Beta'])
# -

coef = pd.Series(elastic.coef_, index=X_train.columns)
imp_coef = pd.concat([coef.sort_values()])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind="barh")
plt.title("Coeficientes en el Modelo Ridge ")

# Veamos que pasa si al l1_ratio lo llevamos a 1

# +
elastic = ElasticNet(alpha=0.4, l1_ratio=1)
elastic.fit(X_train, y_train)

coeficientes = pd.DataFrame(
    {'Feature': X_train.columns, 'Beta': elastic.coef_}, columns=['Feature', 'Beta']
)
coeficientes.sort_values(by=['Beta'])
# -

coef = pd.Series(elastic.coef_, index=X_train.columns)
imp_coef = pd.concat([coef.sort_values()])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind="barh")
plt.title("Coeficientes en el Modelo Ridge ")

# Obtenemos la misma estimación que para Lasso
