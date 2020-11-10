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

# # Cancer de próstata
#
#
# Cargando bibliotecas y datos

# coding: utf-8
# %matplotlib inline  
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
import seaborn as sns
from sklearn import preprocessing
import warnings
from sklearn.model_selection import cross_val_score
warnings.filterwarnings(action='ignore')

# +
GSPREADHSEET_DOWNLOAD_URL = (
    "https://docs.google.com/spreadsheets/d/{gid}/export?format=csv&id={gid}".format
)

SYSARMY_2020_2_GID = '12ydHqyonOaxXI9q0hJV231HHJG0bDIH3K7-rxidjpJI'
# -

raw_data = pd.read_csv(GSPREADHSEET_DOWNLOAD_URL(gid=SYSARMY_2020_2_GID))

raw_data.head(10)

raw_data.shape

# Este es un conjunto de datos bastante estándar, relativamente pequeño, con 97 observaciones y 9 variables. Aquí definiremos como objetivo y la cantidad de expresión de antígeno que está asociada con la detección de este cáncer (la columna lpsa del conjunto de datos). Las otras variables son características asociadas.
#
# Recuperamos nuestras variables explicativas para la regresión, que colocamos en una matriz X y nuestra variable explicada Y por separado. No recuperamos la última columna del conjunto de datos, que es un valor booleano asociado con la presencia de cáncer, porque no tratamos las variables discretas en este caso.

X_train = raw_data[['lweight','age','lbph','svi','lcp','gleason','pgg45']]
y_train = raw_data[['lpsa']]


def rmse_cv(model, X_train, y_train):
    rmse = np.sqrt(
        -cross_val_score(
            model, X_train, y_train, scoring="neg_mean_squared_error", cv=5
        )
    )
    return rmse.mean()


# # Baseline : regresión lineal clásica

# +
# Recuperamos el MSE en el conjunto de datos de prueba como baseline del error
baseline_error = rmse_cv(LinearRegression(), X_train, y_train)

print(baseline_error)
# -

# # Ridge Regresssion
#
# Antes que nada, escalamos los datos

plt.figure(figsize=[12, 8])
plt.xlabel('Variables', fontsize=20)
sns.set_theme(style="whitegrid")
sns.boxplot(data = raw_data[['lweight','age','lbph','svi','lcp','gleason','pgg45']])

scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
to_scale = raw_data.iloc[:,1:10]
scaler.fit(to_scale)
scaled_df = scaler.transform(to_scale)
names = to_scale.columns
scaled_prostate_df = pd.DataFrame(scaled_df, columns=names)
scaled_prostate_df['train'] = raw_data.iloc[:,10]

plt.figure(figsize=[12, 8])
plt.xlabel('Variables', fontsize=20)
sns.set_theme(style="whitegrid")
sns.boxplot(data = scaled_prostate_df[['lweight','age','lbph','svi','lcp','gleason','pgg45']])

X_train = scaled_prostate_df.iloc[:,:][['lweight','age','lbph','svi','lcp','gleason','pgg45']]
y_train = raw_data.iloc[:]['lpsa']


# Llamamos alfa al coeficiente de regularización. 
# Vamos a probar varios de ellos para determinar cuál es el más óptimo.

n_alphas = 200
alphas = np.logspace(-5, 5, n_alphas)

# Ahora podemos probar todas las regresiones estriadas con los diferentes valores del hiperparámetro α. Recuperamos las ponderaciones de los distintos coeficientes de la regresión asociada así como el error cuadrático.

# +
ridge = linear_model.Ridge()

coefs = []
errors = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)
    errors.append([baseline_error, rmse_cv(Ridge(alpha=a), X_train, y_train)])
# -

# Puede visualizar la evolución del valor de los diferentes pesos asociados a los parámetros:

# +
plt.figure(figsize=[12, 8])
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')

plt.xlabel('alpha', fontsize=20)
plt.ylabel('pesos', fontsize=20)
plt.title('Coeficientes de la regresión Ridge en función del alpha', fontsize=26)
plt.axis('tight')
plt.show()
# -

# El valor de alfa disminuye los pesos de todos los parámetros de la regresión. Ahora estudiemos el valor del error cuadrático:

# +

plt.figure(figsize=[12, 8])
ax = plt.gca()

[baseError,RidgeError] = ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha', fontsize=20)
plt.ylabel('error', fontsize=20)
plt.axis('tight')
leg2 = ax.legend([baseError,RidgeError],['Baseline','Ridge'], loc='lower right')
ax.add_artist(leg2)
plt.show()
# -

# Como podemos ver, la regularización reduce el error en el conjunto de datos de prueba. Hacia alfa = 10, el mínimo parece encontrarse para la regresión Ridge. Podemos recuperar el valor mínimo:

min(errors)

# El primer valor corresponde al error con regresión lineal clásica y el segundo valor corresponde al error con regresión Ridge.

# # Regresión Lasso

# También probamos distintos valores de alpha Lasso

# +
n_alphas = 300
alphas = np.logspace(-5, 1, n_alphas)
lasso = linear_model.Lasso(fit_intercept=False)

coefs = []
errors = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    errors.append([baseline_error,rmse_cv(linear_model.Lasso(alpha=a), X_train, y_train)])

# +
plt.figure(figsize=[12, 8])
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha', fontsize=20)
plt.ylabel('weights', fontsize=20)
plt.axis('tight')

plt.show()
# -

# Como podemos ver, Lasso te permite eliminar variables estableciendo su peso en cero. Este es el caso si dos variables están correlacionadas. Uno será seleccionado por Lasso y el otro borrado. Esta es también su ventaja sobre una regresión Ridge que no seleccionará variables.
#
# Ahora podemos observar el comportamiento del error.

# +
plt.figure(figsize=[12, 8])
ax = plt.gca()
[baseError,LassoError] = ax.plot(alphas, errors)

ax.set_xscale('log')
plt.xlabel('alpha', fontsize=20)
plt.ylabel('error', fontsize=20)
plt.axis('tight')
leg2 = ax.legend([baseError,LassoError],['Baseline','Lasso'], loc='lower right')
ax.add_artist(leg2)
plt.show()
# -

min(errors)

# En este caso Ridge funcionó mejor!
#
# Veamos ahora ElasticNet

# # Elastic Net

# +
n_alphas = 300
alphas = np.logspace(-5, 1, n_alphas)


coefs = []
errors = []
for a in alphas:
    elasticNet = ElasticNet(alpha=a, l1_ratio=0.5)
    elasticNet.fit(X_train, y_train)
    coefs.append(elasticNet.coef_)
    errors.append([baseline_error,rmse_cv(ElasticNet(alpha=a, l1_ratio=0.5), X_train, y_train)])

# +
plt.figure(figsize=[12, 8])
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha', fontsize=20)
plt.ylabel('weights', fontsize=20)
plt.axis('tight')
plt.show()
# -

# ¿Por qué Elastic Net también a la larga anula (aunque mas tarde) los coeficientes?

# +
plt.figure(figsize=[12, 8])
ax = plt.gca()
[baseError,ElasticNetError] = ax.plot(alphas, errors)

ax.set_xscale('log')
plt.xlabel('alpha', fontsize=20)
plt.ylabel('error', fontsize=20)
plt.axis('tight')
leg2 = ax.legend([baseError,ElasticNetError],['Baseline','ElasticNet'], loc='lower right')
ax.add_artist(leg2)
plt.show()
# -

min(errors)

# Si bien Elastic Net mejora respecto a la regresión clásica, la competencia la ganó Ridge!
#
# ¿Por qué puede ser esto?
