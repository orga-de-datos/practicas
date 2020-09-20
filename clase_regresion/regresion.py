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

# # Regresión Lineal en Python

# En este notebook vamos a aprender a implementar regresiones utilizando Scikit-Learn en Python.
# Para esto vamos a utilizar un dataset compuesto por características de distintos vinos. La idea es predecir la calidad del vino en base a diferentes factores. La calidad se determina por un puntaje de 0 a 10. El dataset puede encontrarse en este [link](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009).

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as seabornInstance
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split

# +
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('../datasets/calidad_vino.csv')

dataset.shape

dataset.describe()

dataset

# Analizamos la correlación de las variables de a pares

seabornInstance.pairplot(dataset)

dataset.plot(x='density', y='fixed acidity', style='o')
plt.title('Acidez vs Densidad')
plt.xlabel('density')
plt.ylabel('fixed acidity')
plt.show()

plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.distplot(dataset['density'])

# Determinamos los _atributos_ y los _labels_. Los atributos son las variables independientes, mientras que los labels son las variables que queremos determinar.
# Como queremos determinar la Acidez en base a la Densidad, la Densidad es nuestra variable X y la Acidez es nuestro label Y.
#
# #### Y = AX + B

X = dataset['density'].values.reshape(-1, 1)
Y = dataset['fixed acidity'].values.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(X, Y)  # Entrenamos el algoritmo

# La regresión lineal nos determina el valor de los parámetros **A** y **B**. **A** representa la pendiente de la recta y **B** la ordenada al origen.

# Imprimimos el valor de A
print(regressor.coef_)

# Imprimimos el valor de B
print(regressor.intercept_)

# #### Significado de los coeficientes estimados
#
# Teóricamente, el valor de la <b>ordenada al origen</b>, es decir, -605.95990133 es el valor de Acidez de un vino con Densidad cero, por lo que su interpretación individual no tiene sentido. La <b>pendiente</b> de la recta estimada es 616.28450984, es decir, que por cada aumento de un punto en la tasa de Densidad, la Acidez sube 616 puntos <b>en promedio</b>.

# #### Predicciones
#
# Ahora que entrenamos el algoritmo, es hora de realizar nuestra predicción

Y_pred = regressor.predict(X)

plt.scatter(X, Y, color='gray')
plt.plot(X, Y_pred, color='red', linewidth=2)
plt.show()

# ### Errores
#
# Calculamos el error del modelo

dataset = pd.DataFrame({'Actual': Y.flatten(), 'Predicted': Y_pred.flatten()})
dataset

# 1) Error Cuadrático Medio (Mean Squared Error)

print("MSE: " + str(metrics.mean_squared_error(Y, Y_pred, squared=True)))

# 2) Raíz del Error Cuadrático Medio (Root Mean Squared Error)

print("RMSE: " + str(metrics.mean_squared_error(Y, Y_pred, squared=False)))

# ### Gráfico de Residuos vs Predichos

# +
plt.rcParams['figure.figsize'] = (10, 5)

preds = pd.DataFrame({"Predicciones": Y_pred.flatten(), "true": Y.flatten()})
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter")
# -

# ### Regresión Lineal Múltiple
#

dataset = pd.read_csv('../datasets/calidad_vino.csv')

dataset.shape

dataset.describe()

# Dividimos los datos en atributos y labels
X_dataset = dataset[
    [
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol',
    ]
]
X = dataset[
    [
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol',
    ]
].values
y = dataset['quality'].values

# Separamos nuestros datos en set de entrenamiento (80%) y set de test (20%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entrenamos el modelo.

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Como en este caso tenemos múltiples variables, la regresión debe encontrar los coeficientes óptimos para cada atributo.

# Visualizamos los coeficientes determinados por nuestro modelo
coeff_df = pd.DataFrame(regressor.coef_, X_dataset.columns, columns=['Coeficiente'])
coeff_df

# Estos resultados nos indican, por ejemplo, que al incrementar 1 unidad de densidad (density) se disminuye en 31.52 unidades la calidad del vino.

# Ahora realizamos nuestra predicción de calidad del vino
y_pred = regressor.predict(X_test)

# Observamos la diferencia entre lo predicho y los valores reales
df = pd.DataFrame({'Actual': y_test, 'Predicción': y_pred})
df1 = df.head(25)
df1

# Analizamos el error de nuestro modelo
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# ### Gráfico de Residuos vs Predichos

# +
plt.rcParams['figure.figsize'] = (10, 5)

preds = pd.DataFrame({"Predicciones": y_pred, "true": y_test})
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter")
# -

# Vemos que no es una nube de puntos sin estructura, sino que existen ciertos patrones. Esto nos dice que existen correlaciones entre residuos y predichos.

# ### Regresión Polinomial

# Lo que primero debemos hacer es crear nuestras nuevas variables polinomiales. Vamos a crear X_2 = density<sup>2</sup>

polynomial_features = PolynomialFeatures(degree=2)

df = pd.read_csv('../datasets/calidad_vino.csv')
df = df.sort_values(by=['density'])
x_poly = polynomial_features.fit_transform(df['density'].values.reshape(-1, 1))

x_poly
# -

# Ahora volvemos a entrenar nuestro modelo lineal pero utilizando esta nueva variable

polymodel = LinearRegression()
polymodel.fit(x_poly, df['fixed acidity'].values.reshape(-1, 1))

density_pred = polymodel.predict(x_poly)
fig = plt.figure(figsize=(12, 6))
plt.scatter(df['density'], df['fixed acidity'])
plt.plot(df['density'], density_pred, color='red')
plt.xlabel('Densidad')
plt.ylabel('Acidez')
plt.show()

# Ahora analicemos qué error tenemos en este caso.

print(
    "MSE: "
    + str(metrics.mean_squared_error(df['fixed acidity'], density_pred, squared=True))
)

print(
    "RMSE: "
    + str(metrics.mean_squared_error(df['fixed acidity'], density_pred, squared=False))
)

# El error decreció, lo cual es bueno, veamos ahora el grafico de residuos para analizar si los supuestos se ajustan mejor

# +
plt.rcParams['figure.figsize'] = (10, 5)

preds = pd.DataFrame(
    {"Predicciones": density_pred.reshape(1599), "true": df['fixed acidity']}
)
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter")
# -

# Ahora probamos con un polinomio de grado 3

# +
polynomial_features = PolynomialFeatures(degree=3)

df = df.sort_values(by=['density'])
x_poly_3 = polynomial_features.fit_transform(df['density'].values.reshape(-1, 1))
polymodel_3 = LinearRegression()
polymodel_3.fit(x_poly_3, df['fixed acidity'].values.reshape(-1, 1))
life_pred = polymodel_3.predict(x_poly_3)
fig = plt.figure(figsize=(12, 6))
plt.scatter(df['density'], df['fixed acidity'])
plt.plot(df['density'], life_pred, color='red')
plt.xlabel('Densidad')
plt.ylabel('Acidez')
plt.show()


# -

# #### ¿Que sucede si aumentamos mucho el grado del polinomio?
#
# Recordemos que para tener medidas realistas de nuestros modelos es necesario, evaluar el error sobre <b>datos no usados en el entrenamiento</b>.
#
# Así que utilizaremos de nuevo k-fold cross validation, para lograr medidas mas realistas de cada tipo de regresión.
#


def rmse_cv(model, X_train, y_train):
    rmse = np.sqrt(
        -cross_val_score(
            model, X_train, y_train, scoring="neg_mean_squared_error", cv=5
        )
    )
    return rmse.mean()


errors = []
for i in range(1, 31):
    polynomial_features = PolynomialFeatures(degree=i)
    x_poly = polynomial_features.fit_transform(
        df['fixed acidity'].values.reshape(-1, 1)
    )
    y = df['density'].values.reshape(-1, 1)
    regressions = LinearRegression()
    errors.append(rmse_cv(regressions, x_poly, y))

errores = pd.DataFrame({"grado": range(1, 31), "error": errors[0:30]})
errores.plot(x="grado", y="error")

errors
