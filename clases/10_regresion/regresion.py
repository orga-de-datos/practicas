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
# Para esto vamos a utilizar un dataset compuesto por características de distintos vinos. El dataset puede encontrarse en este [link](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009).

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as seabornInstance
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split

# +
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv(
    'https://drive.google.com/uc?export=download&id=14G9ZOBJcVVT0d3ozPRNQBOZUr_qL-iXk'
)

dataset.shape
# -

dataset.describe()

dataset

# Analizamos la correlación de las variables de a pares
seabornInstance.pairplot(dataset)

seabornInstance.pairplot(dataset[['fixed acidity', 'density']])

# Nos quedamos con la siguiente correlación
dataset.plot(x='density', y='fixed acidity', style='o')
plt.title('Acidez vs Densidad')
plt.xlabel('Density')
plt.ylabel('Fixed acidity')
plt.show()


# Graficamos la distribución de los valores
plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.histplot(dataset['density'])

# Determinamos los _atributos_ y los _labels_. Los atributos son las variables independientes, mientras que los labels son las variables que queremos determinar.
# Como queremos determinar la Acidez en base a la Densidad, los valores de Densidad son nuestros atributos que se reflejan en la variable X y los valores de Acidez son nuestros labels, representados por la variable Y.
#
# #### $Y = B_{1} * X + B_{0}$

# +
X = dataset['density'].values.reshape(-1, 1)
Y = dataset['fixed acidity'].values.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(X, Y)  # Entrenamos el algoritmo
# -

# La regresión lineal nos determina el valor de los parámetros **B1** y **B0**. **B1** representa la pendiente de la recta y **B0** la ordenada al origen.

# Imprimimos el valor de B1
print(regressor.coef_)

# Imprimimos el valor de B0
print(regressor.intercept_)

# #### Significado de los coeficientes estimados
#
# Teóricamente, el valor de la <b>ordenada al origen</b>, es decir, -605.95990133 es el valor de Acidez de un vino con Densidad cero, por lo que su interpretación individual no tiene sentido. La <b>pendiente</b> de la recta estimada es 616.28450984, es decir, que por cada aumento de un punto en la tasa de Densidad, la Acidez sube 616 puntos <b>en promedio</b>.

# #### Predicciones
#
# Ahora que entrenamos el algoritmo, es hora de realizar nuestra predicción

# +
Y_pred = regressor.predict(X)

plt.scatter(X, Y, color='gray')
plt.plot(X, Y_pred, color='red', linewidth=2)
plt.title('Acidez vs Densidad')
plt.xlabel('Density')
plt.ylabel('Fixed acidity')
plt.show()
# -

# Vemos que el algoritmo resulta en una recta que refleja la relación entre ambas variables.

# ### Errores
#
# Calculamos el error del modelo

dataset = pd.DataFrame({'Actual': Y.flatten(), 'Predicted': Y_pred.flatten()})
dataset

# **1) Error Cuadrático Medio (Mean Squared Error)**
#
# Medida de qué tan cercana es la recta de regresión a los puntos que representan los datos. Mientras más chico más cerca está nuestro modelo de los datos reales. Al ser un valor elevado al cuadrado, es sensible a valores de diferencias grandes.
#
# #### $MSE = \frac1n \sum_{i=1}^n (y - \hat{y} )^2$

print("MSE: " + str(metrics.mean_squared_error(Y, Y_pred, squared=True)))

# **2) Raíz del Error Cuadrático Medio (Root Mean Squared Error)**
#
# Tiene las mismas unidades que los valores representados en el eje vertical. Es la distancia de un punto hasta la recta de regresión, medida en línea recta. Mide el desvío estándar (cuánto se alejan los valores de la media).
#
# #### $RMSE =  \sqrt{\frac1n \sum_{i=1}^n (y - \hat{y} )^2}$

print("RMSE: " + str(metrics.mean_squared_error(Y, Y_pred, squared=False)))

# ### Gráfico de Residuos vs Predichos

# +
plt.rcParams['figure.figsize'] = (10, 5)

preds = pd.DataFrame({"Predicciones": Y_pred.flatten(), "true": Y.flatten()})
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter", title="Residuos vs Predichos")
# -

# Observamos el gráfico de Residuos vs Predichos para ver si tiene forma de "nube sin esctructura". Tiene forma de nube, pero presenta un agrupamiento en el centro lo que le da cierta estructura. Sin embargo, el agrupamiento se presenta en valores cercanos a 0, por lo que nuestro modelo presenta buenos resultados.

# ### Regresión Polinomial

# Lo que primero debemos hacer es crear nuestras nuevas variables polinomiales. Vamos a crear $X_{2}$ = density<sup>2</sup>

# #### Ejemplo

# Polynomial features permite generar nuevas features mediante la multiplicación de las features actuales
polynomial_features = PolynomialFeatures(degree=2)

# +
df = pd.read_csv(
    'https://drive.google.com/uc?export=download&id=14G9ZOBJcVVT0d3ozPRNQBOZUr_qL-iXk'
)
df = df.sort_values(by=['density'])

print("Density: \n")
print(df['density'].values.reshape(-1, 1))
print()

print("Volatile Acidity: \n")
print(df['volatile acidity'].values.reshape(-1, 1))
print()

# Por ejemplo
x_poly_2 = polynomial_features.fit_transform(df[['density', 'volatile acidity']].values)
x_poly_2
# -

# #### Volvemos al caso de estudio

# Para este caso vamos a agarrar la variable density y elevarla al cuadrado
x_poly = polynomial_features.fit_transform(df['density'].values.reshape(-1, 1))
print(x_poly)

# Ahora volvemos a entrenar nuestro modelo lineal pero utilizando esta nueva variable.

polymodel = LinearRegression()
polymodel.fit(x_poly, df['fixed acidity'].values.reshape(-1, 1))
print(polymodel.coef_)
print(polymodel.intercept_)

# Nos quedaría una ecuación con la siguiente estructura, con _density_ representando $X_{2}$:
# #### $Y = B_{3} * X_{2}^{2} + B_{2} * X_{2} + B_{1} * 1 + B_{0}$

# +
acidity_pred = polymodel.predict(x_poly)

fig = plt.figure(figsize=(12, 6))
plt.scatter(df['density'], df['fixed acidity'])
plt.plot(df['density'], acidity_pred, color='red')
plt.title('Acidez vs Densidad')
plt.xlabel('Densidad')
plt.ylabel('Acidez')
plt.show()
# -

# La línea de regresión, al estar representada ahora con una curva, se acopla mejor a los datos.

# Ahora analicemos qué error tenemos en este caso.

print(
    "MSE: "
    + str(metrics.mean_squared_error(df['fixed acidity'], acidity_pred, squared=True))
)

print(
    "RMSE: "
    + str(metrics.mean_squared_error(df['fixed acidity'], acidity_pred, squared=False))
)
len(df.index)

# El error disminuyó, lo cual es bueno, veamos ahora el grafico de residuos para analizar si los supuestos se ajustan mejor

# +
plt.rcParams['figure.figsize'] = (10, 5)

preds = pd.DataFrame(
    {"Predicciones": acidity_pred.reshape(len(df.index)), "true": df['fixed acidity']}
)
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter", title="Residuos vs Predichos")
# -

# El gráfico no logra superar al que obtuvimos con regresión lineal simple. La nube de puntos no está distribuida uniformemente. Muestra una agrupación en el centro, como en el caso de regresión simple, pero también presenta valores más alejados que resultan en una nube menos equilibrada.

# Ahora probamos con un polinomio de grado 3

# +
polynomial_features = PolynomialFeatures(degree=3)

df = df.sort_values(by=['density'])
x_poly_3 = polynomial_features.fit_transform(df['density'].values.reshape(-1, 1))
polymodel_3 = LinearRegression()
polymodel_3.fit(x_poly_3, df['fixed acidity'].values.reshape(-1, 1))
acidity_poly_pred = polymodel_3.predict(x_poly_3)

fig = plt.figure(figsize=(12, 6))
plt.scatter(df['density'], df['fixed acidity'])
plt.plot(df['density'], acidity_poly_pred, color='red')
plt.title('Acidez vs Densidad')
plt.xlabel('Densidad')
plt.ylabel('Acidez')
plt.show()


# -

# La curva se ajusta aún más a los datos.

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


custom_degree = 200
errors = []
for i in range(1, custom_degree + 1):
    polynomial_features = PolynomialFeatures(degree=i)
    x_poly = polynomial_features.fit_transform(
        df['density'].values.reshape(-1, 1)
    )
    y = df['fixed acidity'].values.reshape(-1, 1)
    regressions = LinearRegression()
    errors.append(rmse_cv(regressions, x_poly, y))

errores = pd.DataFrame({"grado": range(1, custom_degree + 1), "RMSE": errors})
errores.plot(x="grado", y="RMSE", xlabel="Grado", ylabel="RMSE", title="RMSE según el grado del polinomio")

# Aumentar el grado del polinomio no es sinónimo de mayor precisión en los resultados. 

# #### ¿Qué pasa si probamos polinomios de mayor grado en el set de entrenamiento?

# +
errors_2 = []
    
for i in range(1, custom_degree + 1):
    polynomial_features = PolynomialFeatures(degree=i)
    x_poly = polynomial_features.fit_transform(
        df['density'].values.reshape(-1, 1)
    )
    y = df['fixed acidity'].values.reshape(-1, 1)
    regressions = LinearRegression()
    regressions.fit(x_poly, y)
    errors_2.append(metrics.mean_squared_error(y, regressions.predict(x_poly), squared=False))

# -

errores_2 = pd.DataFrame({"grado": range(1, custom_degree + 1), "RMSE": errors_2[0:custom_degree + 1]})
errores_2.plot(x="grado", y="RMSE", xlabel="Grado", ylabel="RMSE", title="RMSE según el grado del polinomio")

# +
polynomial_features = PolynomialFeatures(degree=custom_degree)

df = df.sort_values(by=['density'])
x_poly_train = polynomial_features.fit_transform(
        df['density'].values.reshape(-1, 1)
    )
polymodel_train = LinearRegression()
polymodel_train.fit(x_poly_train, df['fixed acidity'].values.reshape(-1, 1))
acidity_poly_pred = polymodel_train.predict(x_poly_train)

fig = plt.figure(figsize=(12, 6))
plt.scatter(df['density'].values, df['fixed acidity'].values)
plt.plot(df['density'].values, acidity_poly_pred, color='red')
plt.xlabel('Densidad')
plt.ylabel('Acidez')
plt.title('Acidez vs Densidad')
plt.show()

# +
df = df.sort_values(by=['density'])
x_poly_train = polynomial_features.fit_transform(
        df['density'].values.reshape(-1, 1)
    )
polymodel_train = LinearRegression()
polymodel_train.fit(x_poly_train, df['fixed acidity'].values.reshape(-1, 1))
acidity_poly_pred = polymodel_train.predict(x_poly_train)

fig = plt.figure(figsize=(12, 6))
plt.scatter(df['density'].values[600:800], df['fixed acidity'].values[600:800])
plt.plot(df['density'].values[600:800], acidity_poly_pred[600:800], color='red')
plt.xlabel('Densidad')
plt.ylabel('Acidez')
plt.title('Acidez vs Densidad Zoom')
plt.show()
# -

# ### Regresión Lineal Múltiple
#

# Ahora vamos a armar una regresión con más de una variable. Para esto, vamos a predecir la calidad del vino a través de varias variables que nos brinda el dataset original. La calidad se determina por un puntaje de 0 a 10.

dataset = pd.read_csv(
    'https://drive.google.com/uc?export=download&id=14G9ZOBJcVVT0d3ozPRNQBOZUr_qL-iXk'
)

dataset.shape

dataset.head()

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
X = X_dataset.values
y = dataset['quality'].values

# Separamos nuestros datos en set de entrenamiento (80%) y set de test (holdout) (20%)

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Entrenamos el modelo.

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Como en este caso tenemos múltiples variables, la regresión debe encontrar los coeficientes óptimos para cada atributo.

# Visualizamos los coeficientes determinados por nuestro modelo
coeff_df = pd.DataFrame(regressor.coef_, X_dataset.columns, columns=['Coeficiente'])
coeff_df

# Estos resultados nos indican, por ejemplo, que al incrementar 1 unidad de densidad (density) se disminuye en 31.52 unidades la calidad del vino.

# Ahora realizamos nuestra predicción de calidad del vino
y_pred = regressor.predict(X_holdout)

# Observamos la diferencia entre lo predicho y los valores reales
df = pd.DataFrame({'Actual': y_holdout, 'Predicción': y_pred})
df1 = df.head(25)
df1

# Analizamos el error de nuestro modelo
print('MSE:', metrics.mean_squared_error(y_holdout, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_holdout, y_pred)))

# ### Gráfico de Residuos vs Predichos

# +
plt.rcParams['figure.figsize'] = (10, 5)

preds = pd.DataFrame({"Predicciones": y_pred, "true": y_holdout})
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter", title="Residuos vs Predichos")
# -

# Vemos que no es una nube de puntos sin estructura, sino que existen ciertos patrones. Esto nos dice que existen correlaciones entre residuos y predichos. Tal vez existe un mejor modelo para predecir la variable target que la regresión lineal múltiple que utilizamos.
