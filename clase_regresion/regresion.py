#!/usr/bin/env python
# coding: utf-8
# %%

# # Regresión Lineal en Python

# En este notebook vamos a aprender a implementar regresiones utilizando Scikit-Learn en Python.
# Para esto vamos a utilizar un dataset compuesto por características de distintos vinos. La idea es predecir la calidad del vino en base a diferentes factores. La calidad se determina por un puntaje de 0 a 10. El dataset puede encontrarse en este [link](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009).

# %%


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as seabornInstance
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

get_ipython().run_line_magic('matplotlib', 'inline')


# %%


# dataset = pd.read_csv('../datasets/pokemon.csv', index_col = 'name')
dataset = pd.read_csv('../datasets/calidad_vino.csv')


# %%


dataset.shape


# %%


dataset.describe()


# %%


dataset


# Analizamos la correlación de las variables de a pares

# %%


# seabornInstance.pairplot(dataset[['sp_attack','sp_defense','speed','weight_kg']])
seabornInstance.pairplot(dataset)


# %%


# dataset.plot(x='sp_defense', y='sp_attack', style='o')
# plt.title('Defensa Especial vs Ataque Especial')
# plt.xlabel('sp_defense')
# plt.ylabel('sp_attack')
# plt.show()

dataset.plot(x='density', y='fixed acidity', style='o')
plt.title('Defensa Especial vs Ataque Especial')
plt.xlabel('density')
plt.ylabel('fixed acidity')
plt.show()


# %%


plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.distplot(dataset['sp_attack'])


# Determinamos los _atributos_ y los _labels_. Los atributos son las variables independientes, mientras que los labels son las variables que queremos determinar.
# Como queremos determinar el Ataque Especial en base a la Defensa Especial, el Ataque es nuestra variable X y la Defensa es nuestro label Y.
#
# #### Y = AX + B

# %%


# X = dataset['sp_defense'].values.reshape(-1,1)
# Y = dataset['sp_attack'].values.reshape(-1,1)
X = dataset['density'].values.reshape(-1, 1)
Y = dataset['fixed acidity'].values.reshape(-1, 1)


# %%


regressor = LinearRegression()
regressor.fit(X, Y)  # Entrenamos el algoritmo


# La regresión lineal nos determina el valor de los parámetros **A** y **B**. **A** representa la pendiente de la recta y **B** la ordenada al origen.

# %%


# Imprimimos el valor de A
print(regressor.coef_)


# %%


# Imprimimos el valor de B
print(regressor.intercept_)


# #### Significado de los coeficientes estimados
#
# Teóricamente, el valor de la <b>ordenada al origen</b>, es decir, 29.30889242 es el valor de Ataque Especial de un pokemon con Defensa Especial cero, por lo que su interpretación individual no tiene sentido. La <b>pendiente</b> de la recta estimada es 0.59224608, es decir, que por cada aumento de un punto en la tasa de Defensa Especial, el Ataque Especial sube 0.59 puntos <b>en promedio</b>.

# #### Predicciones
#
# Ahora que entrenamos el algoritmo, es hora de realizar nuestra predicción

# %%


Y_pred = regressor.predict(X)


# %%


plt.scatter(X, Y, color='gray')
plt.plot(X, Y_pred, color='red', linewidth=2)
plt.show()


# ### Errores
#
# Calculamos el error del modelo.
#
# 1) Error Cuadrático Medio (Mean Squared Error)

# %%


dataset = pd.DataFrame({'Actual': Y.flatten(), 'Predicted': Y_pred.flatten()})
dataset


# %%


print("MSE: " + str(metrics.mean_squared_error(Y, Y_pred, squared=True)))


# %%


print("RMSE: " + str(metrics.mean_squared_error(Y, Y_pred, squared=False)))


# ### Gráfico de Residuos vs Predichos

# %%


plt.rcParams['figure.figsize'] = (10, 5)

preds = pd.DataFrame({"Predicciones": Y_pred.flatten(), "true": Y.flatten()})
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter")


# ### Regresión Lineal Múltiple
#

# %%


dataset = pd.read_csv('../datasets/calidad_vino.csv')


# %%


dataset.shape


# %%


dataset.describe()


# %%


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


# Separamos nuestros datos en set de entrenamiento (80%) y set de test (20%).

# %%


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Entrenamos el modelo.

# %%


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Como en este caso tenemos múltiples variables, la regresión debe encontrar los coeficientes óptimos para cada atributo.

# %%


# Visualizamos los coeficientes determinados por nuestro modelo
coeff_df = pd.DataFrame(regressor.coef_, X_dataset.columns, columns=['Coeficiente'])
coeff_df


# Estos resultados nos indican, por ejemplo, que al incrementar 1 unidad de densidad (density) se disminuye en 31.52 unidades la calidad del vino.

# %%


# Ahora realizamos nuestra predicción de calidad del vino
y_pred = regressor.predict(X_test)


# %%


# Observamos la diferencia entre lo predicho y los valores reales
df = pd.DataFrame({'Actual': y_test, 'Predicción': y_pred})
df1 = df.head(25)
df1


# %%


# Analizamos el error de nuestro modelo
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Gráfico de Residuos vs Predichos

# %%


plt.rcParams['figure.figsize'] = (10, 5)

preds = pd.DataFrame({"Predicciones": y_pred, "true": y_test})
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter")


# ### Regresión Polinomial

# Lo que primero debemos hacer es crear nuestras nuevas variables polinomiales. Vamos a regresar a nuestro dataset de Pokemon, en este caso como sólo tenemos una variable explicativa (sp_defense) crearemos X_2 = sp_defense<sup>2</sup>

# %%


polynomial_features = PolynomialFeatures(degree=2)

# df = pd.read_csv('../datasets/pokemon.csv', index_col = 'name')
# df = df.sort_values(by=['sp_defense'])
# x_poly = polynomial_features.fit_transform(df['sp_defense'].values.reshape(-1, 1))

df = pd.read_csv('../datasets/calidad_vino.csv')
df = df.sort_values(by=['density'])
x_poly = polynomial_features.fit_transform(df['density'].values.reshape(-1, 1))

x_poly


# Ahora volvemos a entrenar nuestro modelo lineal pero utilizando esta nueva variable

# %%


polymodel = LinearRegression()
# polymodel.fit(x_poly, df['sp_attack'].values.reshape(-1, 1))
polymodel.fit(x_poly, df['fixed acidity'].values.reshape(-1, 1))


# %%


attack_pred = polymodel.predict(x_poly)
fig = plt.figure(figsize=(12, 6))
# plt.scatter(df['sp_defense'], df['sp_attack'])
# plt.plot(df['sp_defense'],attack_pred , color='red')
# plt.xlabel('Defensa Especial')
# plt.ylabel('Ataque Especial');
plt.scatter(df['density'], df['fixed acidity'])
plt.plot(df['density'], attack_pred, color='red')
plt.xlabel('Densidad')
plt.ylabel('Acidez')
plt.show()


# Ahora analicemos qué error tenemos en este caso.

# %%


# print("MSE: "+str(mean_squared_error(df['sp_attack'], attack_pred, squared=True)))
print("MSE: " + str(mean_squared_error(df['fixed acidity'], attack_pred, squared=True)))


# %%


# print("RMSE: "+str(mean_squared_error(df['sp_attack'], attack_pred, squared=False)))
print(
    "RMSE: " + str(mean_squared_error(df['fixed acidity'], attack_pred, squared=False))
)


# El error decreció, lo cual es bueno, veamos ahora el grafico de residuos para analizar si los supuestos se ajustan mejor

# %%


plt.rcParams['figure.figsize'] = (10, 5)

# preds = pd.DataFrame({"Predicciones":attack_pred.reshape(801), "true":df['sp_attack']})
preds = pd.DataFrame(
    {"Predicciones": attack_pred.reshape(1599), "true": df['fixed acidity']}
)
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter")


# Ahora probamos con un polinomio de grado 3

# %%


polynomial_features = PolynomialFeatures(degree=3)

# df = df.sort_values(by=['sp_defense'])
# x_poly_3 = polynomial_features.fit_transform(df['sp_defense'].values.reshape(-1, 1))
# polymodel_3 = LinearRegression()
# polymodel_3.fit(x_poly_3, df['sp_attack'].values.reshape(-1, 1))
# life_pred=polymodel_3.predict(x_poly_3)
# fig= plt.figure(figsize=(12,6))
# plt.scatter(df['sp_defense'], df['sp_attack'])
# plt.plot(df['sp_defense'],life_pred , color='red')
# plt.xlabel('Defensa Especial')
# plt.ylabel('Ataque Especial');
# plt.show()

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


# %%
