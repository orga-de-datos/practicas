#!/usr/bin/env python
# coding: utf-8

# # Regresión Lineal en Python

# En este notebook vamos a aprender a implementar regresiones utilizando Scikit-Learn en Python.
# Para esto vamos a utilizar un dataset compuesto por características de distintos vinos. La idea es predecir la calidad del vino en base a diferentes factores. La calidad se determina por un puntaje de 0 a 10. El dataset puede encontrarse en este [link](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009).

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as seabornInstance
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures

# In[2]:


dataset = pd.read_csv('../datasets/calidad_vino.csv')


# In[3]:


dataset.shape


# In[4]:


dataset.describe()


# In[5]:


dataset


# Analizamos la correlación de las variables de a pares

# In[6]:


seabornInstance.pairplot(dataset)


# In[7]:


dataset.plot(x='density', y='fixed acidity', style='o')
plt.title('Acidez vs Densidad')
plt.xlabel('density')
plt.ylabel('fixed acidity')
plt.show()


# In[8]:


plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.distplot(dataset['density'])


# Determinamos los _atributos_ y los _labels_. Los atributos son las variables independientes, mientras que los labels son las variables que queremos determinar.
# Como queremos determinar el Ataque Especial en base a la Defensa Especial, el Ataque es nuestra variable X y la Defensa es nuestro label Y.
#
# #### Y = AX + B

# In[9]:


X = dataset['density'].values.reshape(-1, 1)
Y = dataset['fixed acidity'].values.reshape(-1, 1)


# In[10]:


regressor = LinearRegression()
regressor.fit(X, Y)  # Entrenamos el algoritmo


# La regresión lineal nos determina el valor de los parámetros **A** y **B**. **A** representa la pendiente de la recta y **B** la ordenada al origen.

# In[11]:


# Imprimimos el valor de A
print(regressor.coef_)


# In[12]:


# Imprimimos el valor de B
print(regressor.intercept_)


# #### Significado de los coeficientes estimados
#
# Teóricamente, el valor de la <b>ordenada al origen</b>, es decir, 29.30889242 es el valor de Ataque Especial de un pokemon con Defensa Especial cero, por lo que su interpretación individual no tiene sentido. La <b>pendiente</b> de la recta estimada es 0.59224608, es decir, que por cada aumento de un punto en la tasa de Defensa Especial, el Ataque Especial sube 0.59 puntos <b>en promedio</b>.

# #### Predicciones
#
# Ahora que entrenamos el algoritmo, es hora de realizar nuestra predicción

# In[13]:


Y_pred = regressor.predict(X)


# In[14]:


plt.scatter(X, Y, color='gray')
plt.plot(X, Y_pred, color='red', linewidth=2)
plt.show()


# ### Errores
#
# Calculamos el error del modelo.
#
# 1) Error Cuadrático Medio (Mean Squared Error)

# In[15]:


dataset = pd.DataFrame({'Actual': Y.flatten(), 'Predicted': Y_pred.flatten()})
dataset


# In[16]:


print("MSE: " + str(metrics.mean_squared_error(Y, Y_pred, squared=True)))


# In[17]:


print("RMSE: " + str(metrics.mean_squared_error(Y, Y_pred, squared=False)))


# ### Gráfico de Residuos vs Predichos

# In[18]:


plt.rcParams['figure.figsize'] = (10, 5)

preds = pd.DataFrame({"Predicciones": Y_pred.flatten(), "true": Y.flatten()})
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter")


# ### Regresión Lineal Múltiple
#

# In[19]:


dataset = pd.read_csv('../datasets/calidad_vino.csv')


# In[20]:


dataset.shape


# In[21]:


dataset.describe()


# In[22]:


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

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Entrenamos el modelo.

# In[24]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Como en este caso tenemos múltiples variables, la regresión debe encontrar los coeficientes óptimos para cada atributo.

# In[25]:


# Visualizamos los coeficientes determinados por nuestro modelo
coeff_df = pd.DataFrame(regressor.coef_, X_dataset.columns, columns=['Coeficiente'])
coeff_df


# Estos resultados nos indican, por ejemplo, que al incrementar 1 unidad de densidad (density) se disminuye en 31.52 unidades la calidad del vino.

# In[26]:


# Ahora realizamos nuestra predicción de calidad del vino
y_pred = regressor.predict(X_test)


# In[27]:


# Observamos la diferencia entre lo predicho y los valores reales
df = pd.DataFrame({'Actual': y_test, 'Predicción': y_pred})
df1 = df.head(25)
df1


# In[28]:


# Analizamos el error de nuestro modelo
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Gráfico de Residuos vs Predichos

# In[29]:


plt.rcParams['figure.figsize'] = (10, 5)

preds = pd.DataFrame({"Predicciones": y_pred, "true": y_test})
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter")


# ### Regresión Polinomial

# Lo que primero debemos hacer es crear nuestras nuevas variables polinomiales. Vamos a regresar a nuestro dataset de Pokemon, en este caso como sólo tenemos una variable explicativa (sp_defense) crearemos X_2 = sp_defense<sup>2</sup>

# In[30]:


polynomial_features = PolynomialFeatures(degree=2)

df = pd.read_csv('../datasets/calidad_vino.csv')
df = df.sort_values(by=['density'])
x_poly = polynomial_features.fit_transform(df['density'].values.reshape(-1, 1))

x_poly


# Ahora volvemos a entrenar nuestro modelo lineal pero utilizando esta nueva variable

# In[31]:


polymodel = LinearRegression()
polymodel.fit(x_poly, df['fixed acidity'].values.reshape(-1, 1))


# In[35]:


density_pred = polymodel.predict(x_poly)
fig = plt.figure(figsize=(12, 6))
plt.scatter(df['density'], df['fixed acidity'])
plt.plot(df['density'], density_pred, color='red')
plt.xlabel('Densidad')
plt.ylabel('Acidez')
plt.show()


# Ahora analicemos qué error tenemos en este caso.

# In[36]:


print(
    "MSE: "
    + str(metrics.mean_squared_error(df['fixed acidity'], density_pred, squared=True))
)


# In[38]:


print(
    "RMSE: "
    + str(metrics.mean_squared_error(df['fixed acidity'], density_pred, squared=False))
)


# El error decreció, lo cual es bueno, veamos ahora el grafico de residuos para analizar si los supuestos se ajustan mejor

# In[39]:


plt.rcParams['figure.figsize'] = (10, 5)

preds = pd.DataFrame(
    {"Predicciones": density_pred.reshape(1599), "true": df['fixed acidity']}
)
preds["Residuos"] = preds["true"] - preds["Predicciones"]
preds.plot(x="Predicciones", y="Residuos", kind="scatter")


# Ahora probamos con un polinomio de grado 3

# In[40]:


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


# #### ¿Que sucede si aumentamos mucho el grado del polinomio?
#
# Recordemos que para tener medidas realistas de nuestros modelos es necesario, evaluar el error sobre <b>datos no usados en el entrenamiento</b>.
#
# Así que utilizaremos de nuevo k-fold cross validation, para lograr medidas mas realistas de cada tipo de regresión.
#

# In[41]:


def rmse_cv(model, X_train, y_train):
    rmse = np.sqrt(
        -cross_val_score(
            model, X_train, y_train, scoring="neg_mean_squared_error", cv=5
        )
    )
    return rmse.mean()


# In[44]:


errors = []
for i in range(1, 11):
    polynomial_features = PolynomialFeatures(degree=i)
    x_poly = polynomial_features.fit_transform(
        df['fixed acidity'].values.reshape(-1, 1)
    )
    y = df['density'].values.reshape(-1, 1)
    regressions = LinearRegression()
    errors.append(rmse_cv(regressions, x_poly, y))


# In[45]:


errores = pd.DataFrame({"grado": range(1, 11), "error": errors[0:10]})
errores.plot(x="grado", y="error")


# In[46]:


errors


# In[ ]:
