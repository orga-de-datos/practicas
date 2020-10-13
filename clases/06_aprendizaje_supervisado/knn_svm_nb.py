# +
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np

def get_iris_dataset(normalized=None):
    X, y = load_iris(return_X_y=True)
  
    if normalized == 'zero mean and unit variance':
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X) # preunta alumnos: que estoy haciendo mal aca? (pss leaking)

    if normalized == 'zero mean':
        normalizer = preprocessing.Normalizer()
        X = normalizer.fit_transform(X) # preunta alumnos: que estoy haciendo mal aca? (pss leaking)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    return X_train, X_test, y_train, y_test


# -

dataset = load_iris()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target_id'] = dataset.target
df['target_name'] = df.target_id.apply(lambda x: dataset.target_names[x])
df.head()

# # KNN
# https://scikit-learn.org/stable/modules/neighbors.html

# Ejemplo simple

# +
from sklearn.neighbors import KNeighborsClassifier

X_train, X_val, y_train, y_val = get_iris_dataset()
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)
y_pred
# -

print('correctas: ', np.sum(y_val == y_pred))
print('total: ', len(y_val))

# +
# Agarrar un dataset
# Aplicarle KNN
# ver la performance

X_train, X_test, y_train, y_test = get_iris_dataset()

metrics = []
for n in range(1, len(X_test)):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    metrics.append((n, (y_test == y_pred).sum()))
    # print(n,f"Number of mislabeled points out of a total {X_test.shape[0]} points : {(y_test != y_pred).sum()}")

df_metrics = pd.DataFrame(metrics, columns=['vecinos', 'correctos'])
print('mejor puntaje: ', max(df_metrics.correctos), 'correctos')
df_metrics.plot(x='vecinos', figsize=(15,10))
# -

# ### Que pasa si se elige a n vecinos igual al total de puntos?

pd.Series(y_train).value_counts()

pd.Series(y_pred).value_counts()

pd.Series(y_test).value_counts()

# # Todos los atributos de cada ejemplo pesan lo mismo ??
#
#
# normalizar los datos  
# Aplicarle KNN  
# Ver la performance  
#

# +
X_train, X_test, y_train, y_test = get_iris_dataset(normalized='zero mean and unit variance')

metrics = []
for n in range(1, len(X_test)):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    metrics.append((n, (y_test == y_pred).sum()))

df_metrics_normalized_unit_variance = pd.DataFrame(metrics, columns=['vecinos', 'correctos'])
# df_metrics_normalized_unit_variance.plot(x='vecinos',figsize=(15,10))

# +
X_train, X_test, y_train, y_test = get_iris_dataset(normalized='zero mean')

metrics = []
for n in range(1, len(X_test)):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    metrics.append((n, (y_test == y_pred).sum()))

df_metrics_normalized = pd.DataFrame(metrics, columns=['vecinos', 'correctos'])
# df_metrics_normalized.plot(x='vecinos',figsize=(15,10))
# -

# # Poniendo todo junto

df = df_metrics.merge(df_metrics_normalized, on='vecinos', validate="1:1", suffixes=('', '_normalizado'))
df = df.merge(df_metrics_normalized_unit_variance, on='vecinos',
              suffixes=('', '_normalizado_y_varianza_1'),
              validate="1:1")
df.plot(x='vecinos', figsize=(15,10))

# # Que pasa con la distancia entre los puntos, todos los vecinos valen lo mismo?

# +
# probar que pasa si cambio el parametro weights
# weights: "uniform", "distance"
# -

# ## Como se calcula la distancia?
#

# +
# Explicar por arriba que hay distintas funciones de distancia
# mostrar en sklearn como usarlas
# -

# # Como es el orden computacional de esto?
#
#
# Mostrar que hay fit() lazy e eager, explicar un poco la diff y la eficiencia en cada uno.

# +
# explicar MUY por arriba que es el parametro algorithm
# algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}

# +
# TODO: Ver si tiene sentido (Reduccion de dimensionalidad + KNN)

# Ver si tiene sentido:
# siguiendo esta linea: https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html
# Ver si consigo un dataset con muchas dimensiones (o crear uno)
# Aplicar KNN
# Ver la performance
# Reducir dimensionalidad con PCA
# Aplicar KNN
# Ver la performance
# -







# # Naive Bayes
#
# https://scikit-learn.org/stable/modules/naive_bayes.html

# +
# Explicacion del Laplace/Lidstone) smoothing  (sumarle 1 a todas las clases)

# Cuando trabajamos con features continuos (GaussianNB)
# Cuando trabajamos con features discretos (MultinomialNB)
# Cuando trabajamos con features categoricos (CategoricalNB)

# Poray uso esto para explicar las distintas clases que tiene sklearn
# https://towardsdatascience.com/comparing-a-variety-of-naive-bayes-classification-algorithms-fc5fa298379e

# Ver si se puede: mostrar como mezclar features continuos y discretos y categoricos 
# eso no esta en sklearn ( podria ser un PR si sale? )

# -

# Aplicando Gaussian Naive Bayes
#
# porque gaussian NB para este dataset?

# +
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB( )
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
# -

# ## Aplicando Multinomial Naive Bayes

# +
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = MultinomialNB( )
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

# preunta al alumno: porque anda tan mal ?
# -

# ### Mezcalando valores continuos, discretos y categoricos

# TODO: Ensamble agarramos las probabilidades de GaussianNB, MultinomialNB y CategoricalNB y le metemos un gaussianNB al final.


# +
# df = pd.read_csv('/content/superhero-set/heroes_information.csv', index_col='Unnamed: 0')

# from sklearn import preprocessing

# for category
# le = preprocessing.LabelEncoder()
# le.fit(df.name)
# df['name'] = le.transform(df.name)

# le_gender = preprocessing.LabelEncoder()
# le_gender.fit(df.Gender)
# df['Gender'] = le_gender.transform(df.Gender)

# cols = list(df.columns)
# cols.remove('Alignment')
# X = df[cols]
# y = df[['Alignment']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# -

# Support Vector Machines (SVM)
#
# https://scikit-learn.org/stable/modules/svm.html

# +
# TODO: Aplicar SVM en iris dataset
#       - ver como varia la performance si normalizamos los datos
#       - ver como varia la performance si usamos distintos kernels
#
# TODO: Agarrar los links de abajo y explicarlos
# -

# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py
#
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py
#
# https://scikit-learn.org/stable/auto_examples/exercises/plot_iris_exercise.html#sphx-glr-auto-examples-exercises-plot-iris-exercise-py
#

# +
# Demo interactiva
# https://cs.stanford.edu/~karpathy/svmjs/demo/
# -


