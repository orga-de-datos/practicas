# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Interpretabildad usando LIME
#
# Primero instalamos la librería para lime

# !pip3 install lime

import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from __future__ import print_function

# ## Obteniendo datos y entrenando un clasificador
# Para este tutorial, usaremos el conjunto de datos de [20 grupos de noticias](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html). En particular, por simplicidad, usaremos un subconjunto de 2 clases: ateísmo y cristianismo.
#
# El conjunto de datos de "20 grupos de noticias" comprende alrededor de 18000 publicaciones de grupos de noticias sobre 20 temas divididos en dos subconjuntos: uno para train y el otro para test. La división entre el train y el test se basa en mensajes publicados antes y después de una fecha específica. Es un dataset clásico para el desarrollo y experimentación de modelos de clasificación o clustering de texto.

from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

# Como embbedings usaremos TD-IDF

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

# Ahora, digamos que queremos usar Random Forest para la clasificación. Por lo general, es difícil entender qué están haciendo estos modelos, especialmente con muchos árboles.

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)

# Veamos que performance obtuvo sobre los datos de test

pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')

# Vemos que este clasificador logra un F score muy nueno. La guía de sklearn para este dataset indica que Multinomial Naive Bayes sobreajusta este conjunto de datos al aprender cosas irrelevantes, como encabezados. Veamos si los Random Forest hacen lo mismo.
#
# ## Explicando predicciones usando cal
# Los explicadores de Lime asumen que los clasificadores actúan sobre texto sin formato, pero los clasificadores sklearn actúan sobre la representación vectorizada de textos. Para ejecutar de forma secuencial el vectorizado y predicción usaremos make_pipeline de sklearn

from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)

print(c.predict_proba([newsgroups_test.data[0]]))

# Ahora creamos un objeto explicativo. Pasamos class_names como un argumento para una mejor visualización

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)

# Luego generamos una explicación con un máximo de 6 características para un documento arbitrario en el conjunto de prueba.

idx = 83
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
print('Id del documento: %d' % idx)
print('Probabilidad(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
print('True class: %s' % class_names[newsgroups_test.target[idx]])

# El clasificador acertó en este ejemplo (predijo ateísmo).
#
# La explicación se presenta a continuación como una lista de features ponderados.

exp.as_list()


# Estos features ponderadas son un modelo lineal, que se aproxima al comportamiento del clasificador de bosque aleatorio en las proximidades del ejemplo de prueba. Aproximadamente, si eliminamos 'Posting' y 'Host' del documento, la predicción debería moverse hacia la clase opuesta (cristianismo) en aproximadamente 0,27 (la suma de las ponderaciones de ambas características). Veamos si este es el caso.

print('Predicción original para la clase cristianismo:', rf.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['Posting']] = 0
tmp[0,vectorizer.vocabulary_['Host']] = 0
print('Prediction luego de sacar Posting y Host:', rf.predict_proba(tmp)[0,1])
print('Diferencia:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])

# ¡Muy cerca!
#
# Las palabras que explican el modelo en torno a este documento parecen muy arbitrarias, no tienen mucho que ver ni con el cristianismo ni con el ateísmo.
# De hecho, estas son palabras que aparecen en los encabezados de los correos electrónicos (se verá claramente pronto).
#
# ## Visualizando explicaciones
# Las explicaciones se pueden devolver como un barplot de matplotlib:

# %matplotlib inline
fig = exp.as_pyplot_figure()

# Las explicaciones también se pueden exportar como una página html (que podemos renderizar en esta notebook), usando D3.js para renderizar gráficos.

exp.show_in_notebook(text=False)

# Alternativamente, podemos guardar la página html completamente contenida en un archivo:

exp.save_to_file('/tmp/oi.html')

# Finalmente, también podemos incluir una visualización del documento original, con las palabras en las explicaciones resaltadas. Observen cómo las palabras que más afectan al clasificador están todas en el encabezado del correo electrónico.

exp.show_in_notebook(text=True)


