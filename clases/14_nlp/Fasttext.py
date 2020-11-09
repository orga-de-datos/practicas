# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3 (venv)
#     language: python
#     name: python3
# ---

# # Fasttext
#
# Para calcular vectores de palabras (embeddings), se necesita un gran corpus de texto. Dependiendo del corpus, los vectores de palabras capturarán información diferente.
#
# En este tutorial, nos centramos en los artículos de Wikipedia.
#
#
# ## Data
# La descarga del corpus de Wikipedia lleva algún tiempo. Entonces, limitaremos nuestro estudio a los primeros mil millones de bytes de Wikipedia en inglés. Se pueden encontrar en el sitio web de Matt Mahoney y descargar mediante el siguiente código:

# ```bash
# # mkdir data
# wget -c http://mattmahoney.net/dc/enwik9.zip -P data
# unzip data/enwik9.zip -d data
# ```

# Los datos de Wikipedia contienen una gran cantidad de datos HTML / XML. Los preprocesamos con el script wikifil.pl incluido en el [github](https://github.com/facebookresearch/fastText) fastText

# ```bash
# perl wikifil.pl data/enwik9 > data/fil9
# ```

# Podemos verificar el archivo ejecutando el siguiente comando:

# ```bash
# $ head -c 80 data/fil9
# anarchism originated as a term of abuse first used against early working class
# ```

# Podemos ver que el texto está bien preprocesado y se puede utilizar para aprender nuestros vectores de palabras.

# ## El Modelo
# El aprendizaje de vectores de palabras sobre estos datos ahora se puede lograr con un solo comando:

import fasttext

model = fasttext.train_unsupervised('data/fil9')

# Tener en cuenta que esto puede demorar una hora o mas dado que la versión en python es mas lenta y son muchisimos datos.
#
# Una vez que finalizado el entrenamiento, tenemos nuestro modelo listo para poder usar para realizar consultas.

model.words

# Devuelve todas las palabras del vocabulario, ordenadas por frecuencia decreciente. Podemos obtener tamibén la palabra vector mediante la siguiente línea

model.get_word_vector("the")

# Si lo deseamos podemos guardar este modelo en el disco como un archivo binario

model.save_model("wiki.bin")

# y recargarlo cuando queramos en vze de entrenar nuevamente:

model = fasttext.load_model("wiki.bin")

# ## Jugando con los parámetros
#
# Hasta ahora, ejecutamos fastText con los parámetros predeterminados, pero dependiendo de los datos, estos parámetros pueden no ser óptimos. Como explicamos en al teórica, algunos de estos parámetros son importantes y tienen una considerable influencia en nuestro modelo final.
#
# Los parámetros más importantes del modelo son su dimensión y el rango de tamaño de las subpalabras. La dimensión (dim) controla el tamaño de los vectores, cuanto más grandes son, más información pueden capturar, pero requiere que se aprendan más datos. Pero, si son demasiado grandes, son más difíciles y lentos de entrenar. De forma predeterminada, usamos 100 dimensiones, pero cualquier valor en el rango de 100 a 300 es igual de popular. Las subpalabras son todas las subcadenas contenidas en una palabra entre el tamaño mínimo (minn) y el tamaño máximo (maxn). Por defecto, tomamos todas las subpalabras entre 3 y 6 caracteres, pero otro rango podría ser más apropiado para diferentes idiomas:

model = fasttext.train_unsupervised('data/fil9', minn=2, maxn=5, dim=300)

# Dependiendo de la cantidad de datos que tenga, es posible que desee cambiar los parámetros del entrenamiento. El parámetro epoch controla cuántas veces el modelo recorrerá sus datos. De forma predeterminada, recorremos el conjunto de datos 5 veces. Si el conjunto de datos es extremadamente masivo, es posible que desee recorrerlo con menos frecuencia.
#
# Otro parámetro importante es la tasa de aprendizaje lr. Cuanto mayor sea la tasa de aprendizaje, más rápido convergerá el modelo a una solución, pero corremos el riesgo de sobreajustarse al conjunto de datos. El valor predeterminado es 0.05, que es un buen tradeoff. Si quieren jugar con él, les sugerimos que se mantengan en el rango de \[0.01, 1\]:

model = fasttext.train_unsupervised('data/fil9', epoch=1, lr=0.5)

# Finalmente, FastText es multiproceso y usa 12 cores por defecto. Si tiene menos núcleos nuestro CPU (digamos 4), puede establecer fácilmente el número de subprocesos utilizando el flag thread:

model = fasttext.train_unsupervised('data/fil9', thread=4)

# ## Imprimiendo los word vectors
#
# Buscar e imprimir los word vectors directamente desde el archivo fil9.vec es engorroso. Afortunadamente, hay una funcionalidad de hacerlo en fastText.
#
# Por ejemplo, podemos imprimir los wordvectors de las palabras  asparagus, pidgey and yellow con el siguiente comando:

[model.get_word_vector(x) for x in ["asparagus", "pidgey", "yellow"]]

# Una característica interesante es que también pueden consultar palabras que no aparecieron en sus datos. De hecho, las palabras están representadas por la suma de sus subcadenas. Siempre que la palabra desconocida esté formada por subcadenas conocidas, ¡hay una representación de ella!
#
# Como ejemplo, intentemos con una palabra mal escrita (enviroment en vez enviroNment):

model.get_word_vector("enviroment")

# ¡Aún obtienes un word vector para ella! ¿Pero qué tan bueno es?

# ## Nearest neighbor queries
#
# Una forma sencilla de comprobar la calidad de un word vector es mirar a sus vecinos más cercanos. Esto da una intuición del tipo de información semántica que los vectores son capaces de capturar.
#
# Se puede lograr con la funcionalidad de vecino más cercano (nn). Por ejemplo, podemos consultar los 10 vecinos más cercanos de la palabra asparagus (esparragos) ejecutando el siguiente comando:

model.get_nearest_neighbors('asparagus')

# ¡Bien! Parece que los vectores de vegetales son similares.
#
# ¿Qué pasa con los pokemons?

model.get_nearest_neighbors('pidgey')

# ¡Diferentes evoluciones del mismo Pokémon tienen vectores cercanos!
#
# ¿Qué pasará con nuestra palabra mal escrita, su vector se acerca será algo razonable? Vamos a averiguarlo:

model.get_nearest_neighbors('enviroment')

# Gracias a la información contenida en la palabra, el vector de nuestra palabra mal escrita coincide con palabras razonables.

# ## Analogías
#
# Con un espíritu similar, se puede jugar con analogías de palabras. Por ejemplo, podemos ver si nuestro modelo puede adivinar que palabra es para Argentina dado lo qué Berlín es para Alemania.
#
# Esto se puede hacer con la funcionalidad de analogías. Toma un triplete de palabras (como Alemania, Berlín, Argentina) y genera la analogía:

model.get_analogies("berlin", "germany", "argentina")

# La respuesta que nos da nuestro modelo es Buenos y Aires, que es correcto.
# Echemos un vistazo a un ejemplo menos obvio:

model.get_analogies("psx", "sony", "nintendo")

# Nuestro modelo considera que la analogía de nintendo con una psx es el gamecube, lo que parece razonable. Por supuesto, la calidad de las analogías depende del conjunto de datos utilizado para entrenar el modelo y uno solo puede esperar cubrir conceptos que esten en el conjunto de datos.

model.get_analogies("usa", "bush", "brazil")

# ## Importancia de los n-gramas
#
# El uso de información a nivel de tokens es particularmente interesante para construir vectores para palabras desconocidas. Por ejemplo, la palabra gearshift (palanca de cambios) no existe en este dataset de Wikipedia, pero aún podemos consultar sus palabras existentes más cercanas:

model.get_nearest_neighbors('gearshift')

# La mayoría de las palabras devueltas comparten subcadenas sustanciales, pero algunas son bastante diferentes, como rueda cogwheel (engranaje).
#
# Ahora que hemos visto la importancia de la información de los tokens de ngrams para palabras desconocidas, veamos cómo se compara con un modelo que no usa información de subpalabras. Para entrenar un modelo sin subpalabras, simplemente llevamos el maxn a 0:

model_without_subwords = fasttext.train_unsupervised('data/fil9', maxn=0)

# Para ilustrar la diferencia, tomemos una palabra poco común en Wikipedia, como accomodation, que es un error ortográfico de accomModation. Aquí están los vecinos más cercanos obtenidos sin subpalabras:

model_without_subwords.get_nearest_neighbors('accomodation')

# El resultado no tiene mucho sentido, la mayoría de estas palabras no están relacionadas. Por otro lado, el uso de información de subpalabras da la siguiente lista de vecinos más cercanos

model.get_nearest_neighbors('accomodation')

# Los vecinos más cercanos capturan diferentes variaciones en torno a la palabra alojamiento. También obtenemos palabras relacionadas semánticamente como amenities o catering.
