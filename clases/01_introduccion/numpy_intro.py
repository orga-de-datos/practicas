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

# ## Clase 1 - Introducción a Numpy
#
# Numpy es una biblioteca para Python que facilita el manejo de arreglos multidimensionales y ofrece varias herramientas para trabajar con ellos. Muchas de las bibliotecas de Python que son ampliamente usadas hoy en día, como pandas, están construidas sobre numpy.
#
# ### Listas de Python vs arreglos de Numpy
#
# A primera vista, un arreglo de numpy puede resultar idéntico a una lista de python, pero a medida que la cantidad de datos comienza a incrementar, los arreglos de numpy terminan ofreciendo un manejo más eficiente de la memoria.
#
# Para comenzar, vamos a crear un arreglo de numpy:

# +
import numpy as np

np.array([1, 2, 3, 4])
# -

# Los arreglos pueden ser también multidimensionales:

np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# Es importante tener en cuenta que un arreglo de numpy tiene un tipo fijo de datos, entonces si se quiere agregar un dato de un tipo diferente al de la mayoría, este va a ser modificado para adaptarse al resto

enteros = np.array([1, 2, 3, 4])

# Agrego un elemento de tipo flotante en la posición 1

enteros[1] = 8.4727
enteros

# Numpy también nos permite crear arreglos con valores aleatorios del 0 al 1.
# Basta con pasarle las dimensiones del arreglo que queremos crear.

np.random.rand(2, 3)

# ## Slicing
#
# De la misma forma que con las listas de python, pueden obtenerse slices de los arreglos de numpy

enteros[:2]


# +
matriz_de_enteros = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print('Original: ')
print(matriz_de_enteros)

print()

print('Recortada: ')
print(matriz_de_enteros[:2, :3])

# +
# 3D
# arange() genera valores de un intervalo pasado por parámetro
# reshape() modifica la forma del numpy array

x = np.arange(45).reshape(3, 3, 5)
x
# -

x[0]

x[0][1]

x[0][1][2]

#
# ¿Cómo conseguimos estos valores? ([fuente](https://towardsdatascience.com/indexing-and-slicing-of-1d-2d-and-3d-arrays-in-numpy-e731afff0bbe))
#
# ![title](img/matrix.png)

x[1:, 0:2, 1:4]

# ### Copia de arreglos

# +
# Los arreglos no se copian con asignación

a = np.array([1, 2, 3, 4])
b = a
b
# -

b[1] = 20
b

a

# +
# Para copiar un arreglo a otra variable debemos usar copy()

a = np.array([1, 2, 3, 4])
b = a.copy()
b[1] = 20
b
# -

a

# ### Modificación de dimensiones

# Existen varias operaciones para cambiar la forma de un arreglo de numpy

matriz_de_enteros

# +
# Obtener las dimensiones del arreglo

matriz_de_enteros.ndim

# +
# Obtener la forma del arreglo

matriz_de_enteros.shape

# +
# Modificar la forma de un arreglo

enteros = np.array([3, 6, 9, 12])
print(f"enteros: {enteros}")
np.reshape(enteros, (2, 2))

# +
# Aplanar un arreglo

a = np.ones((2, 2))
a
# -

a.flatten()

a


# ### Combinación de arreglos (Stacking)

# +
# Los arreglos se pueden combinar verticalmente (se incrementa la cantidad de filas)

a = np.arange(0, 5)
a
# -

b = np.arange(5, 10)
b

combinados_verticalmente = np.vstack((a, b))
combinados_verticalmente

# +
# También se pueden combinar horizontalmente (se incrementa la cantidad de columnas)

combinados_horizontalmente = np.hstack((a, b))
combinados_horizontalmente
# -

# ### Operaciones matemáticas

# +
a = np.array([1, 2, 3, 4])

a + 2
# -

a ** 2

b = np.ones(4)
print(f"b: {b}")
a + b

# ### Estadística

# +
a = np.array([[5, 2, 1, 8], [26, 4, 17, 9]])

np.min(a)
# -

np.max(a)

np.sum(a)

# ### Más magia

# +
a = np.array([[5, 2, 1, 8], [26, 4, 17, 9]])

a > 5
# -

a[a > 5]

# ### Broadcasting

# Permite realizar operaciones entre dos numpy arrays de distintas dimensiones.
# Para lograr esto, las dimensiones de los mismos deben ser compatibles. Dos dimensiones son compatibles cuando:
# 1. Son iguales
# 2. Alguna de las dos es 1

a = np.array([[5, 2, 1, 8], [26, 4, 17, 9]])
a

# Armamos un array que tenga la misma cantidad de columnas que a
b = np.array([5, 2, 1, 8])
print(f"a: {a.shape}")
print(f"b: {b.shape}")
# a + b
print()
print(f"a + b:\n {a + b}")

# + tags=["raises-exception"]
# Armamos un array que tenga distinta cantidad de filas y columnas que a
b = np.array([5, 1, 8])
print(f"a: {a.shape}")
print(f"b: {b.shape}")
a + b
# -

# Armamos un array que tenga la misma cantidad de filas que a
b = np.array([[2], [1]])
print(f"a: {a.shape}")
print(f"b: {b.shape}")
a + b

# Si b es un entero
b = 4
a + b


