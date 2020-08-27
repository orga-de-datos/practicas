#!/usr/bin/env python
# coding: utf-8
# %%

# # Clase 1 - Introducción a numpy

# Numpy es una biblioteca para Python que facilita el manejo de arreglos multidimensionales y ofrece varias herramientas para trabajar con ellos. Muchas de las bibliotecas de Python que son ampliamente usadas hoy en día, como pandas, están construidas sobre numpy.

# ## Listas de Python vs arreglos de Numpy

# A primera vista, un arreglo de numpy puede resultar idéntico a una lista de python, pero a medida que la cantidad de datos comienza a incrementar, los arreglos de numpy terminan ofreciendo un manejo más eficiente de la memoria.

# Para comenzar, vamos a crear un arreglo de numpy:

# %%


import numpy as np

np.array([1, 2, 3, 4])


# Los arreglos pueden ser también multidimensionales:

# %%


np.array([[1, 2, 3, 4], [5, 6, 7, 8]])


# Es importante tener en cuenta que un arreglo de numpy tiene un tipo fijo de datos, entonces si se quiere agregar un dato de un tipo diferente al de la mayoría, este va a ser modificado para adaptarse al resto

# %%


enteros = np.array([1, 2, 3, 4])

# Agrego un elemento de tipo flotante en la posición 1

enteros[1] = 8.4727
enteros


# Numpy también nos permite crear arreglos con valores aleatorios del 0 al 1

# %%


# Basta con pasarle las dimensiones del arreglo que queremos crear

np.random.rand(2, 3)


# ## Slicing

# De la misma forma que con las listas de python, pueden obtenerse slices de los arreglos de numpy

# %%


enteros[:2]


# %%


matriz_de_enteros = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print('Original: ')
print(matriz_de_enteros)

print()

print('Recortada: ')
print(matriz_de_enteros[:2, :3])

# %% [markdown]
# ### Copia de arreglos

# %%
# Los arreglos no se copian con asignación

a = np.array([1, 2, 3, 4])
b = a
b

# %%
b[1] = 20
b

# %%
a

# %%
# Para copiar un arreglo a otra variable debemos usar copy()

a = np.array([1, 2, 3, 4])
b = a.copy()
b[1] = 20
b

# %%
a

# %% [markdown]
# ### Modificación de dimensiones

# %% [markdown]
# Existen varias operaciones para cambiar la forma de un arreglo de numpy

# %%
# Obtener las dimensiones del arreglo

matriz_de_enteros.ndim

# %%
# Obtener la forma del arreglo

matriz_de_enteros.shape

# %%
# Modificar la forma de un arreglo

enteros = np.array([3, 6, 9, 12])
np.reshape(enteros, (2, 2))

# %%
# Aplanar un arreglo

a = np.ones((2, 2))
a

# %%
a.flatten()

# %%
a


# %% [markdown]
# ### Combinación de arreglos (Stacking)

# %%
# Los arreglos se pueden combinar verticalmente (se incrementa la cantidad de filas)

a = np.arange(0, 5)
a

# %%
b = np.arange(5, 10)
b

# %%
combinados_verticalmente = np.vstack((a, b))
combinados_verticalmente

# %%
# También se pueden combinar horizontalmente (se incrementa la cantidad de columnas)

combinados_horizontalmente = np.hstack((a, b))
combinados_horizontalmente

# %% [markdown]
# ### Operaciones matemáticas

# %%
a = np.array([1, 2, 3, 4])

a + 2

# %%
a ** 2

# %%
b = np.ones(4)
a + b

# %% [markdown]
# ### Estadística

# %%
a = np.array([[5, 2, 1, 8], [26, 4, 17, 9]])

np.min(a)

# %%
np.max(a)

# %%
np.sum(a)

# %% [markdown]
# ### Más magia

# %%
a = np.array([[5, 2, 1, 8], [26, 4, 17, 9]])

a > 5

# %%
a[a > 5]

# %%
