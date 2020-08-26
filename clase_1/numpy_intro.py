#!/usr/bin/env python
# coding: utf-8

# # Clase 1 - Introducción a numpy

# Numpy es una biblioteca para Python que facilita el manejo de arreglos multidimensionales y ofrece varias herramientas para trabajar con ellos. Muchas de las bibliotecas de Python que son ampliamente usadas hoy en día, como pandas, están construidas sobre numpy.

# ## Listas de Python vs arreglos de Numpy

# A primera vista, un arreglo de numpy puede resultar idéntico a una lista de python, pero a medida que la cantidad de datos comienza a incrementar, los arreglos de numpy terminan ofreciendo un manejo más eficiente de la memoria.

# Para comenzar, vamos a crear un arreglo de numpy:

# In[2]:


import numpy as np

np.array([1, 2, 3, 4])


# Los arreglos pueden ser también multidimensionales:

# In[3]:


np.array([[1, 2, 3, 4], [5, 6, 7, 8]])


# Es importante tener en cuenta que un arreglo de numpy tiene un tipo fijo de datos, entonces si se quiere agregar un dato de un tipo diferente al de la mayoría, este va a ser modificado para adaptarse al resto

# In[8]:


enteros = np.array([1, 2, 3, 4])

# Agrego un elemento de tipo flotante en la posición 1

enteros[1] = 8.4727
enteros


# Numpy también nos permite crear arreglos con valores aleatorios del 0 al 1

# In[9]:


# Basta con pasarle las dimensiones del arreglo que queremos crear

np.random.rand(2, 3)


# ## Slicing

# De la misma forma que con las listas de python, pueden obtenerse slices de los arreglos de numpy

# In[11]:


enteros[:2]


# In[12]:


matriz_de_enteros = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

matriz_de_enteros[:2, :3]


# ## Reshape

# In[ ]:
