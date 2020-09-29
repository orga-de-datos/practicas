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

# # Python

# + [markdown] slideshow={"slide_type": "slide"}
# ## Un poco de Historia

# + [markdown] slideshow={"slide_type": "subslide"}
# Python fue creado a finales de los años 80 por un programador holandés llamado **Guido van Rossum**,
# quien sigue siendo aún hoy el líder del desarrollo del lenguaje.
#
# (Edit julio 2018: [ya no más](https://www.mail-archive.com/python-committers@python.org/msg05628.html))

# + [markdown] slideshow={"slide_type": "subslide"}
# El nombre del lenguaje proviene de los humoristas británicos Monty Python.
#
# >*"I chose Python as a working title for the project, being in a slightly irreverent mood (and a big fan of Monty Python's Flying Circus)."*

# + [markdown] slideshow={"slide_type": "slide"}
# ## Caracteristicas
#
# - Interpretado
# - Tipado dinamico
# - Multiparadigma
# - Alto nivel
# - Tiene un recolector de basura (no hay malloc, free, realloc, etc)
# -

# ## ¿Cómo empezar?
#
# * Al ser un lenguaje *interpretado*, se puede ir escribiendo a medida que se ejecuta, sin necesidad de compilar de antemano! Solamente hace falta escribir `python` o `python3` en una terminal para empezar
#
# * También, permite escribir archivos y correrlos. Crear un archivo con extensión `.py` y luego correr `python miarchivo.py` en laterminal

# + [markdown] slideshow={"slide_type": "slide"}
# ## El Zen de Python

# + slideshow={"slide_type": "slide"}
import this

# + [markdown] slideshow={"slide_type": "slide"}
# ## Conocimientos Básicos de Python: Variables y Tipos

# + slideshow={"slide_type": "slide"}
# Este es un comentario

print("Hello World!")
# -

# Los strings en python puden escribirse tanto con comillas simples (`'`) como comillas dobles (`"`). Normalmente vemos texto entre comillas triples para escribir _docstrings_, segun la guia de estilo de Python, el PEP8.

# ### Declaracion de variables

string = 'Hola'
print(string)

entero = 1
print(entero)

flotante = 1.0
print(flotante)

tupla = (entero, flotante)
print(tupla)

nupla = (entero, flotante, string)
print(nupla)

lista = [entero, flotante, string]
print(lista)

diccionario = {'1': tupla, 50: nupla, '3': entero}
print(diccionario)

conjunto = set([1, 2])
print(conjunto)

booleano = True
print(booleano)

nada = None
print(nada)

# Ojo que las variables pueden cambiar de tipo!

# +
elemento = 1
print(elemento)
print(type(elemento))

elemento = str(1)
print(elemento)
print(type(elemento))

# +
elemento = ['dos']

print(elemento)
print(type(elemento))
# -

# ### Tipos basicos

# + [markdown] slideshow={"slide_type": "slide"}
# #### Listas de Python
# -

lista = list()
lista

# + slideshow={"slide_type": "slide"}
lista = []
lista

# + slideshow={"slide_type": "slide"}
lista = [1, 2, 3, 4]
lista

# + slideshow={"slide_type": "slide"}
lista.append(1)  # Inserto un 1 al final
lista.append("dos")  # Inserto un "dos" al final
lista.append(3.0)  # Inserto un 3.0 al final
lista.insert(2, 10)  # Inserto en posicion 2 un 10
print(lista)
# -

len(lista)

lista.pop()

lista.index(10)

lista.remove(10)
lista

for elemento in lista:
    print(elemento)

for i, elemento in enumerate(lista):
    print(f"{i}-ésimo elemento: {elemento}")

sorted(lista)

lista.remove("dos")

sorted(lista)

lista.sort()
lista

# + [markdown] slideshow={"slide_type": "slide"}
# #### Tuplas de Python
#
# Las tuplas son inmutables. No se pueden agregar elementos luego de creadas.

# + slideshow={"slide_type": "slide"}
tupla = (1, 2)

print(tupla)
print(tupla[0])
print(tupla[1])

tupla[1] = 3  # Falla. No se puede mutar

# + [markdown] slideshow={"slide_type": "slide"}
# #### Diferencia entre lista y tupla
# Las listas se caracterizan por ser mutables, es decir, se puede cambiar su contenido en tiempo de ejecución, mientras que las tuplas son inmutables ya que no es posible modificar el contenido una vez creada.
# -

# #### Slices
#
# **Valen para listas, tuplas o strings (_segmentos_)**

numeros = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(numeros)

print(numeros[2])  # Imprimo elemento en la posición 2

print(numeros[-1])  # # Imprimo elemento en la última posición

print(numeros[0:2])  # Imprimo de la pos 0 a la pos 2

print(numeros[-4:-2])

print(numeros[0:80])

print(numeros[:3])

print(numeros[3:])

print(numeros[::2])

numeros[7] = 'siete'  # Las listas se pueden mutar
print(numeros)

numeros = numeros[::-1]
print(numeros)

print(numeros[15])  # Falla. No se puede acceder a una posición inexistente

palabra = 'palabra'
print(palabra)
print(palabra[3])
print(palabra[:3])
print(palabra[3:])

# +
tupla = (0, 1)

print(tupla)
print(tupla[0])
print(tupla[1])

# + [markdown] slideshow={"slide_type": "slide"}
# #### Diccionarios de Python
#
# Son como hashmaps, las claves deben ser inmutables para que no pierda sentido el diccionario. Si se pudieran modificar, se podrían cambiar las claves y generaría conflictos.
#
# Tipos mutables:
# - Listas
# - Diccionarios
# - Sets
#
# Tipos inmutables:
# - Int
# - Float
# - String
# - Tuplas
#

# + slideshow={"slide_type": "slide"}
diccionario = {}
diccionario
# -

diccionario = dict()
diccionario

# + slideshow={"slide_type": "slide"}
# Cómo agregar cosas al diccionario
diccionario['clave1'] = 'valor1'
diccionario[2] = 'valor2'
diccionario['clave3'] = 3
print(diccionario)
# -

# Hay dos formas de obtener valores de un diccionario:
#
# ```python
# diccionario[clave]
# ```
#
# El cual devuelve el valor si existe la clave suministrada o bien lanza `KeyError` si no existe.

diccionario['clave1']

diccionario['clave1000']

# La segunda forma es con `get`:

# +
# diccionario.get?

# + slideshow={"slide_type": "slide"}
diccionario.get('clave1000', 2)

# + slideshow={"slide_type": "slide"}
print('clave1' in diccionario)  # Verifico si la clave está en el diccionario

# + slideshow={"slide_type": "slide"}
# Cómo iterar un diccionario elemento por elemento
for (
    clave,
    valor,
) in (
    diccionario.items()
):  # diccionario.items() va devolviendo tuplas con el formato (clave,valor)
    print(
        f"{clave}: {valor}"
    )  # con esta sintaxis se desempaquetan en clave y valor (similar a enumerate)

# + slideshow={"slide_type": "slide"}
for clave in diccionario.keys():
    print(clave)

# + slideshow={"slide_type": "slide"}
for valor in diccionario.values():
    print(valor)
# -

# #### Sets
#
# Son similares a los diccionarios (en eficiencia) pero se almacenan solo claves, y tienen algunas operaciones particulares.
#
# En particular, no pueden tener elementos iguales (pensar que son conjuntos)

# +
# set??
# -

# Se definen como los diccionarios pero sin hacerlos 'clave:valor', solamente una seguidilla de elementos
{1, 2, 2, 3}

set([1, 2, 2, 3])

# ## Condicionales (if...elif...else)
#
# ```python
# if <condición_1>:
#     <hacer algo_1 si se da la condición_1>
# elif <condición_2>:
#     <hacer algo_2 si se da la condición_2>
# ...
# elif <condición_n>:
#     <hacer algo_n si se da la condición_n>
# else:
#     <hacer otra cosa si no dan las anteriores>
# ```
#
# Algo importante para notar es que los bloques se definen por **niveles de identacion**.

# ## Iteraciones
#
# ```python
# while cond:
#     <codigo>
# ```
#
# ```python
# for elemento in iterable:
#     <codigo>
# ```
#
# Para iterar sobre un rango de valores, usamos `range`

for i in range(1, 11, 2):
    print(i)


# ## Operadores logicos
#
# `not`, `or`, `and`

# + [markdown] slideshow={"slide_type": "slide"}
# ## Funciones en Python

# + slideshow={"slide_type": "slide"}
def busqueda_binaria(lista, elemento):
    if not lista:
        return False
    elif len(lista) == 1:
        return lista[0] == elemento
    mitad = len(lista) // 2  # // es la operación división entera
    if lista[mitad] == elemento:
        return True
    if lista[mitad] > elemento:
        return busqueda_binaria(lista[:mitad], elemento)
    if lista[mitad] < elemento:
        return busqueda_binaria(lista[mitad:], elemento)


print(busqueda_binaria([1, 2, 3, 4, 5], 4))
print(busqueda_binaria([1, 4, 6, 7, 9, 10], 2))


# + slideshow={"slide_type": "slide"}
def suma(a, b):
    return a + b


print(suma(1, 2))
print(suma(1.0, 2.0))
print(suma(1.0, 2))
print(suma("hola ", "como te va"))
print(suma([1, 2, 3], [4, 5]))
print(suma("1", 3))  # Falla


# +
# El valor por default de divisor es 1


def division(dividendo, divisor=1):
    return dividendo / divisor


print(division(4))  # Usa el valor por default
print(division(1, 2))  # Parámetros por orden
print(division(dividendo=1, divisor=2))  # Parámetros por nombre
print(division(divisor=2, dividendo=1))

# +
# Funciones básicas ya en el lenguaje
# Hechas para funcionar para distintos tipos

string_ordenado = sorted('bca')
print(string_ordenado)

lista_ordenada = sorted([1, 3, 2])
print(lista_ordenada)

separadas = "hola, don, pepito".split(",")
print(separadas)
unidas = "".join(separadas)
print(unidas)
# -

# ## Módulos
#
# Para incluir alguna biblioteca de funciones se usa `import`. Pueden ser cosas ya predefinidas en Python (`math`, `random`, etc), nombres de archivos en nuestro directorio (por ejemplo, para `mimodulo.py` ponemos `import mimodulo`) o bibliotecas instaladas por el usuario

# +
import math

print(math.pi)

from math import pi

print(pi)
# -

# ## Manejo de excepciones
#
# Se pueden encapsular errores esperados en un bloque 'try/except' para evitar cortar el flujo del programa

division(1, 0)  # No se puede dividir por cero

try:
    division(1, 0)
except ZeroDivisionError:
    print('No se puede dividir por cero, ojo!')

# + [markdown] slideshow={"slide_type": "slide"}
# ## Lectura y escritura de archivos

# + slideshow={"slide_type": "slide"}
import random

with open(
    'archivo.csv', 'w'
) as archivo:  # Al usar esta sintaxis no es necesario hacer close
    archivo.write("Alumno, nota\n")
    # Tambien de forma similar al fprintf se puede hacer:
    # print("Alumno, nota\n", file=archivo)
    for i in range(0, 10):
        archivo.write(f"{i},{random.randrange(0,10)}\n")

print(archivo)  # Comentario aclaratorio:
# Las variables definidas en un determinado scope siguen existiendo por fuera del mismo.
# Se debe tener cuidado con esto, ya que nada garantiza que por fuera el valor sea el esperado.

# + slideshow={"slide_type": "slide"}
with open('archivo.csv', 'r') as f:
    for linea in f:
        print(linea.strip())
# -

with open('archivo.csv', 'r') as f:
    print(f.read())


# + [markdown] slideshow={"slide_type": "slide"}
# ## Objetos
#
# Los objetos tienen metodos y atributos:
# - Atributos: equivalentes a variables.
# - Métodos: equivalentes a las primitivas.

# + [markdown] slideshow={"slide_type": "slide"}
# ### Cómo creo una clase

# + slideshow={"slide_type": "slide"}
class Nodo(object):
    def __init__(self, dato, siguiente=None):
        self._dato = dato
        self._siguiente = siguiente

    @property
    def dato(self):
        return self._dato

    @property
    def proximo(self):
        return self._siguiente

    @proximo.setter
    def proximo(self, siguiente):
        self._siguiente = siguiente

    def __repr__(self):
        return str(self.dato)

    def __str__(self):
        return str(self.dato)


# + slideshow={"slide_type": "slide"}
nodo = Nodo("hola")
print(nodo)

# + slideshow={"slide_type": "slide"}
nodo2 = Nodo("lala")
print([nodo, nodo2])

# + slideshow={"slide_type": "slide"}
nodo3 = nodo.dato
print(nodo3)


# + [markdown] slideshow={"slide_type": "slide"}
# ### Ejemplo: Lista Enlazada

# + slideshow={"slide_type": "slide"}
class ListaEnlazada(object):
    def __init__(self):
        self._primero = None
        self._ultimo = None
        self._largo = 0

    def __len__(self):
        return self._largo

    def insertar_al_principio(self, dato):
        nodo = Nodo(dato, self._primero)
        self._primero = nodo
        self._largo += 1
        if self._largo == 1:
            self._ultimo = nodo

    def insertar_al_final(self, dato):
        if self._largo != 0:
            nodo = Nodo(dato)
            nodo_anterior = self._ultimo
            nodo_anterior._siguiente = nodo
            self._ultimo = nodo
            self._largo += 1
        else:
            self.insertar_al_principio(dato)

    @property
    def primero(self):
        return self._primero.dato

    def borrar_primero(self):
        dato = self.primero.dato
        self._primero = self.primero.siguiente
        self._largo -= 1
        if self._largo == 0:
            self._ultimo = None
        return dato

    def __str__(self):
        datos = []
        nodo_actual = self._primero
        while nodo_actual:
            datos.append(nodo_actual.dato)
            nodo_actual = nodo_actual.proximo
        return " -> ".join(datos)

    def __repr__(self):
        return self.__str__()


# + slideshow={"slide_type": "slide"}
lista = ListaEnlazada()
lista.insertar_al_principio("Primer Dato")
lista.insertar_al_principio("Primer primer Dato")
len(lista)
# -

lista

# + slideshow={"slide_type": "slide"}
elemento = lista.primero
print(elemento)
# -

# ## Recursos
#
# * [Taller de Python de Algoritmos II](https://github.com/algoritmos-rw/algo2_apuntes)
#
# * [Documentación de Python 3](https://docs.python.org/3/tutorial/)
#
# * [Apunte de Algoritmos y Programación I](https://algoritmos1rw.ddns.net/material)
#
# * [Automate the Boring Stuff with Python](http://automatetheboringstuff.com/)
#
# * [Curso Python](https://pythoncurso.github.io)
#
# * [Python Tutor](http://pythontutor.com/)
#
# * [Learn Python3 in Y minutes](https://learnxinyminutes.com/docs/python3/)
#
# * [Bibliografía de Algoritmos y Programación I](https://algoritmos1rw.ddns.net/bibliografia)
