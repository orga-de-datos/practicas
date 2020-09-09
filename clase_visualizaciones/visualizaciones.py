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

import pandas as pd
from matplotlib import pyplot as plt

# # Intro
#
# La visualización de datos es el proceso de proveer una representacion visual de datos. En esta clase revisaremos algunas de las bibliotecas mas comunes para este proposito dentro del ecosistema de python y al final de la misma tendremos herramientas para comunicar datos de una manera efectiva.
#
# ## Bibliotecas
# - Módulo de plotting de pandas
# - Pyplot
# - Seaborn
# - Pandas profiling

# # Descargamos la data
# Vamos a utilizar los datos de la [encuesta de sueldos 2020.02](https://sysarmy.com/blog/posts/resultados-de-la-encuesta-de-sueldos-2020-2/) de [sysarmy](https://sysarmy.com/es/).

# +
GSPREADHSEET_DOWNLOAD_URL = (
    "https://docs.google.com/spreadsheets/d/{gid}/export?format=csv&id={gid}".format
)

SYSARMY_2020_2_GID = '1FxzaPoS0AkN8E_-aeobpr7FHAAy8U7vWcGE7PY4kJmQ'
# -

df = pd.read_csv(GSPREADHSEET_DOWNLOAD_URL(gid=SYSARMY_2020_2_GID), skiprows=9)

# ## Una pequeña preview

pd.options.display.max_columns = None
df.head()

# # Pandas plotting
#
#

# Pandas incorpora algunas facilidades para [visualizaciones](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html) que son _wrappers_ alrededor de matplotlib. Se pueden utilizar [otros backends](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#plotting-backends) desde la versión 0.25.
#
# Veremos algunos plots sencillos aquí.

# ## Pie chart

# Tenemos una variable binaria en la encuesta de sueldos que indica si la persona que respondió contribuye a proyectos open source. La respuesta es por `sí` o por `no`. Queremos ver como se distribuyen las respuestas. Para esto, hagamos un [pie chart](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#pie-plot).

df['¿Contribuís a proyectos open source?'].value_counts().plot(
    kind='pie', autopct='%1.0f%%'
)

# ## Barplot
#
# Ahora, consideremos la elección de sistemas operativos:

df['¿Qué SO usás en tu laptop/PC para trabajar?'].value_counts().sort_values(
    ascending=False
)

# Con un pie plot, quedaría demasiado... complicado de interpretar:

df['¿Qué SO usás en tu laptop/PC para trabajar?'].value_counts().plot(
    kind='pie', autopct='%1.0f%%'
)

# BSD es una elección poco común, queda totalmente perdida. Por otro lado, sin las anotaciones de los porcentajes, sería muy dificil saber la diferencia entre `macOS` y `GNU/Linux`. `Windows` es más de la mitad, pero... ¿Cuánto mas?
#
# Por esto, es mejor un [bar plot](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#bar-plots):

df['¿Qué SO usás en tu laptop/PC para trabajar?'].value_counts().sort_values(
    ascending=False
).plot(kind='bar')

# ## Scatter plot
#
# Ahora tenemos la duda, ¿Cómo se relaciona el salario bruto con el salario neto? Tenemos puntos en dos dimensiones y queremos entender como se relacionan. Para ello, un [scatter plot](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#scatter-plot) es adecuado:

df.plot(
    x='Salario mensual NETO (en tu moneda local)',
    y='Salario mensual BRUTO (en tu moneda local)',
    kind='scatter',
)

# Tenemos un montón de puntos apelotonados en la diagonal, veamos de reducir el diámetro de cada punto:

df.plot(
    x='Salario mensual NETO (en tu moneda local)',
    y='Salario mensual BRUTO (en tu moneda local)',
    kind='scatter',
    s=5,
)

# ## Histograma
#
# Ahora, la encuesta considera también los ajustes salariales. Ese porcentaje varía por empresa. ¿hay muchos valores únicos?

df['¿De qué % fue el ajuste total?'].nunique()

# Sí, montones. Podemos hacer un bar plot?

# +
# df['¿De qué % fue el ajuste total?'].plot(kind='bar')
# -

# Podemos _intentarlo_ pero tarda una barbaridad en renderizarse.
#
# Entonces, tenemos un soporte continuo, demasiados valores únicos. Seamos inteligentes y hagamos un [histograma](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#histograms).

df['¿De qué % fue el ajuste total?'].plot(kind='hist')

# Cambiando la cantidad de `bins` tenemos mayor granularidad:

df['¿De qué % fue el ajuste total?'].plot(kind='hist', bins=25)

df['¿De qué % fue el ajuste total?'].plot(kind='hist', bins=50)

# ## Box plots
#
# Siguiendo la línea de los salarios netos y brutos... ¿Cuánto es la media? ¿La mediana? Podemos tener un resumen estadístico con un [box plot](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#box-plots)

df[
    [
        'Salario mensual NETO (en tu moneda local)',
        'Salario mensual BRUTO (en tu moneda local)',
    ]
].plot(kind='box')

# Los labels quedan feos... como un hack, podemos renombrarlos:

df[
    [
        'Salario mensual NETO (en tu moneda local)',
        'Salario mensual BRUTO (en tu moneda local)',
    ]
].rename(
    columns={
        'Salario mensual NETO (en tu moneda local)': 'Salario neto',
        'Salario mensual BRUTO (en tu moneda local)': 'Salario bruto',
    }
).plot(
    kind='box'
)

# # Matplotlib

# Dijimos que de fondo pandas usa [matplotlib](https://matplotlib.org) para hacer los plots. Es una librería que permite trabajar a bajo nivel, pero que también tiene un módulo de alto nivel llamado [pyplot](https://matplotlib.org/api/pyplot_api.html) que ofrece una interfaz similar a matlab, y es bastante cómoda. Muchas librerías de visualizaciones usan de fondo matplotlib.
#
# Si revisamos los plots que hemos visto, nos gustaría poder cambiar algunas cosas:
# - el tamaño
# - la escala
# - agregarle título
# - descripción del eje y
# - descripción del eje x
# - etc
#
# Revisaremos estos conceptos para trabajar más comodamente al momento de hacer plots

from matplotlib import pyplot as plt
import matplotlib

# ## Elementos de un plot
# El siguiente código ha sido tomado de [la documentación de matplotlib](https://matplotlib.org/3.1.1/gallery/showcase/anatomy.html#anatomy-of-a-figure) para mostrar los diferentes elementos de un plot

# + jupyter={"source_hidden": true}
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

np.random.seed(19680801)

X = np.linspace(0.5, 3.5, 100)
Y1 = 3 + np.cos(X)
Y2 = 1 + np.cos(1 + X / 0.75) / 2
Y3 = np.random.uniform(Y1, Y2, len(X))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect=1)


def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x


ax.xaxis.set_major_locator(MultipleLocator(1.000))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_major_locator(MultipleLocator(1.000))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

ax.set_xlim(0, 4)
ax.set_ylim(0, 4)

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=10)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

ax.plot(X, Y1, c=(0.25, 0.25, 1.00), lw=2, label="Blue signal", zorder=10)
ax.plot(X, Y2, c=(1.00, 0.25, 0.25), lw=2, label="Red signal")
ax.plot(X, Y3, linewidth=0, marker='o', markerfacecolor='w', markeredgecolor='k')

ax.set_title("Anatomy of a figure", fontsize=20, verticalalignment='bottom')
ax.set_xlabel("X axis label")
ax.set_ylabel("Y axis label")

ax.legend()


def circle(x, y, radius=0.15):
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withStroke

    circle = Circle(
        (x, y),
        radius,
        clip_on=False,
        zorder=10,
        linewidth=1,
        edgecolor='black',
        facecolor=(0, 0, 0, 0.0125),
        path_effects=[withStroke(linewidth=5, foreground='w')],
    )
    ax.add_artist(circle)


def text(x, y, text):
    ax.text(
        x,
        y,
        text,
        backgroundcolor="white",
        ha='center',
        va='top',
        weight='bold',
        color='blue',
    )


# Minor tick
circle(0.50, -0.10)
text(0.50, -0.32, "Minor tick label")

# Major tick
circle(-0.03, 4.00)
text(0.03, 3.80, "Major tick")

# Minor tick
circle(0.00, 3.50)
text(0.00, 3.30, "Minor tick")

# Major tick label
circle(-0.15, 3.00)
text(-0.15, 2.80, "Major tick label")

# X Label
circle(1.80, -0.27)
text(1.80, -0.45, "X axis label")

# Y Label
circle(-0.27, 1.80)
text(-0.27, 1.6, "Y axis label")

# Title
circle(1.60, 4.13)
text(1.60, 3.93, "Title")

# Blue plot
circle(1.75, 2.80)
text(1.75, 2.60, "Line\n(line plot)")

# Red plot
circle(1.20, 0.60)
text(1.20, 0.40, "Line\n(line plot)")

# Scatter plot
circle(3.20, 1.75)
text(3.20, 1.55, "Markers\n(scatter plot)")

# Grid
circle(3.00, 3.00)
text(3.00, 2.80, "Grid")

# Legend
circle(3.70, 3.80)
text(3.70, 3.60, "Legend")

# Axes
circle(0.5, 0.5)
text(0.5, 0.3, "Axes")

# Figure
circle(-0.3, 0.65)
text(-0.3, 0.45, "Figure")

color = 'blue'
ax.annotate(
    'Spines',
    xy=(4.0, 0.35),
    xytext=(3.3, 0.5),
    weight='bold',
    color=color,
    arrowprops=dict(arrowstyle='->', connectionstyle="arc3", color=color),
)

ax.annotate(
    '',
    xy=(3.15, 0.0),
    xytext=(3.45, 0.45),
    weight='bold',
    color=color,
    arrowprops=dict(arrowstyle='->', connectionstyle="arc3", color=color),
)

ax.text(
    4.0, -0.4, "Made with http://matplotlib.org", fontsize=10, ha="right", color='.5'
)

plt.show()
# -

# ### Figuras
# En lo que nos es relevante ahora, una [figura](https://matplotlib.org/faq/usage_faq.html#figure) es un contenedor de plots. Las figuras tienen un identificador único. Podemos obtener la figura activa con `plt.gcf()` (`g`et `c`urrent `f`igure) o crear una nueva con `plt.figure()`.
#
# Algunos parámetros que nos importan:
# ```
# figsize(float, float), default: rcParams["figure.figsize"] (default: [6.4, 4.8])
# Width, height in inches.
#
# dpifloat, default: rcParams["figure.dpi"] (default: 100.0)
# The resolution of the figure in dots-per-inch.
# ```

# ### Axis
# Un [axis](https://matplotlib.org/faq/usage_faq.html#axes) es un plot per se, digamos.

# ### Axis labels
# Son las descripciones en el eje x e y.

# ### Title
# Es el título de la figura (no del plot).

# ### Legend
# Son descripciones de colecciones de datos.

# ## Escalando los plots
#
# Tenemos dos parámetros para ello. Hacen cosas distintas.
#
# Por un lado `figsize` cambia el tamaño en pulgadas de la figura. `dpi` cambia la cantidad de pixels que hay en una pulgada.
#
# Entonces dada una figura con figsize $(w,h)$ y dpi $d$: $$p_x = d*w\\p_y = d*h $$
#
# Por defecto `dpi` vale `100` y `figsize` vale `[6.4, 4.8]`, de modo que obtendremos plots de `640 x 480`.
#
# Podemos cambiar los valores por defecto de `matplotlib` a través del diciconario `rcParams`:
#
# ```python
# matplotlib.rcParams['figure.dpi'] = 150
# ```
#
# Veamos ahora algunos ejemplos.

plt.figure()
df['¿Contribuís a proyectos open source?'].value_counts().plot(
    kind='pie', autopct='%1.0f%%'
)
plt.show()

plt.figure(figsize=(6.4 * 1.5, 4.8 * 1.5), dpi=100)
df['¿Contribuís a proyectos open source?'].value_counts().sort_index().plot(
    kind='pie', autopct='%1.0f%%'
)
plt.show()

plt.figure(figsize=(6.4, 4.8), dpi=150)
df['¿Contribuís a proyectos open source?'].value_counts().sort_index().plot(
    kind='pie', autopct='%1.0f%%'
)
plt.show()

# ## Plots con pyplot
#
# Vamos a repetir un poco los plots anteriores pero revisando la api de pyplot y cambiando algunas cosas.

# ### Bar plot

plt.figure(dpi=(125))
users_per_os = (
    df['¿Qué SO usás en tu laptop/PC para trabajar?']
    .value_counts()
    .sort_values(ascending=False)
)
plt.bar(users_per_os.index, users_per_os.values)
plt.ylabel("Usuarios")
plt.xlabel("Sistema operativo")
plt.title('¿Qué SO usás en tu laptop/PC para trabajar?')
plt.show()

# Algo que podemos ver es que `*BSD` es un valor prácticamente invisible. ¿Mejora si se pone en escala y-logaritmica?

plt.figure(dpi=(125))
users_per_os = (
    df['¿Qué SO usás en tu laptop/PC para trabajar?']
    .value_counts()
    .sort_values(ascending=False)
)
plt.bar(users_per_os.index, users_per_os.values)
plt.yscale("log")
plt.ylabel("Usuarios")
plt.xlabel("Sistema operativo")
plt.title('¿Qué SO usás en tu laptop/PC para trabajar?')
plt.show()

# No mucho, pero al menos se ve que está en el orden de $10^0$
#
# Y que es ese `plt.show()` que estamos poniendo ahora? Básicamente muestra todas las figuras abiertas. En un notebook no hay mucha necesidad de usarlo, pero veremos el `__repr__` del último elemento de la celda si no lo hemos asignado a una variable (o si no le hemos puesto un `;` al final).

# ### Scatter plot
#
# Revisemos un scatter plot, pero con la api de pyplot.

plt.figure(dpi=(150))
plt.scatter(
    x=df['Salario mensual NETO (en tu moneda local)'],
    y=df['Salario mensual BRUTO (en tu moneda local)'],
    s=5,
)
plt.ylabel("Salario bruto")
plt.xlabel("Salario neto")
plt.title('Relación entre salario neto y salario bruto')
plt.show()

# ¿Sería interesante ver que tan conforme está la gente con sus salarios no? Podemos introducir esa columna como color del scatter plot.

plt.figure(dpi=(125))
plt.scatter(
    x=df['Salario mensual NETO (en tu moneda local)'],
    y=df['Salario mensual BRUTO (en tu moneda local)'],
    s=5,
    c=df['¿Qué tan conforme estás con tu sueldo?'],
)
plt.ylabel("Salario bruto")
plt.xlabel("Salario neto")
plt.title('Relación entre salario neto y salario bruto')
plt.show()

# No tenemos ni idea de que es cada color. Pongamos un [legend](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend)!

# +
fig, ax = plt.subplots(dpi=150)

for conformity in df['¿Qué tan conforme estás con tu sueldo?'].unique():
    conformity_df = df[df['¿Qué tan conforme estás con tu sueldo?'] == conformity]
    ax.scatter(
        x=conformity_df['Salario mensual NETO (en tu moneda local)'],
        y=conformity_df['Salario mensual BRUTO (en tu moneda local)'],
        s=5,
        label=conformity,
        alpha=0.65,
    )

ax.legend()
plt.ylabel("Salario bruto")
plt.xlabel("Salario neto")
plt.title('Relación entre salario neto y salario bruto')
plt.show()
# -

# Pero hay outliers que nos complican... veamos los que están dentro del millón para ambas variables

# +
fig, ax = plt.subplots(dpi=150)

df_submm = df[
    (df['Salario mensual NETO (en tu moneda local)'] < 1e6)
    & (df['Salario mensual BRUTO (en tu moneda local)'] < 1e6)
]

for conformity in df_submm['¿Qué tan conforme estás con tu sueldo?'].unique():
    conformity_df = df_submm[
        df_submm['¿Qué tan conforme estás con tu sueldo?'] == conformity
    ]
    ax.scatter(
        x=conformity_df['Salario mensual NETO (en tu moneda local)'],
        y=conformity_df['Salario mensual BRUTO (en tu moneda local)'],
        s=2,
        label=conformity,
        alpha=0.65,
    )

ax.legend()
plt.ylabel("Salario bruto")
plt.xlabel("Salario neto")
plt.title('Relación entre salario neto y salario bruto')
plt.show()
# -

# ### Histograma
#
# Quizás notaron ese llamado a `plt.subplots`. Vamos a ver un poco de qué se trata mientras vemos como hacer histogramas.
#
# Tanto el salario neto como el salario bruto tienen soportes continuos, y demasiados valores diferentes. ¿Estaría bueno ver un histograma de cada uno no? Sería incluso mejor tenerlos lado a lado.

# +
df_submm = df[
    (df['Salario mensual NETO (en tu moneda local)'] < 1e6)
    & (df['Salario mensual BRUTO (en tu moneda local)'] < 1e6)
]

fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, dpi=150, figsize=(6.4 * 2, 4.8))

axes[0].hist(df_submm['Salario mensual BRUTO (en tu moneda local)'], bins=25)
axes[0].set_title("Salario bruto")
axes[0].set_xlabel("Salario")
axes[0].set_ylabel("Cantidad")

axes[1].hist(df_submm['Salario mensual NETO (en tu moneda local)'], bins=25)
axes[1].set_title("Salario neto")
axes[1].set_xlabel("Salario")
axes[1].set_ylabel("Cantidad")

plt.show()

# +
# ZZZZZZ
# -

plt.hist()

plt.figure(dpi=150)
df['¿Contribuís a proyectos open source?'].value_counts()[
    ["Sí", "No"]
].sort_index().plot(kind='pie', autopct='%1.0f%%', colors=['#AEB8AF', "#4AD172"])
plt.show()

plt.figure(dpi=150)
df['¿Contribuís a proyectos open source?'].value_counts()[
    ["Sí", "No"]
].sort_index().plot(kind='pie', autopct='%1.0f%%', colors=['#AEB8AF', "#4AD172"])
plt.title('¿Contribuís a proyectos open source?')
plt.ylabel("")
plt.show()


# ## Barplot

plt.figure(dpi=150)
df['¿Qué SO usás en tu laptop/PC para trabajar?'].value_counts().sort_values(
    ascending=False
).plot(kind='bar', color="dimgrey")
plt.ylabel("Cantidad")
plt.xlabel("Sistema operativo")
plt.title('¿Qué SO usás en tu laptop/PC para trabajar?')
plt.show()

# ## KDE plot

column = df['Salario mensual NETO (en tu moneda local)']
plt.figure(dpi=150)
column.plot(kind='kde', color="dimgrey", xlim=[column.min(), column.max()])
plt.title("Distribución del salario mensual NETO")
plt.xlabel("Salario mensual NETO")
plt.show()

column = df['Salario mensual NETO (en tu moneda local)']
plt.figure(dpi=150)
column.plot(kind='kde', color="dimgrey", xlim=[column.min(), column.max()])
plt.title("Distribución del salario mensual NETO")
plt.xlabel("Salario mensual NETO")
plt.ylabel("")
plt.show()

# Queremos sacar el eje Y por completo, una de las funciones de matplotlib es plt.yticks lo que nos permite setear dos listas:
# * La primer lista indicando los valores en donde queremos texto (en el caso anterior es 0, 2, 4, etc...)
# * La segunda indicando el texto que queremos en esos valores

column = df['Salario mensual NETO (en tu moneda local)']
plt.figure(dpi=150)
column.plot(kind='kde', color="dimgrey", xlim=[column.min(), column.max()])
plt.title("Distribución del salario mensual NETO")
plt.xlabel("Salario mensual NETO")
plt.ylabel("")
plt.yticks([0], ["cero"])
plt.show()

# Si dejamos ambas listas vacias nos deshacemos del eje y

column = df['Salario mensual NETO (en tu moneda local)']
plt.figure(dpi=150)
column.plot(kind='kde', color="dimgrey", xlim=[column.min(), column.max()])
plt.title("Distribución del salario mensual NETO")
plt.xlabel("Salario mensual NETO")
plt.ylabel("")
plt.yticks([], [])
plt.show()

# Observamos que alguien en la encuesta puso que ganaba 2 millones, será por estan en moneda de otro país? Nos quedamos solo con Argentina

column = df[df["Estoy trabajando en"] == "Argentina"][
    'Salario mensual NETO (en tu moneda local)'
]
plt.figure(dpi=150)
column.plot(kind='kde', color="dimgrey", xlim=[column.min(), column.max()])
plt.title("Distribución del salario mensual NETO")
plt.xlabel("Salario mensual NETO")
plt.ylabel("")
plt.yticks([], [])
plt.show()

# Esto no soluciona el problema, al parecer a alguien le parecio gracioso decir que ganaba 2 millones, vamos a limitar el salario neto a 500mil

column = df[df["Estoy trabajando en"] == "Argentina"][
    'Salario mensual NETO (en tu moneda local)'
].where(lambda x: x < 500000)
plt.figure(dpi=150)
column.plot(kind='kde', color="dimgrey", xlim=[column.min(), column.max()])
plt.title("Distribución del salario mensual NETO")
plt.xlabel("Salario mensual NETO")
plt.ylabel("")
plt.yticks([], [])
plt.show()

# ## Scatter plot

plt.figure(dpi=150)
df[df["Estoy trabajando en"] == "Argentina"].plot(
    'Salario mensual NETO (en tu moneda local)',
    'Salario mensual BRUTO (en tu moneda local)',
    kind='scatter',
    ax=plt.gca(),
    color="dimgrey",
)
plt.title("Comparación del salario NETO y BRUTO\npara Argentina")
plt.xlabel("Salario NETO")
plt.ylabel("Salario BRUTO")
plt.show()

plt.figure(dpi=150)
df[
    (df["Estoy trabajando en"] == "Argentina")
    & (df['Salario mensual NETO (en tu moneda local)'] < 500000)
    & (df['Salario mensual BRUTO (en tu moneda local)'] < 500000)
].plot(
    'Salario mensual NETO (en tu moneda local)',
    'Salario mensual BRUTO (en tu moneda local)',
    kind='scatter',
    ax=plt.gca(),
    color="dimgrey",
)
plt.title("Comparación del salario NETO y BRUTO\npara Argentina")
plt.xlabel("Salario NETO")
plt.ylabel("Salario BRUTO")
plt.show()

# ## Histograma

plt.figure(dpi=150)
df['¿De qué % fue el ajuste total?'].plot(kind='hist', color="dimgrey")
plt.title("Distribución del ajuste porcentual\npor inflación para 2019")
plt.ylabel("Frecuencia")
plt.xlabel("% del ajuste de inflación de 2019")
plt.show()

plt.figure(dpi=150)
df['¿De qué % fue el ajuste total?'].plot(kind='hist', bins=30, color="dimgrey")
plt.title("Distribución del ajuste porcentual\npor inflación para 2019")
plt.ylabel("Frecuencia")
plt.xlabel("% del ajuste de inflación de 2019")
plt.show()

# Matplotlib nos permite graficar plots uno encima del otro, usa la misma variable global

plt.figure(dpi=150)
df['¿De qué % fue el ajuste total?'].plot(kind='hist', bins=30, color="dimgrey")
plt.title("Distribución del ajuste porcentual\npor inflación para 2019")
plt.ylabel("Frecuencia")
plt.xlabel("% del ajuste de inflación de 2019")
plt.axvline(x=40, color="darkred")
plt.show()

plt.figure(dpi=150)
df['¿De qué % fue el ajuste total?'].plot(kind='hist', bins=30, color="dimgrey")
plt.title("Distribución del ajuste porcentual\npor inflación para 2019")
plt.ylabel("Frecuencia")
plt.xlabel("% del ajuste de inflación de 2019")
plt.axvline(x=53.8, color="darkred", label="Inflación según INDEC")
plt.legend()
plt.show()

# # Seaborn


import seaborn as sns

sns.set()

# Seaborn tambien usa matplotlib al igual que pandas, por lo que todas las funciones de matplotlib tambien le sirven

# ## Distribution plots

# ### Countplot
#
# El countplot es la forma que tiene seaborn de hacer gráficos de barras, permitiendo dividirlo de distintas formas.

sns.countplot(x="Trabajo de", data=df)
plt.ylabel("Cantidad")
plt.xlabel("Profesión")
plt.title("Cantidad de encuestados según profesión")
plt.show()

# Podemos usar el parametro order para indicar el orden en el que lo queremos pero tambien cuales profesiones queremos

sns.countplot(
    x="Trabajo de", data=df, order=df["Trabajo de"].value_counts().iloc[:20].index
)
plt.ylabel("Cantidad")
plt.xlabel("Profesión")
plt.title("Cantidad de encuestados según profesión")
plt.xticks(rotation=90)
plt.show()

# ### Violinplot

df['Tiene gente a cargo'] = df['¿Gente a cargo?'] > 0
plt.title("Salario NETO según si tiene gente a cargo\nen Argentina")
sns.violinplot(
    data=df[
        (df["Estoy trabajando en"] == "Argentina")
        & (df['Salario mensual NETO (en tu moneda local)'] < 500000)
    ],
    y='Salario mensual NETO (en tu moneda local)',
    x='Tiene gente a cargo',
    palette=['#D17049', "#89D15E"],
)
plt.ylabel("Salario NETO")
plt.show()

df['Tiene gente a cargo'] = df['¿Gente a cargo?'] > 0
plt.title("Distribución del salario NETO según\nsi tiene gente a cargo en Argentina")
sns.violinplot(
    data=df[
        (df["Estoy trabajando en"] == "Argentina")
        & (df['Salario mensual NETO (en tu moneda local)'] < 500000)
    ],
    y='Salario mensual NETO (en tu moneda local)',
    x='Tiene gente a cargo',
    palette=['#D17049', "#89D15E"],
)
plt.ylabel("Salario NETO")
plt.xticks([False, True], ["No", "Sí"])
plt.show()

# ### Boxplot

plt.title("Distribución del salario NETO según\nsi tiene gente a cargo en Argentina")
sns.boxplot(
    data=df[
        (df["Estoy trabajando en"] == "Argentina")
        & (df['Salario mensual NETO (en tu moneda local)'] < 500000)
    ],
    y='Salario mensual NETO (en tu moneda local)',
    x='Tiene gente a cargo',
    palette=['#D17049', "#89D15E"],
)
plt.ylabel("Salario NETO")
plt.xticks([False, True], ["No", "Sí"])
plt.show()

# ## Comparison plots
# ### Heatmap

cooccurrence = pd.pivot_table(
    df,
    '¿Gente a cargo?',
    'Cómo creés que está tu sueldo con respecto al último semestre',
    '¿Qué tan conforme estás con tu sueldo?',
    'count',
).sort_index()
plt.ylabel("Cómo creés que está tu sueldo con respecto al último semestre", fontsize=9)
sns.heatmap(cooccurrence.reindex([4, 3, 2, 1]), square=True, cmap="Wistia")

# ## Regression plots
# ### Regplot

sns.regplot(
    data=df,
    x='Salario mensual NETO (en tu moneda local)',
    y='Salario mensual BRUTO (en tu moneda local)',
)

# ## Relational plots
# ### Lineplot

sns.lineplot(
    data=df, x='Años de experiencia', y='Salario mensual BRUTO (en tu moneda local)'
)

# # Paletas de colores
#
# La elección de colores no es una decisión menor. Permite el mapeo de números a una representación visual y distinción entre grupos distintos. Hay mucha literatura sobre los criterios que debe cumplir una paleta de colores, algunos criterios relevantes son:
# - que no sean sensitivas a deficiencias visuales
# - el ordenamiento de los colores debe ser el mismo para todas las personas intuitivamente
# - la interpolación percibida debe corresponderse con el mapa escalar subyacente

# ## Taxonomía de paletas

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# ### Cualitativas
# Se usan para representar colecciones de clases discretas sin orden. Los colores no tienen un ordenamiento, por lo tanto no s on apropiados para mapearse a un valor escalar.

sns.palplot(sns.color_palette('pastel'))
plt.show()
sns.palplot(sns.color_palette('colorblind'))
plt.show()
sns.palplot(sns.color_palette('muted'))
plt.show()

# ### Secuenciales
# Son casi monocromaticas, van de un color altamente saturado hacia distintos niveles de saturación más baja. Se suele aumentar la luminancia a medida que decrece la saturación, de modo que la paleta termina en colores cercanos al blanco. Se usa para representar información que tiene un ordenamiento.

sns.palplot(sns.color_palette('Blues'))
plt.show()
sns.palplot(sns.color_palette('Blues_r'))
plt.show()
sns.palplot(sns.color_palette('Blues_d'))
plt.show()

# #### Cubehelix
# Es un sistema de paletas de colores que tienen un crecimiento/decrecimiento lineal en brillo y alguna variacion de tono. Lo cual implica que **se preserva la información al convertirse a blanco y negro**. Es ideal para **imprimir**.

sns.palplot(sns.color_palette("cubehelix", 12))
plt.show()

# ### Divergentes
# Tienen dos componentes principales de color, transicionando entre ambos pasando por un color poco saturado (blanco, amarillo). Se suelen usar para representar vvalores esacalares con un valor significativo cerca de la mediana.
#
# Es importante tratar de no usar rojo y verde.

sns.palplot(sns.color_palette('coolwarm', 7))
plt.show()
sns.palplot(sns.color_palette('RdBu_r', 7))
plt.show()
sns.palplot(sns.color_palette('BrBG', 7))
plt.show()

# ### Cíclicas
# Tienen dos componentes principales de color, que se encuentran en el medio y extremos en un color poco saturado. Se usan para valores que ciclan.

# +
sns.palplot(sns.color_palette("hls", 12))
plt.show()

# brillo percibido mas uniformemente
sns.palplot(sns.color_palette("husl", 12))
plt.show()
# -

# ## Referencias
#
# - [Documentación de seaborn](http://seaborn.pydata.org/tutorial/color_palettes.html)
# - [Diverging Color Maps for Scientific Visualization - Kenneth Moreland](https://cfwebprod.sandia.gov/cfdocs/CompResearch/docs/ColorMapsExpanded.pdf)
# - [XKCD color survey](https://blog.xkcd.com/2010/05/03/color-survey-results/)
# - [Subtleties of colors series](https://earthobservatory.nasa.gov/blogs/elegantfigures/2013/08/05/subtleties-of-color-part-1-of-6/)
# - [Documentación de matplotlib](https://matplotlib.org/tutorials/colors/colormaps.html)

# # Pandas profiling


# +
from pandas_profiling import ProfileReport

report = ProfileReport(
    df, title='Encuesta de sueldos sysarmy 2020.02', explorative=True, lazy=False
)
# -

report.to_notebook_iframe()
