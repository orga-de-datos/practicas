# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
from matplotlib import pyplot as plt

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

# ## Pie chart

# Para hacer plots pandas usa matplotlib, por lo que podemos usar en la misma celda cosas relacionadas a esa libreria como _plt.show()_

df['¿Contribuís a proyectos open source?'].value_counts().sort_index().plot(
    kind='pie', autopct='%1.0f%%'
)
plt.show()

plt.figure(dpi=150)
df['¿Contribuís a proyectos open source?'].value_counts().sort_index().plot(
    kind='pie', autopct='%1.0f%%'
)
plt.show()

# ### DPI
#
# Son cantidad de pixeles por pulgada, las figuras por default tienen [6.4, 4.8] pulgadas y 100 dpi, por lo que van a tener 640x480 pixeles.

plt.figure(dpi=100, figsize=[15, 9])
df['¿Contribuís a proyectos open source?'].value_counts().sort_index().plot(
    kind='pie', autopct='%1.0f%%'
)
plt.show()

# Podemos conservar los DPIs agrandando la figura, pero como las fuentes estan en "pt" se van a quedar del mismo tamaño, relativo al dpi.
#
# __Recomendacion__: no cambiar el figsize si lo que se quiere es agrandar la figura, solo cambiarlo si adrede se quiere cambiar la forma.

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

# ### ¿Cómo sabe matplotlib a quien cambiarle todos esos atributos?
#
# Yo no tengo en ningun lado una variable con el gráfico al que le pida cambiar_titulo(grafico, "Titulo") o algo similar, cómo sabe aplicarselo a ese?
# Hay una variable global con el plot que estamos usando actualmente que nos permite esta magia, esta está en plt.gca()
#
# Cuando el plot se muestra esta variable se reinicia a un plot en blanco.
# Cuando llamamos a plt.figure estamos creando una nueva y pisando la que había.
#
# ### Colores
#
# Podemos usar colores en formato hexadecimal, RGB o alguno de [los colores que tienen nombre](https://matplotlib.org/3.1.0/gallery/color/named_colors.html).

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

# # Matplotlib

from matplotlib import pyplot as plt
import matplotlib

# Dejamos 150 dpi por default

matplotlib.rcParams['figure.dpi'] = 150


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

# # Pandas profiling


# +
from pandas_profiling import ProfileReport

report = ProfileReport(
    df, title='Encuesta de sueldos sysarmy 2020.02', explorative=True, lazy=False
)
# -

report.to_notebook_iframe()
