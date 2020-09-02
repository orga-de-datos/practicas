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
import seaborn as sns
from matplotlib import pyplot as plt

# +
from pandas_profiling import ProfileReport

# # Descargamos la data
#
# Vamos a utilizar los datos de la [encuesta de sueldos 2020.02](https://sysarmy.com/blog/posts/resultados-de-la-encuesta-de-sueldos-2020-2/) de [sysarmy](https://sysarmy.com/es/).

GSPREADHSEET_DOWNLOAD_URL(gid=1980145505)

# +
GSPREADHSEET_DOWNLOAD_URL = (
    "https://docs.google.com/spreadsheets/d/{gid}/export?format=csv&id={gid}".format
)

SYSARMY_2020_2_GID = '1FxzaPoS0AkN8E_-aeobpr7FHAAy8U7vWcGE7PY4kJmQ'

df = pd.read_csv(GSPREADHSEET_DOWNLOAD_URL(gid=SYSARMY_2020_2_GID), skiprows=9)
# -

# ## Una pequeña preview

pd.options.display.max_columns = None
df.head()

# # Pandas plotting

# ## Pie chart

df['¿Contribuís a proyectos open source?'].value_counts().sort_index().plot(
    kind='pie', autopct='%1.0f%%'
)
plt.show()

# ## Barplot

df['¿Qué SO usás en tu laptop/PC para trabajar?'].value_counts().sort_values(
    ascending=False
).plot(kind='bar')
plt.show()

# ## KDE plot

column = df['Salario mensual NETO (en tu moneda local)']
column.plot(kind='kde', xlim=[column.min(), column.max()])
plt.show()

# ## Scatter plot

df.plot(
    'Salario mensual NETO (en tu moneda local)',
    'Salario mensual BRUTO (en tu moneda local)',
    kind='scatter',
)
plt.show()

# ## Histograma

df['¿De qué % fue el ajuste total?'].plot(kind='hist')
plt.show()

# # Seaborn


sns.set()

# ## Categorical plots

# ### Violinplot

df['Tiene gente a cargo'] = df['¿Gente a cargo?'] > 0
sns.violinplot(
    data=df, y='Salario mensual NETO (en tu moneda local)', x='Tiene gente a cargo'
)

# ### Boxplot

sns.boxplot(
    data=df, y='Salario mensual NETO (en tu moneda local)', x='Tiene gente a cargo'
)

# ## Matrix plots
# ### Heatmap

cooccurrence = pd.pivot_table(
    df,
    '¿Gente a cargo?',
    'Cómo creés que está tu sueldo con respecto al último semestre',
    '¿Qué tan conforme estás con tu sueldo?',
    'count',
).sort_index()
sns.heatmap(cooccurrence, square=True)

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


report = ProfileReport(
    df, title='Encuesta de sueldos sysarmy 2020.02', explorative=True, lazy=False
)
# -

report.to_notebook_iframe()
