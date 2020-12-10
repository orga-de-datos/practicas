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
#     display_name: Python 3 (venv)
#     language: python
#     name: python3
# ---

# +
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.animation as mpl_animation
import matplotlib
from IPython.display import HTML, Markdown

# -

sns.set()


# # Red neuronal multiclase
#
# En el caso que quisieramos entrenar una red neuronal en un problema con mas de dos clases, tenemos que cuidar algunos parametros:
# - la activacion de la ultima capa debe ser `softmax`
#     - da la probabilidad de cada clase
#     - la suma de las probabilidades de todas las clases debe dar 1
# - la cantidad de neuronas de la ultima capa debe ser 3
# - la perdida ya no es mas `binary_crossentropy`, sino `categorical_crossentropy`

# ## Dataset
#
# Creamos un dataset linealmente separable

# +
def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


X, y = make_classification(
    n_features=2,
    n_redundant=0,
    n_classes=3,
    n_informative=2,
    random_state=117,
    n_clusters_per_class=1,
    class_sep=0.90,
)

X = rotate(X, degrees=30)

plt.figure(dpi=150)
sns.scatterplot(X[:, 0], X[:, 1], hue=y, palette='Dark2')
plt.show()

# +
import keras

in_l = keras.layers.Input(shape=(2,))
d1 = keras.layers.Dense(128, activation='relu')(in_l)
out_l = keras.layers.Dense(3, activation='softmax')(d1)

m = keras.models.Model(inputs=[in_l], outputs=[out_l])

m.compile('sgd', loss='categorical_crossentropy', metrics=['accuracy'])

h = m.fit(
    X,
    keras.utils.to_categorical(y),
    epochs=500,
    batch_size=32,
    verbose=0,
    validation_split=0.3,
)
# -

plt.figure(dpi=125, figsize=(12, 4))
plt.plot(h.history['loss'], label="loss")
plt.plot(h.history['val_loss'], label="validation loss")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.figure(dpi=125, figsize=(12, 6))
plt.plot(h.history['accuracy'], label="accuracy")
plt.plot(h.history['val_accuracy'], label="validation accuracy")
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

# +
from sklearn.metrics import confusion_matrix
import pandas as pd


def plot_confusion_matrix(y_true, y_pred):
    names = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, names)
    df_cm = pd.DataFrame(cm, names, names)

    plt.figure(dpi=100)
    plt.title("Matriz de confusion")
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g', square=True)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


preds = np.argmax(m.predict(X), axis=1)
plot_confusion_matrix(y, preds)

# +
plt.figure(dpi=200)
ax = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='Dark2')

G = 250

xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(*xlim, G), np.linspace(*ylim, G))
preds = m.predict(np.c_[xx.ravel(), yy.ravel()])
pred_classes = np.argmax(preds, axis=1)
z = pred_classes.reshape(xx.shape)
ax.contourf(xx, yy, z, alpha=0.4, cmap='Dark2')
ax.axis(False)

plt.plot()
