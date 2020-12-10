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


# # Perceptrón base
#
# Primero veamos la estructura mas basica posible: el perceptron base.

# ## Generacion de dataset
#
# Armaremos primero un dataset sintetico de dos clases linealmente separables. Esto quiere decir que podemos encontrar un hiperplano que separe el espacio de modo que cada clase quede en una sola region. Que sea linealmente separable nos sera importante para entender algunos detalles.

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
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
    class_sep=0.55,
)

X = rotate(X, degrees=30)

plt.figure(dpi=150)
sns.scatterplot(X[:, 0], X[:, 1], hue=y, palette='RdBu')
plt.show()
# -

# ## Modelo en numpy
#
# Hacer una red neuronal es numpy es, en lineas generales, una mala idea. Sirve de todos modos a propositos educativos, para que podamos jugar.
#
# ### Perceptron base
# Recordemos que nuestras features vive en $R^2$. Y recordemos:
#
# $$f(b + \sum_{i=1}^{n}x_i \times w_i)$$
#
# Donde:
# - $f$ es nuestra funcion de activacion
# - $b$ es el bias
# - $x_i$ es la i-esima feature de una observacion
# - $w_i$ es el peso de la red para las i-esimas features
#
# Como entrenaremos nuestro perceptron?
#
# ```text
# mientras no sea estable:
#     para cada par X_i, y_i:
#         yhat = f(X_i*w + b)
#         si y_hat != y_i:
#             actualizo w con X_i
#             actualizo b
# ```
#
# > estable: no se han actualizado los pesos durante todo un epoch

# +
epoch = 1

b = 0.0
w = np.zeros(2).T


def heaviside_step(x):
    return max(0.0, np.sign(x))


while True:
    stable = True
    print(f"epoch {epoch}")
    for (Xi, yi) in zip(X, y):
        yhat = heaviside_step(np.dot(Xi, w) + b)

        w += (yi - yhat) * Xi
        b += yi - yhat
        if (yi != yhat).any():
            stable = False
        # stable &= yi == yhat

    if stable:
        break

    epoch += 1


# -


def plot_hiperplane(w, b, X, y, title=None, _ax=None):
    ax = _ax
    if _ax is None:
        fig, ax = plt.subplots(dpi=150)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='RdBu', alpha=1, ax=ax)

    G = 500

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(*xlim, G), np.linspace(*ylim, G))
    z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    z = z.reshape(xx.shape)

    ax.contourf(xx, yy, z, alpha=0.1, cmap='RdBu')

    x_vals = np.array(xlim)
    y_vals = b - (w[0] / w[1]) * x_vals
    ax.plot(x_vals, y_vals, ':', color='Purple')

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)

    if title:
        ax.set_title(title)
    if _ax is None:
        plt.show()
    else:
        return ax


plot_hiperplane(w, b, X, y)
# + jupyter={"source_hidden": true}
display(
    Markdown(
        f"""Entonces, luego de 2 epochs es estable y ha encontrado un hiperplano separador $$y = {b} - {w[0]/w[1]:.2f}x$$"""
    )
)
# -
# Se puede demostrar la convergencia bajo ciertas condiciones.
#
# - [aca por ejemplo](https://www.cse.iitb.ac.in/~shivaram/teaching/old/cs344+386-s2017/resources/classnote-1.pdf)

# # Learning rate
# Ahora tenemos nuestro perceptron base. Una pequenia (perdon, no tengo la enie mapeada (ni tildes)) modificacion que podemos hacerle es pesar $X_i$ al momento de actualizar $w$ y $b$.
#
# ## Momento de reflexion
# Nos sirve de algo esto si iteramos en el mismo orden las observaciones y son linealmente separables nuestras clases?
#
# ## Dataset no-linealmente-separable
# Para ilustrar por que nos puede servir una tasa de aprendizaje, utilizemos un dataset no-linealmente-separable.

# +
X, y = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
    class_sep=0.35,
)
X = rotate(X, degrees=-30)

# X, y = make_moons(n_samples=1000, random_state=0, noise=0.1)

plt.figure(dpi=150)
sns.scatterplot(X[:, 0], X[:, 1], hue=y, palette='RdBu')
plt.show()


# -


def plot_lr(X, y, lr=0.1, epochs=50):
    fig, ax = plt.subplots(dpi=150)

    def _plot_hiperplane(frame):
        ax.clear()
        epoch, b, w = frame
        s = " + " if w[0] < 0 else " - "
        plot_hiperplane(
            w,
            b,
            X,
            y,
            f"epoch {epoch} | $y = {b:.2f}{s}{np.abs(w[0]/w[1]):.2f}x$",
            _ax=ax,
        )
        return

    def train_with_learning_rate(X, y, lr=0.1, epochs=50):
        max_epochs = epochs
        epoch = 1

        alpha = lr
        b = 0.0
        w = np.zeros(2).T

        frames = [(epoch, np.copy(b), np.copy(w))]

        while True:
            if epoch >= max_epochs:
                break
            stable = True
            for (Xi, yi) in zip(X, y):
                yhat = max(0.0, np.sign(np.dot(Xi, w) + b))

                w += (yi - yhat) * Xi * alpha
                b += (yi - yhat) * alpha
                stable &= yi == yhat

            if stable:
                break

            epoch += 1
            frames.append((epoch, np.copy(b), np.copy(w)))

        return frames

    def init_anim():
        pass

    anim = mpl_animation.FuncAnimation(
        fig,
        _plot_hiperplane,
        frames=train_with_learning_rate(X, y, lr, epochs),
        interval=200,
        init_func=init_anim,
    )
    plt.close()
    matplotlib.rc('animation', html='jshtml')
    display(HTML(anim.to_jshtml()))


plot_lr(X, y, lr=0.001, epochs=50)

plot_lr(X, y, lr=1.0, epochs=50)

plot_lr(X, y, lr=2.0, epochs=50)

# # Funciones de activacion
#
# Hasta aqui, teniamos dos clases y queriamos predecir la clase. No siempre sera el caso. Podriamos tener un problema mutliclase, podriamos querer predecir $P(y_i=1|X_i)$. Por otro lado vimos en la teorica que si quisieramos resolver un problema no-linealmente-separable como el de recien, deberiamos _combinar_ perceptrones mediante funciones de activacion no lineales.
#
# Pero que funciones de activacion podemos usar? Cuales son las propiedades que deben cumplir?

# ## Identidad
#
# La función identidad está dada por $$f(x) = x$$

# + jupyter={"source_hidden": true}
x = np.linspace(-5, 5)
y = x

plt.figure()
sns.lineplot(x, y)
plt.title("identidad")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.show()
# -

# ## Escalón
#
# La función escalón está dada por $$f(x)= \left\{ \begin{array}{lcc}
#              0 &   si  & x \lt 0 \\
#              \\ 1 &  si & x \geq 0
#              \end{array}
#    \right. $$
#
# Te suena de algun lado?

# + jupyter={"source_hidden": true}
x = np.linspace(-5, 5)
y = np.maximum(0.0, np.sign(x))

plt.figure()
sns.lineplot(x, y)
plt.title("escalón")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.show()
# -

# ## Tangente hiperbólica
#
# La función tangente hiperbólica está dada por $$f(x) = \tanh{x}$$

# + jupyter={"source_hidden": true}
x = np.linspace(-5, 5)
y = np.tanh(x)

plt.figure()
sns.lineplot(x, y)
plt.title("tanh")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.show()
# -

# ## ReLU (Rectified Linear Unit)
#
# La función Relu viene dada por $$f(x) = \max(0, x) = \left\{ \begin{array}{lcc}
#              0 &   si  & x \lt 0 \\
#              \\ x &  si & x \geq 0
#              \end{array}
#    \right. $$

# + jupyter={"source_hidden": true}
x = np.linspace(-5, 5)
y = np.maximum(0, x)

plt.figure()
sns.lineplot(x, y)
plt.title("ReLU")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.show()
# -

# # Primera red neuronal (MLP)
#
# Resolvamos el ejemplo de antes! Veamos nuestro bello dataset, primero.

# +
X, y = make_moons(n_samples=1000, random_state=117, noise=0.1)

plt.figure(dpi=200)
sns.scatterplot(X[:, 0], X[:, 1], hue=y, palette='RdBu')
plt.show()
# -

# Con un poco de buen pulso y una birome podriamos encontrar una curva que separe las dos clases, no? Pero las impresiones a color estan caras, asi que haremos una red neuronal para encontrar la curva y graficarla.
#
# Como dijimos antes, hacerlo en numpy podria funcionar para propositos educativos (les invitamos a intentarlo!) pero es poco practico par la Vida Real™. Hay variedades de frameworks para construir redes neuronales, pero mostraremos dos de los mas conocidos.

# # Keras
#
# [Keras](https://keras.io) se describe como:
#
# > Deep learning for humans.
#
# > Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. It also has extensive documentation and developer guides.
#
# Se puede acceder con tensorflow como [`tf.keras`](https://www.tensorflow.org/guide/keras?hl=es). [TensorFlow](https://www.tensorflow.org/?hl=es) es otro framework, desarrollado por Google.

# +
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# +
import keras

in_l = keras.layers.Input(shape=(2,))
d1 = keras.layers.Dense(32, activation='relu')(in_l)
out_l = keras.layers.Dense(1, activation='sigmoid')(d1)

m = keras.models.Model(inputs=[in_l], outputs=[out_l])
# -

m.compile('sgd', loss='binary_crossentropy', metrics=['accuracy'])

m.summary()

h = m.fit(X, y, epochs=500, batch_size=16, verbose=0, validation_split=0.3)

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
plt.figure(dpi=200)

xrange = X[:, 0].max() - X[:, 0].min()
yrange = X[:, 1].max() - X[:, 1].min()

xlim = (X[:, 0].min() - xrange * 0.05, X[:, 0].max() + xrange * 0.05)
ylim = (X[:, 1].min() - yrange * 0.05, X[:, 1].max() + yrange * 0.05)

G = 250

xx, yy = np.meshgrid(np.linspace(*xlim, G), np.linspace(*ylim, G))
z = m.predict(np.c_[xx.ravel(), yy.ravel()])[:, 0]
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, alpha=0.4, cmap='RdBu')
ax = sns.scatterplot(X[:, 0], X[:, 1], hue=y, palette='RdBu', alpha=1)
ax.set(yticklabels=[])
ax.set(xticklabels=[])

plt.show()
# -


# # PyTorch
#
# [PyTorch](https://pytorch.org) se define como
#
# > An open source machine learning framework that accelerates the path from research prototyping to production deployment.
#
# Fue principalmente desarrollado por Facebook. Es de mas bajo nivel (el equivalente seria mas cercado a tensorflow) pero permite gran flexibilidad, esta muy actualizado y tiene mucho uso en academia.

import torch
from progressbar import progressbar
from sklearn.model_selection import ShuffleSplit

# +
hidden_layers = 128
epochs = 500
val_split = 0.3

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idxs = np.arange(len(X))
np.random.shuffle(idxs)
size = int(np.floor(val_split * len(X)))
train_idxs = idxs[size:]
val_idxs = idxs[:size]

train_sampler = torch.utils.data.SubsetRandomSampler(train_idxs)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_idxs)
# -

train = torch.utils.data.TensorDataset(
    torch.from_numpy(X).float(), torch.from_numpy(y).float()
)
train_loader = torch.utils.data.DataLoader(train, batch_size=16, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train, batch_size=16, sampler=valid_sampler)

model = torch.nn.Sequential(
    torch.nn.Linear(2, hidden_layers),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layers, 1),
    torch.nn.Sigmoid(),
)

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# +
losses = []
accuracies = []
val_losses = []
val_accuracies = []

for t in progressbar(range(epochs), max_value=epochs):
    train_loss = 0
    accuracy = 0
    val_loss = 0
    val_accuracy = 0

    model.train()

    for data, target in train_loader:
        optimizer.zero_grad()
        batch_y_pred = model(data)
        loss = loss_fn(batch_y_pred, target.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        accuracy += ((batch_y_pred.reshape(-1,) > 0.5) == target).sum()

    model.eval()

    for data, target in val_loader:
        batch_y_pred = model(data)
        loss = loss_fn(batch_y_pred, target.reshape(-1, 1))
        val_loss += loss.item() * data.size(0)
        val_accuracy += ((batch_y_pred.reshape(-1,) > 0.5) == target).sum()

    losses.append(train_loss / float(train_loader.sampler.indices.size))
    accuracies.append(accuracy / float(train_loader.sampler.indices.size))
    val_losses.append(val_loss / float(val_loader.sampler.indices.size))
    val_accuracies.append(val_accuracy / float(val_loader.sampler.indices.size))
# -

plt.figure(dpi=125, figsize=(12, 4))
plt.plot(losses, label="loss")
plt.plot(val_losses, label="val_loss")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.figure(dpi=125, figsize=(12, 6))
plt.plot(accuracies, label="accuracy")
plt.plot(val_accuracies, label="val_accuracy")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

# +
plt.figure(dpi=200)

xrange = X[:, 0].max() - X[:, 0].min()
yrange = X[:, 1].max() - X[:, 1].min()

xlim = (X[:, 0].min() - xrange * 0.05, X[:, 0].max() + xrange * 0.05)
ylim = (X[:, 1].min() - yrange * 0.05, X[:, 1].max() + yrange * 0.05)

G = 250

xx, yy = np.meshgrid(np.linspace(*xlim, G), np.linspace(*ylim, G))

mesh_X = np.c_[xx.ravel(), yy.ravel()]

z = model(torch.from_numpy(mesh_X).float()).detach().numpy()[:, 0]
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, alpha=0.4, cmap='RdBu')
ax = sns.scatterplot(X[:, 0], X[:, 1], hue=y, palette='RdBu', alpha=1)
ax.set(yticklabels=[])
ax.set(xticklabels=[])

plt.show()
# -

# # Funciones de pérdida/costo
#
# Momento... que es ese `binary_crossentropy`?
#
# Es la **funcion de perdida**. Tienen la forma $f(y, y_{hat}) \to R$, se trata de minimizar este error. Al aplicar descenso por gradiente (o la regla de la cadena en backpropagation) usamos este valor para ajustar los parametros del modelo.
#
# La implementacion de distintas funciones de activacion dependen del framework. La funcion de costo se debe decidir en base al entorno del problema que estemos tratando de resolver.


# ## Algunas funciones de perdida comunes
# Mas o menos lo mismo de siempre, pero por las dudas...
#
# ### Clasificacion
#
# #### Binary crossentropy
# Sirve cuando se tiene un problema de clasificacion binario.
#
# $$H_p = - \frac{1}{N} \sum_{i=1}^{N} y_i \log{(p(y_i))} + (1 - y_i) \log{(1-p(y_i))}$$
#
# - [keras](https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-function)
# - [Explicacion visual](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
#
# #### Categorical crossentropy
# Sirve cuando se tiene un problema de clasificacion multiclase.
#
# $$C = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} 1_{y_i \in C_c} \log p_{modelo}[y_i \in C_c] $$
#
# En el caso que el problema no sea multi-label, podemos simplificar la expresion a
#
# $$C = -\frac{1}{N} \sum_{i=1}^{N} \log p_{modelo}[y_i \in C_{y_i}] $$
#
# - [keras](https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-function)
#
# ### Regresion
#
# #### MSE
# $$\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2$$
#
# - [keras](https://keras.io/api/losses/regression_losses/#meansquarederror-function)
#
# #### MAE
# $$\text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|$$
#
# - [keras](https://keras.io/api/losses/regression_losses/#meanabsoluteerror-function)
#
# #### MAPE
# $$\text{MAPE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} 100 * \frac{\left| y_i - \hat{y}_i \right|}{y_i}$$
#
#
# - [keras](https://keras.io/api/losses/regression_losses/#meanabsolutepercentageerror-function)

# # Optimizadores
#
# Si nos fijamos en el modelo de Keras, tenemos otro parametro: `optimizer`. Es el metodo para optimizar. Vimos en la teorica SGD, pero hay otros con distintas propiedaes.
