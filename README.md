# Prácticas
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/orga-de-datos/practicas/notebooks) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/orga-de-datos/practicas/blob/notebooks)

Este repositorio contiene los notebooks de las clases prácticas y la guía de ejercicios.

Consultar la siguiente sección para instalar correctamente el entorno y poder ejecutarlos localmente. Alternativamente, desde los badges se puede ejecutar desde el navegador utilizando [binder](https://mybinder.org) o [google colaboratory](https://colab.research.google.com).

# Guía de instalación del entorno
La siguiente guía supone que se está usando ubuntu. Se ha probado en una instalación limpia de ubuntu 20.10. Cualquier problema que pueda surgir, consultar por slack ([leer esto primero](#como-hacer-buenas-preguntas-por-slack)).

Leer atentamente **toda** la guía antes de proceder.

## Dependencias del sistema
Primero tenemos que asegurarnos de tener instaladas las dependencias del sistema.

```bash
# add-apt-repository universe
# apt update
# apt install --yes python3 python3-dev python3-virtualenv python3-pip git graphviz make cmake
```

## Clonar el repositorio
```bash
$ git clone https://github.com/orga-de-datos/practicas.git
$ cd practicas
$ git checkout notebooks
```

**IMPORTANTE**: el branch a utilizar es `notebooks`, no `master`.

## [Opcional] Crear un entorno virtual
[venv](https://docs.python.org/3/library/venv.html) es un modulo que permite crear entornos virtuales livianos de python. Esto es muy útil para que no haya conflictos entre dependencias requeridas en distintos proyectos/entornos.

Este paso es opcional pero **muy** recomendado hacerlo.

```bash
virtualenv -p python3 venv
source venv/bin/activate
```

Luego, cada vez que querramos usarlo, tendremos que activarlo
```bash
source <ubicacion del virtualenv>/venv/bin/activate
```

Para mayor comodidad, se puede agregar un alias con ese comando a `~/.bashrc`:
```bash
echo "alias venv_datos=\"source $(pwd)/venv/bin/activate\"" >> ~/.bashrc
source ~/.bashrc
```

__Nota__: para desactivar el virtualenv se usa el comando `deactivate`.

## Instalar dependencias de python
Dentro de la carpeta del repo (y con el virtualenv activado si se optó por crearlo):
```bash
pip install setuptools
pip install -r requirements.txt
```

## Instalar plugins
Vamos a usar el comando [labextension](https://jupyterlab.readthedocs.io/en/stable/user/extensions.html) para instalar algunos plugins usados durante la cursada. Previamente necesitamos instalar [node.js](https://nodejs.org/en/).

```bash
# apt install -y curl
$ curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash
$ source ~/.bashrc
$ nvm install node
$ source ~/.profile
```

### Jupyter widgets
Para ver gráficos interactivos
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter lab build
```

### Plotly
Para ver los gráficos hechos con plotly:
```bash
jupyter labextension install jupyterlab-plotly
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget
jupyter lab build
```

## Levantar jupyter lab
Al levantar `jupyter lab`, por defecto se toma la ubicación actual como base del árbol de navegación. Por tanto, desde el repositorio:

```bash
jupyter lab
```

__Nota__: Si en los pasos anteriores usamos un `virtualenv`, hay que activarlo primero.

### Cerrar jupyter lab
Con doble `Ctrl+C` se cierra sin confirmación. Con `Ctrl+C` una sola vez, pedirá confirmación.

__Nota__: Esto se debe hacer en la terminal donde hemos dejado levantado el server.

## Comprobación del entorno
Con estos pasos ejecutados, se debería abrir en el navegador la interfaz de jupyterlab. Probar ejecutar algún notebook.

# Levantar el entorno con docker/podman
```bash
docker run -p 127.0.0.1:9999:8888 -v .:/notebooks ghcr.io/crossnox/jupyter_datos:latest
```

## Build de la imagen
Desde el repo clonado:
```bash
docker build -t jupyter_datos .
docker run -p 127.0.0.1:8888:8888 -v .:/notebooks jupyter_datos
```

Luego abrir en un navegador la dirección `127.0.0.1:8888`.

__Nota__: si el puerto `8888` está en uso por otra aplicación, cambiarlo (en el primer 8888).

# Como hacer buenas preguntas por slack
[Este](http://www.bioinformaticscareerguide.com/2017/09/how-to-ask-good-programming-question_20.html) post y [este](https://stackoverflow.com/help/how-to-ask) articulo de ayuda de stackoverflow dan una buena idea de como pedir ayuda:
1. Buscar si no han preguntado algo similar antes.
2. Explicar que es lo que querés hacer.
3. Comentar que pasos seguiste, que errores te encontraste y como intentaste solucionarlos.
4. Incluir información relevante: sistema operativo, logs de errores, etc. Si envías código o logs, usar backticks (`) para formatear el texto. Por otro lado, texto >>>> una imagen: la imagen queda cortada, y quizás lo que está mas arriba a vos no te dice nada pero a quien lo lee, sí. Por otro lado, se puede copiar/pegar y buscar dentro del texto.
5. Dejar que te ayuden: si te piden información extra, es por algún motivo. Te puede parecer obvia la respuesta, pero es muy difícil saber que es lo que el otro no sabe. Si no funciona, dar detalles de por qué no funciona.
6. Cuando tengas la solución, escribila. Como ejercicio para reafirmar lo que acabas de aprender y para que otras personas puedan valerse de eso si llegan a tu misma situación.


# Entorno de desarrollo
Esta sección es relevante para docentes y quien quiera contribuir a este repositorio. Las secciones siguientes explican los pasos adicionales requeridos para contribuir y los jobs de `Github actions` que hacen el deploy de notebooks y la guía de ejercicios.

## Jupytext
Ver PRs de notebooks es difícil, porque son JSONs. Si fueron ejecutados, además puede haber imágenes en base64, tablas o texto largo incluido. En este sentido, ver el diff de un notebook es muy difícil desde github (a fecha de abril 2021). Como apuntamos a que los cambios sean colaborativos, lo ideal es que quien revise un PR (si fuera necesario) pueda entender los cambios desde el mismo PR.

Para ayudar con este problema podemos usar [jupytext](https://jupytext.readthedocs.io/en/latest/index.html). En resumen, cada notebook queda ligado a un script con la metadata necesaria, y ese `.py` es el que se versiona. Es decir, no pusheamos archivos `ipynb` sino archivos `py`.

Para poder usarlo (así como otras extensiones), es necesario tener instalado [node](https://nodejs.org/en/).

Para instalar el paquete y activar la extension
```bash
pip install jupytext==1.6.0
jupyter serverextension enable jupytext
jupyter lab build
```

__Nota__: si se decidió usar un virtualenv, activarlo previamente.

## Pre-commit
Para mejorar la calidad del código subido, usamos [pre-commit](https://pre-commit.com), que instala varios hooks que se ejecutan antes de un commit o push. Estos hooks están definidos en `.pre-commit-config.yaml`. El motivo de usar estas herramientas es facilitar la lectura del código y no preocuparse por formatearlo manualmente.

```bash
pip install pre-commit==2.8.2
pre-commit install
pre-commit install -t pre-push
```

__Nota__: si se decidió usar un virtualenv, activarlo previamente.

### Hooks definidos
#### Black
[Black](https://github.com/psf/black) es un formateador automático de código.

# Github Actions
En la carpeta `.github/workflows` hay archivos `.yml` que definen los pipelines de github actions. Cumplen distintas funciones, detalladas a continuación.

## `tex.yml`
Este pipeline compila el archivo `guia/guia.tex` a `pdf` y lo sube al branch `guia-ejs`. Solo se ejecuta al pushear a `master`.

El `pdf` generado es ignorado dentro del repo, para que no se suba accidentalmente, de este modo el repo es mas liviano, tenemos un build reproducible y evitamos meter archivos binarios al repo.

## `notebooks.yml`
Este pipeline recorre la carpeta `clases`, pasa cada `.py` creados con `jupytext` a `.ipynb` y los ejecuta. Luego copia los notebooks ejecutados y el archivo `requirements.txt` al branch `notebooks`. El archivo `requirements.txt` se pasa para poder correr los notebooks desde binder.

Si algún notebook no se renderizara, se pueden revisar los logs de github actions a ver cual fue el error.

Como recordatorio, todos los notebooks deberían poder correr de punta a punta sin errores para poder ser pasados a `.ipynb`. Si alguna celda se **requiere** que falle, hay que recordar ponerle el tag `raises-exception` a dicha celda. Esto además asegura que quienes quieran usar los notebooks para seguir las clases puedan hacerlo sin problemas.

## GH-actions: docker
Los tags del formato `jl-docker-v*` lanzan un gh-action que hace el build and push de la imagen de docker.
