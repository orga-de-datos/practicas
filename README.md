# Prácticas
Contenido para clases prácticas. Notebooks, guías, etc

# Guía de instalación del entorno
La siguiente guía supone que se está usando ubuntu.

## Dependencias del sistema
Primero tenemos que asegurarnos de tener instaladas las dependencias del sistema.

```bash
# apt update
# apt install python3 python3-dev python3-venv python3-pip
```

## [Opcional] Crear un virtualenv
[venv](https://docs.python.org/3/library/venv.html) es un modulo que permite crear entornos virtuales livianos de python. Esto es muy util para que no haya conflictos entre dependencias requeridas en distintos proyectos/entornos.

Vamos a crear un entorno virtual
```bash
cd <ubicacion deseada>
virtualenv -p python3 venv
```

Luego, cada vez que querramos usarlo, tendremos que activarlo
```bash
source <ubicacion del virtualenv>/venv/bin/activate
```

Para mayor comodidad, se puede agregar un alias con ese comando a `~/.alias` o `~/.bashrc`.

## Instalar dependencias de python
Dentro de la carpeta del repo
```bash
pip3 install -r requirements.txt
```

## Jupytext
Ver PRs de notebooks es dificil, porque son JSONs. Si fueron ejecutados, además puede haber imágenes en base64 o tablas
o texto largo. Para ayudar con este problema podemos usar [jupytext](https://jupytext.readthedocs.io/en/latest/index.html).
Está en las dependencias incluido. En resumen, cada notebook queda ligado a un script con la metadata necesaria, y ese `.py` es el que se versiona.

Para poder usarlo (asi como otras extensiones), es necesario tener instalado [node](https://nodejs.org/en/).

Para activar la extension
```bash
jupyter serverextension enable jupytext
jupyter lab build
```

## ToC
Para ver mejor el árbol de contenidos:
```bash
jupyter labextension install @jupyterlab/toc
jupyter lab build
```

## Jupyter widgets
Para ver gráficos interactivos
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter lab build
```

## Levantar jupyter lab
Estando en la carpeta del repo
```bash
jupyter lab
```

_Nota_: Si en los pasos anteriores usamos un `virtualenv`, hay que activarlo primero.

## Cerrar jupyter lab
Con doble `Ctrl+C` se apaga sin confirmación. Con `Ctrl+C` una sola vez, pedirá confirmación.

_Nota_: Esto se debe hacer en la terminal donde hemos dejado levantado el server.

# Entorno de desarrollo

## Pre-commit
Para mejorar la calidad del codigo subido, usamos [pre-commit](https://pre-commit.com), que instala varios hooks que se ejecutan
antes de un commit o push. Estos hooks estan definidos en `.pre-commit-config.yaml`.

```bash
$ pre-commit install
$ pre-commit install -t pre-push
```

# Github Actions
En la carpeta `.github/workflows` hay archivos `.yml` que definen los pipelines de gitub actions.

## `tex.yml`
Este pipeline compila el archivo `guia/guia.tex` a `pdf` y lo sube al branch `guia-ejs`. Solo se ejecuta al pushear a `master`.

Esto nos asegura que los últimos cambios queden disponibilizados. Poríamos olvidarnos de pushear el pdf, que además es un archivo binario-ish que puede ser medio molesto tener en PRs.

## `notebooks.yml`
Este pipeline recorre la carpeta `clases`, pasa cada `.py` creados con `jupytext` a `.ipynb` y los ejecuta. Luego copia los notebooks ejecutados y el archivo `requirements.txt` al branch `notebooks`.

Si algún notebook no se renderizara, se pueden revisar los logs de github actions a ver cual fue el error.

_Nota_: los notebooks deberian poder correr de punta a punta sin errores para poder ser pasados a `.ipynb`. Si alguna celda se **requiere** que falle, hay que recordar ponerle el tag `raises-exception` a dicha celda.

