# Prácticas
Contenido para clases prácticas. Notebooks, guías, etc

# Dev workflow
## Dependencias
Instalar las dependencias de dev:
```bash
pip install -r dev-requirements.txt
```

## Jupytext
Ver PRs de notebooks es dificil, porque son JSONs. Si fueron ejecutados, además puede haber imágenes en base64 o tablas 
o texto largo. Para ayudar con este problema podemos usar [jupytext](https://jupytext.readthedocs.io/en/latest/introduction.html). 
Está en las dependencias de desarrollo incluido.

## ToC
Para ver mejor el árbol de contenidos:
```bash
jupyter labextension install @jupyterlab/toc
```

## Levantar jupyter lab
```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab [--port=xxxx]
```

## Pre-commit
Para mejorar la calidad del codigo subido. [pre-commit](https://pre-commit.com) instala varios hooks que se ejecutan 
antes de un commit o push. Estos hooks estan definidos en `.pre-commit-config.yaml`.

```bash
$ pre-commit install
$ pre-commit install -t pre-push
```

