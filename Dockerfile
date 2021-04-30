FROM ubuntu:focal

MAINTAINER CrossNox <imermet@fi.uba.ar>

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

EXPOSE 8888

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --assume-yes && apt-get install -y software-properties-common

RUN add-apt-repository universe && \
    apt update --assume-yes && \
    apt install -y --no-install-recommends git curl gcc g++ cmake zlib1g-dev libz-dev

RUN apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl \
    graphviz

ENV NVM_DIR /root/.nvm
ENV NODE_VERSION stable

RUN curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm use stable \
    && DEFAULT_NODE_VERSION=$(nvm version default) \
    && ln -sf /root/.nvm/versions/node/$DEFAULT_NODE_VERSION/bin/node /usr/bin/nodejs \
    && ln -sf /root/.nvm/versions/node/$DEFAULT_NODE_VERSION/bin/node /usr/bin/node \
    && ln -sf /root/.nvm/versions/node/$DEFAULT_NODE_VERSION/bin/npm /usr/bin/npm

ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV PYTHON_VERSION 3.7.9
RUN set -ex \
    && curl https://pyenv.run | bash \
    && pyenv update \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash

RUN pip3 install setuptools

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip3 install -r requirements.txt

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly plotlywidget
RUN jupyter lab build

VOLUME /notebooks

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--notebook-dir=/notebooks"]
