FROM python:3.7.9-slim-buster

MAINTAINER CrossNox <imermet@fi.uba.ar>

EXPOSE 8888

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --assume-yes && apt-get upgrade --assume-yes
RUN apt-get install -y --no-install-recommends git curl gcc g++ cmake zlib1g-dev libz-dev

RUN apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl graphviz

RUN apt-get purge --auto-remove -yqq  \
        && apt-get autoremove -yqq --purge \
        && apt-get clean \
        && rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

ENV NVM_DIR /root/.nvm
ENV NODE_VERSION stable

RUN curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm use stable \
    && DEFAULT_NODE_VERSION=$(nvm version default) \
    && ln -sf /root/.nvm/versions/node/$DEFAULT_NODE_VERSION/bin/node /usr/bin/nodejs \
    && ln -sf /root/.nvm/versions/node/$DEFAULT_NODE_VERSION/bin/node /usr/bin/node \
    && ln -sf /root/.nvm/versions/node/$DEFAULT_NODE_VERSION/bin/npm /usr/bin/npm \
    && nvm cache clear

RUN pip3 install setuptools --no-cache-dir

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip3 install -r requirements.txt --no-cache-dir

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly plotlywidget
RUN jupyter lab build && jupyter lab clean && jlpm cache clean

VOLUME /notebooks

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--notebook-dir=/notebooks"]
