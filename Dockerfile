FROM ubuntu:focal

MAINTAINER CrossNox <imermet@fi.uba.ar>

EXPOSE 8888

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --assume-yes && apt-get install -y software-properties-common

RUN add-apt-repository universe && \
    apt update --assume-yes && \
    apt install -y --no-install-recommends python3 python3-dev python3-virtualenv python3-pip git curl

RUN pip3 install setuptools
ENV NVM_DIR /root/.nvm
RUN curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install node

COPY . /practicas

WORKDIR /practicas
RUN pip3 install -r requirements.txt
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly plotlywidget
RUN jupyter lab build

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
