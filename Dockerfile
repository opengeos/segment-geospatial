FROM jupyter/base-notebook:latest
LABEL maintainer="Qiusheng Wu"
LABEL repo="https://github.com/opengeos/segment-geospatial"

USER root
RUN apt-get update -y && apt-get install libgl1 sqlite3 -y

USER 1000
RUN mamba install -c conda-forge leafmap localtileserver segment-geospatial sam2==0.4.1 -y && \
    pip install -U segment-geospatial jupyter-server-proxy && \
    mamba update -c conda-forge sqlite -y && \
    jupyter server extension enable --sys-prefix jupyter_server_proxy && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN mkdir ./examples
COPY /docs/examples ./examples

ENV PROJ_LIB='/opt/conda/share/proj'
ENV JUPYTER_ENABLE_LAB=yes

ARG LOCALTILESERVER_CLIENT_PREFIX='proxy/{port}'
ENV LOCALTILESERVER_CLIENT_PREFIX=$LOCALTILESERVER_CLIENT_PREFIX

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
