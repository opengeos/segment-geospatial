FROM jupyter/base-notebook:latest
LABEL maintainer="Qiusheng Wu"
LABEL repo="https://github.com/opengeos/segment-geospatial"

RUN mamba install -c conda-forge leafmap localtileserver segment-geospatial -y && \
    pip install -U segment-geospatial jupyter-server-proxy && \
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
