# Installation

## Install from PyPI

**segment-geospatial** is available on [PyPI](https://pypi.org/project/segment-geospatial/). To install **segment-geospatial**, run this command in your terminal:

```bash
pip install segment-geospatial
```

## Install from conda-forge

**segment-geospatial** is also available on [conda-forge](https://anaconda.org/conda-forge/segment-geospatial). If you have
[Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your computer, you can install segment-geospatial using the following command:

```bash
conda install -c conda-forge segment-geospatial
```

It is recommended to create a fresh conda environment for **segment-geospatial**. The following command will create a new conda environment named `geo` and install **segment-geospatial** and its dependencies:

```bash
conda install -n base mamba -c conda-forge
mamba create -n geo segment-geospatial python -c conda-forge
```

## Install from GitHub

To install the development version from GitHub using [Git](https://git-scm.com/), run the following command in your terminal:

```bash
pip install git+https://github.com/opengeos/segment-geospatial
```
