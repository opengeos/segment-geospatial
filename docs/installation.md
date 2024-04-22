# Installation

## Install from PyPI

**segment-geospatial** is available on [PyPI](https://pypi.org/project/segment-geospatial/). To install **segment-geospatial**, run this command in your terminal:

```bash
pip install segment-geospatial
```

## Install from conda-forge

**segment-geospatial** is also available on [conda-forge](https://anaconda.org/conda-forge/segment-geospatial). If you have
[Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your computer, you can install segment-geospatial using the following commands. It is recommended to create a fresh conda environment for **segment-geospatial**. The following commands will create a new conda environment named `geo` and install **segment-geospatial** and its dependencies:

```bash
conda create -n geo python
conda activate geo
conda install -c conda-forge mamba
mamba install -c conda-forge segment-geospatial
```

If your system has a GPU, but the above commands do not install the GPU version of pytorch, you can force the installation of the GPU version of pytorch using the following command:

```bash
mamba install -c conda-forge segment-geospatial "pytorch=*=cuda*"
```

Samgeo-geospatial has some optional dependencies that are not included in the default conda environment. To install these dependencies, run the following command:

```bash
mamba install -c conda-forge groundingdino-py segment-anything-fast
```

## Install from GitHub

To install the development version from GitHub using [Git](https://git-scm.com/), run the following command in your terminal:

```bash
pip install git+https://github.com/opengeos/segment-geospatial
```

## Use docker

You can also use [docker](https://hub.docker.com/r/giswqs/segment-geospatial/) to run segment-geospatial:

```bash
docker run -it -p 8888:8888 giswqs/segment-geospatial:latest
```

To enable GPU for segment-geospatial, run the following command to run a short benchmark on your GPU:

```bash
docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

The output should be similar to the following:

```text
Run "nbody -benchmark [-numbodies=<numBodies>]" to measure performance.
        -fullscreen       (run n-body simulation in fullscreen mode)
        -fp64             (use double precision floating point values for simulation)
        -hostmem          (stores simulation data in host memory)
        -benchmark        (run benchmark to measure performance)
        -numbodies=<N>    (number of bodies (>= 1) to run in simulation)
        -device=<d>       (where d=0,1,2.... for the CUDA device to use)
        -numdevices=<i>   (where i=(number of CUDA devices > 0) to use for simulation)
        -compare          (compares simulation results running once on the default GPU and once on the CPU)
        -cpu              (run n-body simulation on the CPU)
        -tipsy=<file.bin> (load a tipsy model file for simulation)

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

> Windowed mode
> Simulation data stored in video memory
> Single precision floating point simulation
> 1 Devices used for simulation
GPU Device 0: "Turing" with compute capability 7.5

> Compute 7.5 CUDA device: [Quadro RTX 5000]
49152 bodies, total time for 10 iterations: 69.386 ms
= 348.185 billion interactions per second
= 6963.703 single-precision GFLOP/s at 20 flops per interaction
```

If you encounter the following error:

```text
nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown.
```

Try adding `sudo` to the command:

```bash
sudo docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

Once everything is working, you can run the following command to start a Jupyter Notebook server:

```bash
docker run -it -p 8888:8888 --gpus=all giswqs/segment-geospatial:latest
```
