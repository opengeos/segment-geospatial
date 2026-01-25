# Installation

## Install with pixi (Recommended)

Installing **segment-geospatial** with `uv` or `pip` can be challenging on some platforms (especially Windows) due to complicated pytorch/cuda dependencies and numpy version conflicts. [Pixi](https://pixi.prefix.dev/latest) is recommended to avoid these issues, as it provides faster and more reliable dependency resolution than conda or mamba.

### 1) Install Pixi

#### Linux/macOS (bash/zsh)

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Close and re-open your terminal (or reload your shell) so `pixi` is on your `PATH`. Then confirm:

```bash
pixi --version
```

#### Windows (PowerShell)

Open **PowerShell** (preferably as a normal user, Admin not required), then run:

```powershell
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

Close and re-open PowerShell, then confirm:

```powershell
pixi --version
```

---

### 2) Create a Pixi project

Navigate to a directory where you want to create the project and run:

```bash
pixi init geo
cd geo
```

---

### 3) Configure `pixi.toml`

Open `pixi.toml` in the `geo` directory and replace its contents with the following depending on your system.

If you have an NVIDIA GPU with CUDA, run `nvidia-smi` to check the CUDA version.

#### For GPU with CUDA 12.x:

```toml
[workspace]
channels = ["https://prefix.dev/conda-forge"]
name = "geo"
platforms = ["linux-64", "win-64"]

[system-requirements]
cuda = "12.0"

[dependencies]
python = "3.12.*"
pytorch-gpu = ">=2.7.1,<3"
segment-geospatial = ">=1.2.0"
sam3 = ">=0.1.0.20251211"
jupyterlab = "*"
ipykernel = "*"
libopenblas = ">=0.3.30"
```

#### For GPU with CUDA 13.x:

```toml
[workspace]
channels = ["https://prefix.dev/conda-forge"]
name = "geo"
platforms = ["linux-64", "win-64"]

[system-requirements]
cuda = "13.0"

[dependencies]
python = "3.12.*"
pytorch-gpu = ">=2.7.1,<3"
segment-geospatial = ">=1.2.0"
sam3 = ">=0.1.0.20251211"
jupyterlab = "*"
ipykernel = "*"
```

#### For CPU:

```toml
[workspace]
channels = ["https://prefix.dev/conda-forge"]
name = "geo"
platforms = ["linux-64", "win-64"]

[dependencies]
python = "3.12.*"
pytorch-cpu = ">=2.7.1,<3"
segment-geospatial = ">=1.2.0"
sam3 = ">=0.1.0.20251211"
jupyterlab = "*"
ipykernel = "*"
libopenblas = ">=0.3.30"
```

---

### 4) Install the environment

From the `geo` folder:

```bash
pixi install
```

This step may take several minutes on first install depending on your internet connection and system.

---

### 5) Verify PyTorch + CUDA

If you have a NVIDIA GPU with CUDA, run the following command to verify the PyTorch + CUDA installation:

```bash
pixi run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'))"
```

Expected output should be like this:

-   `PyTorch: 2.7.1` (or higher)
-   `CUDA available: True`
-   `GPU: NVIDIA RTX 4090` (your GPU name)

If CUDA is `False`, check:

-   `nvidia-smi` works in your terminal
-   NVIDIA driver is up to date

---

### 6) Request access to SAM 3 (Optional)

To use SAM 3, you will need to request access by filling out this form on Hugging Face at <https://huggingface.co/facebook/sam3>. Once your request has been approved, run the following command in the terminal to authenticate:

```bash
pixi run hf auth login
```

After authentication, you can download the SAM 3 model from Hugging Face:

```bash
pixi run hf download facebook/sam3
```

**Important Note**: SAM 3 currently requires a NVIDIA GPU with CUDA support. You won't be able to use SAM 3 if you have a CPU only system ([source](https://github.com/facebookresearch/sam3/issues/164)). You will get an error message like this: `Failed to load model: Torch not compiled with CUDA enabled`.

---

### 7) Start Jupyter Lab

To start using segment-geospatial in Jupyter Lab:

```bash
pixi run jupyter lab
```

This will open Jupyter Lab in your default browser. You can now create a new notebook and start using segment-geospatial!

---

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
