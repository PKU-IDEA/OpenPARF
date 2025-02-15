# Installation Guide

## Table of Contents

- [Build from Source](#build-from-source)
  - [Install Dependencies](#install-dependencies)
  - [Install Gurobi (Optional)](#install-gurobi-optional)
- [Build with Docker](#build-with-docker)
  - [Docker Image Options](#docker-image-options)
  - [Running the Docker Container](#running-the-docker-container)
  - [Enter Container](#enter-container)
- [Build and Install OpenPARF](#build-and-install-openparf)
  - [Get the Source](#get-the-source)
  - [Install](#install)
  - [Build Options (Optional)](#build-options-optional)

OpenPARF provides two ways to build and install:
- Build from source
- Build with Docker

## Build from Source

### Install Dependencies

We recommend using Anaconda/Mamba for dependency management:

```bash
# Create and activate conda environment
mamba create --name openparf python=3.7
mamba activate openparf

# Install common packages
mamba install cmake boost bison

# Install PyTorch 1.7.1
mamba install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch

# Install Python packages
pip install hummingbird-ml pyyaml networkx tqdm
```

### Install Gurobi (Optional)

1. Download [gurobi9.5.1_linux64.tar.gz](https://packages.gurobi.com/9.5/gurobi9.5.1_linux64.tar.gz)
2. Extract to `<your Gurobi home>`
3. Obtain a license from Gurobi website
4. Set environment variables in `~/.bashrc`:

```bash
export GUROBI_HOME="<your Gurobi home>/linux64"
export GRB_LICENSE_FILE="<your Gurobi license path>"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

## Build with Docker

### Docker Image Options

#### Using Pre-built Images
```bash
docker pull magic3007/openparf:2.0
```

#### Building the Image Yourself
```bash
cd <source directory>/docker
docker build . -t openparf:2.0 -f openparf.dockerfile
```

### Running the Docker Container

**Without CUDA Support:**
```bash
docker run -itd --restart=always --network host -e TERM=$TERM \
  --name openparf \
  -v /etc/localtime:/etc/localtime:ro \
  -v <project directory on host>:/root/OpenPARF \
  -v <benchmark directory on host>:/root/benchmarks \
  openparf:2.0 \
  /bin/bash
```

**With CUDA Support:**
```bash
docker run -itd --restart=always --network host -e TERM=$TERM \
  --name openparf \
  --gpus all \
  -v /etc/localtime:/etc/localtime:ro \
  -v <project directory on host>:/root/OpenPARF \
  -v <benchmark directory on host>:/root/benchmarks \
  openparf:2.0 \
  /bin/bash
```

### Enter Container
```bash
docker exec -it openparf /bin/bash
```

## Build and Install OpenPARF

### Get the Source
```bash
git clone --recursive https://github.com/PKU-IDEA/OpenPARF.git
```

### Install
```bash
mkdir build
cd build
cmake ../OpenPARF -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DPYTHON_EXECUTABLE=$(which python) -DPython3_EXECUTABLE=$(which python) -DCMAKE_INSTALL_PREFIX=<installation directory>
make -j8
make install
```

### Build Options (Optional)

Adjustable CMake variables:
- `CMAKE_INSTALL_PREFIX`: Installation directory
- `CMAKE_BUILD_TYPE`: Release/Debug (Default: Release)
- `USE_CCACHE`: Enable ccache (Default: OFF)
- `ENABLE_ROUTER`: Enable router compilation (Default: OFF)