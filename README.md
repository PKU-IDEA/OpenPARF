# OpenPARF

🕹 OpenPARF is an open-source FPGA placement and routing framework build upon the deep learning toolkit [PyTorch](https://github.com/pytorch/pytorch). It is designed to be flexible, efficient, and extensible.

<!-- toc -->

- [OpenPARF](#openparf)
  - [More About OpenPARF](#more-about-openparf)
    - [A Multi-Electrostatic-based FPGA P\&R Framework](#a-multi-electrostatic-based-fpga-pr-framework)
    - [Reference Flow](#reference-flow)
    - [Demo](#demo)
  - [Prerequisites](#prerequisites)
    - [Build from Source](#build-from-source)
      - [Install Dependencies](#install-dependencies)
      - [Install Gurobi (Optional)](#install-gurobi-optional)
    - [Build with Docker](#build-with-docker)
      - [Docker Image](#docker-image)
        - [Using pre-built images](#using-pre-built-images)
        - [Building the image yourself](#building-the-image-yourself)
      - [Running the Docker Image](#running-the-docker-image)
      - [Entering the Docker Container](#entering-the-docker-container)
  - [Build and Install OpenPARF](#build-and-install-openparf)
    - [Get the OpenPARF Source](#get-the-openparf-source)
    - [Install OpenPARF](#install-openparf)
    - [Adjust Build Options (Optional)](#adjust-build-options-optional)
  - [Getting Started](#getting-started)
    - [ISPD 2016/2017 Benchmarks](#ispd-20162017-benchmarks)
      - [Obtaining Benchmarks](#obtaining-benchmarks)
      - [Linking Benchmarks](#linking-benchmarks)
      - [Running the Benchmarks](#running-the-benchmarks)
      - [Adjust Benchmark Options (Optional)](#adjust-benchmark-options-optional)
      - [More Advanced Usages](#more-advanced-usages)
        - [Running Benchmarks in Batches](#running-benchmarks-in-batches)
        - [Vivado Flow for Placement Evaluation](#vivado-flow-for-placement-evaluation)
  - [Resources](#resources)
  - [Releases and Contributing](#releases-and-contributing)
  - [The Team](#the-team)
  - [Publications](#publications)
  - [License](#license)

<!-- tocstop -->

## More About OpenPARF

OpenPARF is an open-source framework for FPGA rapid placement and routing, which can run on both CPU and GPU. OpenPARF provides a number of APIs to enable researchers to quickly prototype their own FPGA algorithms and evaluate their performance on real FPGA hardware.

At a granular level, OpenPARF is a framework that consists of the following components:

|    **Component**     |                                  **Description**                                  |
| :------------------: | :-------------------------------------------------------------------------------: |
|      `openparf`      |                        The core placement and routing tool                        |
|    `openparf.ops`    | A collection of operators that allow the implementation of various P&R algorithms |
| `openparf.placement` |                   A set of APIs for performing placement tasks                    |
|  `openparf.routing`  |                    A set of APIs for performing routing tasks                     |
| `openparf.py_utils`  |              Provides other utility functions for Python convenience              |

OpenPARF provides a compilation stack to integrate your placement and routing algorithms into operators that can be used in Python. You can extend OpenPARF as needed, making it the ideal environment for FPGA optimization enthusiasts.

Elaborating Further:

### A Multi-Electrostatic-based FPGA P&R Framework

OpenPARF is a powerful FPGA P&R framework that utilizes a multi-electrostatic-based approach to achieve optimal results with respect to routed wirelength and placement speed. OpenPARF supports the following features:

```
- A wide range of logical unit types including LUT, FF, DSP, BRAM, IO, Distributed RAM, and Shift Register
- Support for SLICEL and SLICEM CLB heterogeneous types
- Clock routing constraints
- ...
```

OpenPARF has proven to be a powerful tool, outperforming the state-of-the-art academic FPGA P&R tools in terms of wired length and placement speed. With a `0.4-12.7%` improvement in routed wirelength and a speedup of over `2X` in placement, OpenPARF is a highly efficient FPGA P&R framework that offers optimized results with minimal manual intervention.

### Reference Flow

![overflow](README.assets/overflow.jpg)

### Demo

The following are the visualization for electrostatic fields in benchmark `ISPD2016/FPGA06`.

|            **LUT**            |           **FF**            |            **DSP**            |            **BRAM**             |
| :---------------------------: | :-------------------------: | :---------------------------: | :-----------------------------: |
| ![LUT](README.assets/lut.gif) | ![ff](README.assets/ff.gif) | ![dsp](README.assets/dsp.gif) | ![BRAM](README.assets/bram.gif) |

## Prerequisites

OpenPARF is written in C++ and Python. The following are the prerequisites for building OpenPARF.

- Python 3.7 or above.
- C++ compiler with **C++14** support, recommended to use GCC 7.5. Other version may also work, but have not been tested.
- **PyTorch 1.7.1**. Other version may also work, but have not been tested. Please refer to the [next section](#install-dependencies) to install PyTorch through conda environment.
- **Gurobi 9.5** (optional, if the router is compiled). Other version may also work, but have not been tested. Please make sure to obtain a valid license and follow the installation instructions provided on the Gurobi website.
  Note that only the router uses gurobi in OpenPARF. If you do not need a router, you can leave gurobi uninstalled and set `ENABL_ROUTER` to `OFF` when compiling OpenPARF.
- **NVIDIA CUDA 11.0** (optional, if compiled with CUDA support). Other versions may also work, but have not been tested. If CUDA is found, the project can run on the GPU implementation, otherwise it will only run on the CPU implementation.

We have provided two ways to build OpenPARF,

- one is to [build from source](#build-from-source),
- and the other is to [build with docker](#build-with-docker).

### Build from Source

#### Install Dependencies

We highly recommend installing an [Anaconda](https://www.anaconda.com/data-science-platform#download-section) or [Mamba](https://github.com/mamba-org/mamba) environment. You will get controlled dependency versions regardless of your Linux distro.

```bash
# * create and activate conda environment
mamba create --name openparf python=3.7
mamba activate openparf

# * common packages
mamba install cmake boost bison

# * Pytorch 1.7.1. Other version may also work, but have not been tested.
mamba install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch

# * python packages
pip install hummingbird-ml pyyaml
```

#### Install Gurobi (Optional)

Permit me to illuminate the exquisite process of installing Gurobi, using the recommended Gurobi 9.5.1 as a quintessential example:

1. Download the [gurobi9.5.1_linux64.tar.gz](https://packages.gurobi.com/9.5/gurobi9.5.1_linux64.tar.gz), and then extract it to a location of your choosing, aptly referred to as `<your Gurobi home>`.
2. Next, you must obtain a license for Gurobi. To do so, you must first create an account on the Gurobi website. Once you have created an account, you will be able to request an license. Once you receive your license, you will be able to download it from the Gurobi website. The license will be a file with the extension `.lic`. Save this file to a location of your choosing, aptly referred to as `<your Gurobi license path>`.
3. Finally, you must set the following environment variables. You can do so by adding the following lines to your `~/.bashrc` file.

```bash
export GUROBI_HOME="<your Gurobi home>/linux64"
# For example,
# export GUROBI_HOME="/home/jingmai/softwares/gurobi951/linux64"
export GRB_LICENSE_FILE="<your Gurobi license path>"
# For example,
# export GRB_LICENSE_FILE="/home/jingmai/licenses/gurobi.lic"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

Now, please go to the [Build and Install OpenPARF](#build-and-install-openparf) section to build and install OpenPARF.

### Build with Docker

#### Docker Image

##### Using pre-built images

You can also pull a pre-built docker image from Docker Hub.

```bash
docker pull magic3007/openparf:1.0
```

##### Building the image yourself

You can also build the docker image yourself. The dockerfile is located in `docker/openparf.dockerfile`. To build the image, run the following command:

```bash
cd <source directory>/docker
docker build . -t openparf:1.0 -f openparf.dockerfile
```

#### Running the Docker Image

We recommend that you have the following two directories/files on your host before running docker:

1. `<source directory on host>`: The directory where you store the OpenPARF source code on host. See [Get the OpenPARF Source](#get-the-openparf-source) for more details.
2. `<benchmark directory on host>`: The directory where you store the ISPD 2016/2017 benchmarks on host. See [Obtaining Benchmarks](#obtaining-benchmarks) for more details.

**Without CUDA Support**
To run the docker image without CUDA support, run the following command:

```bash
docker run -itd --restart=always --network host -e TERM=$TERM \
  --name openparf \
  -v /etc/localtime:/etc/localtime:ro \
  -v <project directory on host>:/root/OpenPARF \
  -v <benchmark directory on host>:/root/benchmarks \
  openparf:1.0 \
  /bin/bash;
```

**With CUDA Support**
To run the docker image with CUDA support, run the following command:

```bash
docker run -itd --restart=always --network host -e TERM=$TERM \
  --name openparf \
  --gpus all \
  -v /etc/localtime:/etc/localtime:ro \
  -v <project directory on host>:/root/OpenPARF \
  -v <benchmark directory on host>:/root/benchmarks \
  openparf:1.0 \
  /bin/bash;
```

#### Entering the Docker Container

Once the docker image is running, you can enter the docker container by running the following command:

```bash
docker exec -it openparf /bin/bash
```

Within the docker container:

- the OpenPARF source code will be located in `/root/OpenPARF`.
- the ISPD 2016/2017 benchmarks will be located in `/root/benchmarks`.

**NOTE** that though we have gurobi 9.5.1 installed in docker (located in `/opt/gurobi`), due to permission reasons, if you need router, you still need to get gurobi license and place it as `/root/gurobi.lic`.

Now, please go to the [Build and Install OpenPARF](#build-and-install-openparf) section to build and install OpenPARF.

## Build and Install OpenPARF

### Get the OpenPARF Source

```bash
git clone --recursive https://github.com/PKU-IDEA/OpenPARF.git
```

If you have already clone the repository, e.g., the repository is mounted in the docker container,
you can skip this step.

### Install OpenPARF

Assuming that OpenAPRF is a subfolder in the current directory, i.e., `./OpenPARF` is the path to the OpenPARF source code.

```bash
mkdir build
cd build
cmake ../OpenPARF -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DPYTHON_EXECUTABLE=$(which python) -DPython3_EXECUTABLE=$(which python) -DCMAKE_INSTALL_PREFIX=<installation directory>
make -j8
make install
```

Where `<installation directory>` is the directory where you want to install OpenPARF (e.g., `../install`).

### Adjust Build Options (Optional)

You can adjust the configuration of cmake variables optionally (without buiding first), by doing the following.

- `CMAKE_INSTALL_PREFIX`: The directory where you want to install OpenPARF (e.g., `../install`).
- `CMAKE_BUILD_TYPE`: The build type, can be `Release` or `Debug`. Default is `Release`.
- `USE_CCACHE`: Whether to use ccache to speed up compilation. Default is `OFF`.
- `ENABLE_ROUTER`: Whether to compile the router. Default is `ON`. If you do not need a router, you can set it to `OFF`.

## Getting Started

### ISPD 2016/2017 Benchmarks

#### Obtaining Benchmarks

To obtain the benchmarks, you can download the benchmark zip files from the provided Google Drive links. There are separate links for ISPD 2016 and ISPD 2017 FPGA Placement Benchmarks.

- [ISPD 2016 FPGA Placement Benchmarks](https://drive.google.com/file/d/1kzg0NfEmJvwzhJADPE_Q0UjS6UpVCMZZ/view?usp=share_link)

- [ISPD 2017 FPGA Placement Benchmarks](https://drive.google.com/file/d/1Uf9qIZ8WL_jk03sIlAoS9dIrvYH3d1pz/view?usp=sharing)

> <details>
> <summary> 💡 Toggle to see how to download files from Google Drive in command line</summary>
>
> 1. Click on the download button in the browser (e.g. Chrome)
>
> 2. Copy the download link from the browser
>
> 3. Use the curl command in the terminal to download the file
>
> ```bash
> curl <url> --output <output filename>
> ```
>
> </details>

Once the files have downloaded, extract their contents to the `<benchmark directory>` folder. Under this directory, you should then have two new directories, one for ISPD 2016 and another for ISPD 2017. Remember to keep these benchmark directories separate as OpenPARF supports ISPD2016 and ISPD2017 benchmarks.

```bash
<benchmark directory>
├── ispd2016
├── ispd2017
```

#### Linking Benchmarks

Link ispd2016 and ispd2017 folders under `<installation directory>/benchmarks` by soft link.

```bash
ln -s <benchmark directory>/ispd2016 <installation directory>/benchmarks/ispd2016
ln -s <benchmark directory>/ispd2017 <installation directory>/benchmarks/ispd2017
```

#### Running the Benchmarks

To run the benchmarks, navigate to the installation directory and execute the following command:

```bash
cd <installation directory>
python openparf.py --config unittest/regression/ispd2016/FPGA01.json
```

Note that the `openparf.py` script requires the `--config` option to specify the configuration file to use. The appropriate configuration file for the benchmark should exist in the corresponding directory (`unittest/regression/ispd2016/`) before running the command. Other benchmarks can also be run by selecting the appropriate configuration from their respective directory such as `unittest/regression/ispd2016` and `unittest/regression/ispd2017`.

It is essential to ensure all dependencies and the Python environment have been correctly set up before running the command. Once everything is in order, the benchmark will commence, and the outcomes will be sent to the output directory.

#### Adjust Benchmark Options (Optional)

OpenPARF allows configuration of benchmark parameters using JSON. Default parameters can be found in `openparf/params.json`, while users can use `--config` to pass custom parameters which will override defaults. For example:

```bash
python openparf.py --config unittest/regression/ispd2016/FPGA01.json
```

The parameter configuration in `unittest/regression/ispd2016/FPGA01.json` will override the defaults in `openparf/params.json`.

Common modifiable parameters include

- `gpu`: Enable GPU acceleration or not. Default is 0.
- `result_dir`: Specify the directory to save P&R results.
- `route_flag`: Enable router or not. Default is 0.

For more detailed information on parameters, please see the `description` field in `openparf/params.json`.

#### More Advanced Usages

More advanced usages are available to customize the benchmark run.

##### Running Benchmarks in Batches

We also provide scripts to run ISPD2016 and ISPD2017 in batches. Navigate to the installation directory and execute the following commands:

```bash
cd <installation directory>
<source directory>/scripts/run_ispd2016_benchmark.sh # ispd 2016
<source directory>/scripts/run_ispd2017_benchmark.sh # ispd 2017
```

The results can be found in `<installation directory>/../ispd2016_log` and `<installation directory>/../ispd2017_log`, respectively. Please refer to the script for specific configurations.

##### Vivado Flow for Placement Evaluation

If you are looking to evaluate a placement algorithm, you can do so directly with OpenPARF. Alternatively, you can import the placement results from OpenPARF into Vivado for evaluation. This is particularly useful if you are competing in the ISPD [2016](https://www.ispd.cc/contests/16/Flow.txt)/[2017](https://www.ispd.cc/contests/17/Flow.txt) contest and want to use the Vivado flow for placement evaluation. The official websites for ISPD offer instructions on how to load the placement results into the Vivado flow.

When you import the OpenPARF placement result file into Vivado, it will be located in `<result dir>/<benchmark name>.pl` (e.g., `results/FPGA01/FPGA01.pl`). Keep in mind that the `<result dir>` and `<benchmark name>` are parameters set within the JSON configuration.

**NOTE** that if you want the evaluate the placement via Vivado, you can disable the routing stage in OpenPARF by simply setting the `route_flag` to 0 before running the tool.

## Resources

- [ISPD 2016: Routability-Driven FPGA Placement Contest](http://www.ispd.cc/contests/16/ispd2016_contest.html)
  - Official ISPD 2016 FPGA Placement benchmarks can be downloaded from [here](https://www.ispd.cc/contests/17/downloads/2016/index.html).
- [ISPD 2017: Clock-Aware FPGA Placement Contest](http://www.ispd.cc/contests/17/)

## Releases and Contributing

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core library in a different direction than you might be aware of.

## The Team

OpenPARF is maintained by [PKU-IDEA Lab](https://github.com/PKU-IDEA) at Peking University, supervised by [Prof. Yibo Lin](https://yibolin.com/).

- [Jing Mai](https://magic3007.github.io/), [Jiarui Wang](https://tomjerry213.github.io/) and [Yibai Meng](https://www.mengyibai.com/) composed the initial release.

## Publications

- [Jing Mai](https://magic3007.github.io/), [Jiarui Wang](https://tomjerry213.github.io/), [Zhixiong Di](http://www.dizhixiong.cn/), [Guojie Luo](https://scholar.google.com/citations?user=8-mT29YAAAAJ&hl=en), [Yun Liang](https://ericlyun.github.io/) and [Yibo Lin](https://yibolin.com/), "**OpenPARF: An Open-Source Placement and Routing Framework for Large-Scale Heterogeneous FPGAs with Deep Learning Toolkit**", _International Conference on ASIC (ASICON)_, 2023. [[paper]](https://arxiv.org/abs/2306.16665)
- [Jing Mai](https://magic3007.github.io/), [Yibai Meng](https://www.mengyibai.com/), [Zhixiong Di](http://www.dizhixiong.cn/), and [Yibo Lin](https://yibolin.com/), “**Multi-electrostatic FPGA placement considering SLICEL-SLICEM heterogeneity and clock feasibility**,” in _Proceedings of the 59th ACM/IEEE Design Automation Conference (DAC)_, San Francisco California: ACM, Jul. 2022, pp. 649–654. doi: 10.1145/3489517.3530568. [[paper]](https://doi.org/10.1145/3489517.3530568)
- [Jiarui Wang](https://tomjerry213.github.io/), [Jing Mai](https://magic3007.github.io/), [Zhixiong Di](http://www.dizhixiong.cn/), and [Yibo Lin](https://yibolin.com/), “**A Robust FPGA Router with Concurrent Intra-CLB Rerouting**,” in _Proceedings of the 28th Asia and South Pacific Design Automation Conference (ASP-DAC)_, Tokyo Japan: ACM, Jan. 2023, pp. 529–534. doi: 10.1145/3566097.3567898. [[paper]](https://doi.org/10.1145/3566097.3567898)
<details><summary>CLICK ME to show the bibtex.</summary>

<p>

```bibtex
@inproceedings{PLACE_ASICON23_Mai_OpenPARF,
  title         = {OpenPARF: An Open-Source Placement and Routing Framework for Large-Scale Heterogeneous FPGAs with Deep Learning Toolkit},
  author        = {Jing Mai and Jiarui Wang and Zhixiong Di and Guojie Luo and Yun Liang and Yibo Lin},
  booktitle     = {International Conference on ASIC (ASICON)}
  year          = {2023},
}

@inproceedings{PLACE_DAC22_Mai,
  title         = {Multi-electrostatic {FPGA} placement considering {SLICEL-SLICEM} heterogeneity and clock feasibility},
  author        = {Jing Mai and Yibai Meng and Zhixiong Di and Yibo Lin},
  booktitle     = {ACM/IEEE Design Automation Conference (DAC)},
  pages         = {649--654},
  year          = {2022},
}

@inproceedings{ROUTE_ASPDAC2023_Wang,
  title         = {{{A Robust FPGA Router with Concurrent Intra-CLB Rerouting}}},
  author        = {Jiarui Wang and Jing Mai and Zhixiong Di and Yibo Lin},
  booktitle     = {IEEE/ACM Asia and South Pacific Design Automation Conference (ASPDAC)},
  pages         = {529--534},
  year          = {2023}
}
```

</p>
</details>

## License

This software is released under BSD 3-Clause License, Please refer to [LICENSE](./LICENSE) for details.
