<div align="center">
  <img src="README.assets/openparf_logo.web.jpeg">
</div>
<p align="center">
  🕹 An open-source FPGA placement and routing framework built upon <a href="https://github.com/pytorch/pytorch">PyTorch</a>
</p>

<p align="center">
  <strong>Flexible</strong> &nbsp;|&nbsp; <strong>Efficient</strong> &nbsp;|&nbsp; <strong>Extensible</strong>
</p>

## Table of Contents

- [News](#news)
- [Features](#features)
- [Demo](#demo)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Benchmarks](#running-benchmarks)
- [Documentation](#documentation)
- [Resources](#resources)
- [Team](#team)
- [Publications](#publications)
- [License](#license)

## News

- 🎉 (2024/02/02) We are excited to announce the release of [OpenPARF 2.0](https://github.com/PKU-IDEA/OpenPARF/releases/tag/2.0.0)! This release includes multi-die FPGA placement support, Flexshelf architecture format, and various improvements.

## Features

- **Multi-Electrostatic-based Placement**: Achieves optimal results for routed wirelength and placement speed
- **Multi-Die Support**: Advanced support for multi-die FPGA placement with SLL count optimization
- **Comprehensive Unit Support**: Handles LUT, FF, DSP, BRAM, IO, Distributed RAM, and Shift Register
- **High Performance**: 0.4-12.7% improvement in routed wirelength and 2X+ speedup in placement
- **GPU Acceleration**: Leverages CUDA for significant performance gains

## Demo

The following visualizations show the electrostatic fields in benchmark `ISPD2016/FPGA06`:

|            **LUT**            |           **FF**            |            **DSP**            |            **BRAM**             |
| :---------------------------: | :-------------------------: | :---------------------------: | :-----------------------------: |
| ![LUT](README.assets/lut.gif) | ![ff](README.assets/ff.gif) | ![dsp](README.assets/dsp.gif) | ![BRAM](README.assets/bram.gif) |

## Quick Start

### Prerequisites
- Python 3.7+
- C++14 compatible compiler
- PyTorch 1.7.1
- CUDA 11.0 (optional)
- Gurobi 9.5 (optional)

### Installation

```bash
# Create conda environment
mamba create --name openparf python=3.7
mamba activate openparf

# Install dependencies
mamba install cmake boost bison
mamba install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
pip install hummingbird-ml pyyaml networkx tqdm

# Build OpenPARF
git clone --recursive https://github.com/PKU-IDEA/OpenPARF.git
mkdir build && cd build
cmake ../OpenPARF -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DPYTHON_EXECUTABLE=$(which python)
make -j8 && make install
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Running Benchmarks

```bash
cd <installation directory>
python openparf.py --config unittest/regression/ispd2016/FPGA01.json
```

See [BENCHMARKS.md](BENCHMARKS.md) for more information about benchmarks and evaluation.

## Documentation

- [Installation Guide](INSTALL.md)
- [Benchmark Guide](BENCHMARKS.md)
- [Architecture Customization](ARCHITECTURE.md)

## Resources

- [ISPD 2016 Contest](http://www.ispd.cc/contests/16/ispd2016_contest.html)
- [ISPD 2017 Contest](http://www.ispd.cc/contests/17/)

## Team

OpenPARF is maintained by [PKU-IDEA Lab](https://github.com/PKU-IDEA) at Peking University, supervised by [Prof. Yibo Lin](https://yibolin.com/).

- [Jing Mai](https://magic3007.github.io/), [Jiarui Wang](https://tomjerry213.github.io/) and [Yibai Meng](https://www.mengyibai.com/) composed the initial release.

- Runzhe Tao developed and integrated the LEAPS for multi-die placement in OpenPARF.

- [Jing Mai](https://magic3007.github.io/), [Jiarui Wang](https://tomjerry213.github.io/), Yifan Chen, [Zizheng Guo](https://guozz.cn), Xun Jiang, [Yun Liang](https://ericlyun.github.io), [Yibo Lin](https://yibolin.com/) developed OpenPARF 3.0 with macro placement support.

## Publications

See our [publications list](PUBLICATIONS.md) for academic papers about OpenPARF.

## License

This software is released under BSD 3-Clause License. See [LICENSE](LICENSE) for details.
