# Benchmark Guide

## Table of Contents

- [ISPD 2016/2017 Benchmarks](#ispd-20162017-benchmarks)
  - [Obtaining Benchmarks](#obtaining-benchmarks)
  - [Setting Up Benchmarks](#setting-up-benchmarks)
  - [Running Benchmarks](#running-benchmarks)
  - [Configuration Options](#configuration-options)
  - [Vivado Flow Integration](#vivado-flow-integration)

## ISPD 2016/2017 Benchmarks

### Obtaining Benchmarks

Download benchmark files from:

- ISPD 2016 FPGA Placement Benchmarks [[Google Drive](https://drive.google.com/file/d/1kzg0NfEmJvwzhJADPE_Q0UjS6UpVCMZZ/view?usp=share_link) / [Baidu Drive](https://pan.baidu.com/s/11TnGIyiCbAOvjIRliuamPg?pwd=521g)]

- ISPD 2016 FPGA Placement Flexshelf Benchmarks [[Google Drive](https://drive.google.com/file/d/1lwYSwfIPfzOxi_SfOZj5DyiwROLJDFd0/view?usp=sharing) / [Baidu Drive](https://pan.baidu.com/s/1xHJ7kXTHshe-jCl4HgTuXA?pwd=o57k)]

- ISPD 2017 FPGA Placement Benchmarks [[Google Drive](https://drive.google.com/file/d/1Uf9qIZ8WL_jk03sIlAoS9dIrvYH3d1pz/view?usp=sharing) / [Baidu Drive](https://pan.baidu.com/s/12Ixpa5nuCK5BOPZisI3g-A?pwd=dmny)]

- ISPD 2017 FPGA Placement Flexshelf Benchmarks [[Google Drive](https://drive.google.com/file/d/1smt4lGUFdhs0TjPBzi9PqiyfA9n2Uwoy/view?usp=sharing) / [Baidu Drive](https://pan.baidu.com/s/1S7cfv26zURKo9W6WXwB3JA?pwd=8nqv)]

Extract to `<benchmark directory>` with structure:
```
<benchmark directory>
├── ispd2016
├── ispd2016_flexshelf
├── ispd2017
├── ispd2017_flexshelf
```

### Setting Up Benchmarks

Link benchmark folders to installation directory:
```bash
ln -s <benchmark directory>/ispd2016 <installation directory>/benchmarks/ispd2016
ln -s <benchmark directory>/ispd2016_flexshelf <installation directory>/benchmarks/ispd2016_flexshelf
ln -s <benchmark directory>/ispd2017 <installation directory>/benchmarks/ispd2017
ln -s <benchmark directory>/ispd2017_flexshelf <installation directory>/benchmarks/ispd2017_flexshelf
```

### Running Benchmarks

Single benchmark:
```bash
cd <installation directory>
python openparf.py --config unittest/regression/ispd2016/FPGA01.json
```

Batch processing:
```bash
cd <installation directory>
<source directory>/scripts/run_ispd2016_benchmark.sh # ISPD 2016
<source directory>/scripts/run_ispd2016_flexshelf_benchmark.sh # ISPD 2016 Flexshelf
<source directory>/scripts/run_ispd2017_benchmark.sh # ISPD 2017
<source directory>/scripts/run_ispd2017_flexshelf_benchmark.sh # ISPD 2017 Flexshelf
```

### Configuration Options

Common parameters in JSON config files:
- `gpu`: Enable GPU acceleration (Default: 0)
- `result_dir`: Results directory
- `route_flag`: Enable router (Default: 0)
- `slr_aware_flag`: Enable multi-die placer (Default: 1)

See `openparf/params.json` for all available parameters.

### Vivado Flow Integration

For placement evaluation using Vivado:
1. Disable routing by setting `route_flag: 0`
2. Find placement results in `<result dir>/<benchmark name>.pl`
3. Follow ISPD [2016](https://www.ispd.cc/contests/16/Flow.txt)/[2017](https://www.ispd.cc/contests/17/Flow.txt) instructions for Vivado integration