## OpenPARF Router For FPGA'24 Contest

This is the router part of OpenPARF project, and it is the solution for FPGA'24 contest by Team Cuckoo. The Idea of this router is refered to [here](https://xilinx.github.io/fpga24_routing_contest/results.html)

## How to Build This Router

```
mkdir build
cd build
cmake ../src -DCMAKE_PREFIX_PATH=/path/to/env -DCMAKE_BUILD_TYPE=Release
make
```

## How to Run
If you want to directly load RRG:
```
./fpgarouter -rrg /path/to/rrg_folder -phys /path/to/unrouted.phys -ifout /path/to/output.phys
```
If you want to load device file of FPGA Interchange Format (see [Repo of FPGA'24 Contest](https://github.com/Xilinx/fpga24_routing_contest)):
```
./fpgarouter -device /path/to/input.device -phys /path/to/unrouted.phys -ifout /path/to/output.phys
```
