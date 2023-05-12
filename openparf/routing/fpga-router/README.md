## FPGA Router

#### How to build
```bash
mkdir build
cd build
cmake ..
make
```

#### How to run
```bash
cd build
./fpga-router -xml [xml architecture file] -pl [ispd place result] -net [ispd net file] -node [ispd node file] -out [output xml file] -mt [0(Single Thread)|1(Static Schedule)|2(Taskflow Schedule)]
```
