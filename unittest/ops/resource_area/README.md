\ingroup GrpDummyPages
\page UnitTest_OPS_ResourceArea Resource Area 

Resource area operator for control set optimization. 

The data folder contains results dumped from a run of elfplace, on the ISPD2017 FPGA02 dataset.

- pl: placement vector before resource_area computation
- lut_type_array: list of k-v pairs. k is inst_id of luts, v is the lut type
- ff_ctrlsets: list of k-v pairs. k is inst_id of ffs, v are the cksr and ce group the ff is in
- lut demand_map, ff_demand_map: demand map calculated from Gaussian distribution, dumped after fillGaussianDemandMap. Row major order
- lut_area_map, ff_area_map: area map calcluated from the demand map, dumpded after computeInstanceAreaMap. Row major order
- inst_area: final results, list of resource_areas. inst_area[i] is the resource_area of inst i


