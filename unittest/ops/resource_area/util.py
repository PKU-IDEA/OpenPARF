##
# @file   util.py
# @author Yibai Meng
# @date   Aug 2020
# Generates the input and output of this step from elfplace dumps
# The first adjustment step of FPGA02
import torch
def load_vector(filename :str, dtype):
    ls = []
    with open(filename) as fp:
        for l in fp:
            ls.append(float.fromhex(l.strip()))
    res = torch.tensor(ls, dtype=dtype)
    return res
def load_xy_vector(filename, dtype):
    ls = []
    with open(filename) as fp:
        for l in fp:
            ls.append(float.fromhex(l.strip()))
    r = []
    for i in range(len(ls)):
        if i % 2 == 0:
            r.append([ls[i], ls[i+1]])
    res = torch.tensor(r, dtype=dtype)
    return res
def load_lut_type_array(filename :str, is_luts):
    with open(filename) as fp:
        for l in fp:
            a = l.split()
            is_luts[int(a[0])] = int(a[1])
def load_ff_ctrlsets_array(filename :str, ff_ctrlsets,is_ffs):
    with open(filename) as fp:
        for l in fp:
            a = l.split()
            ff_ctrlsets[int(a[0])][0] = int(a[1])
            ff_ctrlsets[int(a[0])][1] = int(a[2])
            is_ffs[int(a[0])] = 1
def load_lut_bin_map(filename, lut_bin_map):
   ls = []
   with open(filename) as fp:
       idx = 0 
       cnt = 0
       for l in fp:
            a = l.strip()
            if a == "--":
                idx+=1
                cnt = 0
            else:
                lut_bin_map[idx*6+cnt] = float.fromhex(a)
                cnt+=1
def load_ff_bin_map(filename, ff_bin_map):
   ls = []
   with open(filename) as fp:
       idx = 0 
       for l in fp:
            a = l.strip()
            if a == "#":
                idx+=1
            else:
                a = a.split()
                try:
                    ff_bin_map[idx*11*11 + int(a[0]) * 11 + int(a[1])] = float.fromhex(a[2])
                except IndexError:
                    print(idx, a[0], a[1], a[2])

if __name__ == "__main__":
    import sys
    ff_cksr_size = 11
    ff_ce_size = 11
    inst_pos = load_xy_vector("pl", torch.float64)
    num_insts = 166374  # No need to deal with those fillers inst_pos.size()[0]
    num_x = 131
    num_y = 373
    is_luts = torch.zeros(num_insts, dtype=torch.uint8)
    is_ffs = torch.zeros(num_insts, dtype=torch.uint8)
    ff_ctrlsets = torch.zeros([num_insts, 2], dtype=torch.int32) 
    load_lut_type_array("lut_type_array", is_luts)
    load_ff_ctrlsets_array("ff_ctrlsets", ff_ctrlsets, is_ffs)
    inst_area = load_vector("inst_area", torch.float64)
    lut_demand_map = torch.zeros((num_x, num_y,6), dtype=torch.float64)
    ff_demand_map = torch.zeros((num_x, num_y, ff_cksr_size, ff_ce_size), dtype=torch.float64)
    load_lut_bin_map("lut_demand_map", lut_demand_map.view(num_x * num_y * 6))
    load_ff_bin_map("ff_demand_map", ff_demand_map.view(num_x * num_y * ff_cksr_size * ff_ce_size))
    torch.save({"inst_pos" : inst_pos, "is_luts" : is_luts, "is_ffs" : is_ffs, "ff_ctrlsets" : ff_ctrlsets, "inst_area" : inst_area, "lut_demand_map" : lut_demand_map, "ff_demand_map" : ff_demand_map}, "unittest_resource_area_dump.pt")
