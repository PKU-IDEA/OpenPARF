##
# @file   unittest_ff_ctrlsets.py
# @author Yibai Meng
# @date   Aug 2020
#

import os
import sys
import unittest
import torch
import numpy as np
import math

print(sys.argv)
if len(sys.argv) < 2:
    print("usage: python script.py [project_dir]")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    project_dir = os.path.abspath(sys.argv[1])
print("use project_dir = %s" % project_dir)

sys.path.append(project_dir)
from openparf.ops.resource_area import resource_area
sys.path.pop()


def compute_ff_ctrlsets_golden(
    inst_pins,
    inst_pins_start,
    pin2net_map,
    pin_signal_types,
    is_inst_ffs):
    cksr_mapping = dict()
    ce_mapping = dict()
    cksr_id = 0
    ce_id = 0
    num_insts = inst_pins_start.numel() - 1
    ff_ctrlsets = torch.full([num_insts, 2], 2147483647, dtype=torch.int32)
    for (inst, is_ff) in enumerate(is_inst_ffs):
        if not is_ff:
            continue
        # First find the nets of the Clock, Set/Reset, Clock Enable pins are
        inst_cksr = None
        inst_ce = None
        pins = inst_pins[inst_pins_start[inst]:inst_pins_start[inst+1]]
        # See enum class SignalType in util/enum.h for id of types
        types = [pin_signal_types[i].item() for i in pins]
        try:
            idx_ck = types.index(0)
            net_ck = pin2net_map[pins[idx_ck]].item()
        except ValueError:
            raise("No Clock in pins of instance", inst)
        try:
            idx_sr = types.index(1)
            net_sr = pin2net_map[pins[idx_sr]].item()
        except ValueError:
            net_sr = -1
        try:
            idx_ce = types.index(2)
            net_ce = pin2net_map[pins[idx_ce]].item()
        except ValueError:
            net_ce = -1
        if net_ck in cksr_mapping:
            sr_mapping = cksr_mapping[net_ck]
            if net_sr in sr_mapping:
                inst_cksr = sr_mapping[net_sr]
            else:
                inst_cksr = cksr_id
                sr_mapping[net_sr] = cksr_id
                cksr_id += 1
        else:
            inst_cksr = cksr_id
            cksr_mapping[net_ck] = dict()
            cksr_mapping[net_ck][net_sr] = cksr_id
            cksr_id += 1

        if net_ce in ce_mapping:
            inst_ce = ce_mapping[net_ce]
        else:
            inst_ce = ce_id
            ce_mapping[net_ce] = ce_id
            ce_id += 1
        ff_ctrlsets[inst][0] = inst_cksr
        ff_ctrlsets[inst][1] = inst_ce
    return ff_ctrlsets, cksr_id, ce_id

class FFCtrlSetsUnittest(unittest.TestCase):
    def test_ff_ctrlsets_area(self):
        print("Testing FF ctrlsets of FPGA02")
        dump = torch.load(os.path.dirname(os.path.abspath(__file__)) + "/unittest_ff_ctrlsets_dump_2.pt")
        ff_ctrlsets, cksr_size, ce_size = resource_area.compute_control_sets(
                 dump["inst_pins"],
                 dump["inst_pins_start"],
                 dump["pin2net_map"],
                 dump["pin_signal_types"],
                 dump["is_inst_ffs"])

        ff_ctrlsets_golden, cksr_size_golden,ce_size_golden = compute_ff_ctrlsets_golden(
                 dump["inst_pins"],
                 dump["inst_pins_start"],
                 dump["pin2net_map"],
                 dump["pin_signal_types"],
                 dump["is_inst_ffs"])
        diff = ff_ctrlsets - ff_ctrlsets_golden
        is_close = torch.allclose(ff_ctrlsets, ff_ctrlsets_golden)
        self.assertTrue(is_close)
        #print(ff_ctrlsets, ff_ctrlsets_golden, cksr_size, cksr_size_golden, ce_size, ce_size_golden)
        #print(diff, torch.max(diff), torch.nonzero(diff))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()  # Ignore the first one!
    unittest.main()
