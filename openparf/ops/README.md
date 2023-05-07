\ingroup GrpDummyPages
\page OpenPARF_Operators Operators

Operator implementations.

| Operator | Functionality | CPU | GPU |
|----------|---------------|-----|-----|
| dct      | Discrete cosine transformation kernels              | Y   | Y   |
| density_map | Distribute instances into MxN bins and compute the density map and overflow          | Y   | Y   |
| electric_potential | Compute the electric density, overflow, potential, and force according to elfPlace    | Y   | Y   |
| hpwl     | Compute half-perimeter wirelength              | Y   | Y   |
| wawl     | Compute weighted-average wirelength              | Y   | Y   |
| legality_check | Check legality of placement solutions        | Y   | N   |
| mcf_lg   | Legalize single-site-single-instance resources like RAM and DSP with min-cost flow              | Y   | N   |
| direct_lg | Legalize LUT and flip-flops             | Y   | N   |
| ism_dp   |  Detailed placement with independent set matching algorithm             | Y   | N   |
| move_boundary | Move instances back to the nearest locations inside the boundary of the layout         | Y   | Y   |
| pin_pos  | Compute pin locations from instance locations              | Y   | Y   |
| pin_utilization | Distribute pins into MxN bins and compute pin utilization       | Y   | Y   |
| raw_instance_area |      | Y   | N   |
| resource_area | Compute the clustering compatibility optimized area for LUTs and FFs, to better estimate the actual area occupied by instances in a feasible clustering solution | Y   | N   |
| rudy     | Compute RUDY/RISA routing utilization map              | Y   | Y   |
| stable_div | Compute stable divison to handle zero-divisors (by making the result zero)            | Y   | Y   |
