# Architecture Customization Guide

## Table of Contents

- [Multi-Die FPGA Architecture](#multi-die-fpga-architecture)
- [Architecture Definition Format](#architecture-definition-format)
  - [Overview](#overview)
  - [Example Architecture](#example-architecture)
- [Customizing the Architecture](#customizing-the-architecture)
  - [Steps to Customize](#steps-to-customize)
- [SLL Counts Lookup Table](#sll-counts-lookup-table)
  - [Generating Custom Tables](#generating-custom-tables)
  - [Integration Steps](#integration-steps)
- [Best Practices](#best-practices)

## Multi-Die FPGA Architecture

OpenPARF supports customization of multi-die FPGA architectures through a structured XML format. This guide explains how to define and modify architectures for your specific needs.

## Architecture Definition Format

### Overview
Architecture files are located at `benchmarks/arch/ultrascale/multi-die_layout_<{num_cols}x{num_rows}>.xml`. The XML format includes several key sections:

- `<resources>`: Define available FPGA resources
- `<primitives>`: Specify primitive elements
- `<global_sw>`: Global switch configurations
- `<tile_blocks>`: Define tile block structures
- `<cores>`: Core specifications
- `<chip>`: Overall chip topology and SLR configuration

### Example Architecture

Here's a sample `2x2` SLR topology:

```xml
<chip name="demo_chip">
    <grid name="chip_grid" cols="2" rows="2">
        <core name="CORE_0" type="CORE_A" x="0" y="0" width="168" height="120"/>
        <core name="CORE_1" type="CORE_A" x="0" y="1" width="168" height="120"/>
        <core name="CORE_2" type="CORE_B" x="1" y="0" width="168" height="120"/>
        <core name="CORE_3" type="CORE_B" x="1" y="1" width="168" height="120"/>
    </grid>
</chip>
```

## Customizing the Architecture

### Steps to Customize

1. **Define SLR Topology**
   - Modify the `<grid>` attributes in `<chip>` section
   - Set appropriate `cols` and `rows` values

2. **Configure Cores**
   - Define each core with unique `name`
   - Specify `type`, position (`x`,`y`), and dimensions (`width`,`height`)
   - Ensure core specifications match SLR requirements

3. **Adjust Resources**
   - Customize `<resources>`, `<primitives>`, and other sections as needed
   - Maintain consistency with core configurations

4. **Verify Configuration**
   - Check all specifications for correctness
   - Ensure topology matches design requirements

## SLL Counts Lookup Table

### Generating Custom Tables

For SLR topologies beyond `1x4` or `2x2`, generate a custom SLL counts lookup table:

```bash
python compute_sll_counts_table.py --num_cols <num cols> --num_rows <num rows> --output <filename>
```

### Integration Steps

1. **Generate Table**
   - Run script with desired dimensions
   - Table will be saved as `<filename>.npy`

2. **Install Table**
   - Move generated `.npy` file to `<installation directory>/openparf/ops/sll/`
   - Update code in `sll.py` to use new table:

```python
else:
    self.sll_counts_table = torch.from_numpy(
        np.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "<filename>.npy"))).to(dtype=torch.int32)
```

## Best Practices

1. Keep dimensions reasonable (typically ≤ 5x5 due to fabrication limits)
2. Maintain consistent resource distribution
3. Verify all specifications before implementation
4. Document custom configurations
5. Test thoroughly after modifications