
.. _analog-circuits-datasets:

**********************
Analog Circuit Datasets
**********************

This page documents, in depth, how the analog circuit datasets **AICircuit** and **AnalogGenie** are represented and processed inside TopoBench. Both are converted from SPICE-like netlists to **hypergraphs**, where:

- **Nodes** = circuit nets (wires)
- **Hyperedges** = devices (components)
- **Incidence roles** = pin-level roles (drain/gate/source/bulk, collector/base/emitter, …)
- **Hyperedge params** = parsed numeric parameters (best-effort)

High-level summary
==================

- **AICircuit** (https://arxiv.org/abs/2407.18272): graph-level regression  
  - Has graph attributes (design parameters) and targets (performance metrics).  
  - Source: `Dataset/<name>/<name>.csv` + `Simulation/Netlists/<name>/netlist`.
- **AnalogGenie** (https://github.com/xz-group/AnalogGenie, https://arxiv.org/abs/2503.00205): unsupervised  
  - No labels; large collection of `.cir` netlists.  
  - Source: `Dataset/<id>/<id>.cir` across the full GitHub repo (~493 MB).

Download sources
================

- AICircuit: GitHub zip `https://github.com/AvestimehrResearchGroup/AICircuit/archive/refs/heads/main.zip`
- AnalogGenie: GitHub repo `https://github.com/xz-group/AnalogGenie` (full clone used)

SPICE parsing and subcircuit flattening
=======================================

Both datasets share the same parsing logic:

1. **Subcircuits**: `.subckt ... .ends` blocks are stored; any `X*` instance is recursively expanded (pins are mapped to parent nets, nested subcircuits are flattened).
2. **Line tokenization** (simplified SPICE grammar):
   - MOS (M*, mos4…): `name drain gate source bulk [model] [params…]`
   - BJT (Q*, bjt/npn/pnp): `name collector base emitter [model] [params…]`
   - Other 2-terminal (R/L/C/V/I…): `name node1 node2 [type/model] [params…]`
   - Continuation lines starting with `+` are appended to the previous line.
   - Comments (`*`, `//`) and empty lines are skipped.

Hypergraph fields
=================

For every graph (circuit) we store:

- **x**: node features, 1D float code inferred from net names  
  - Codes: 0 generic, 1 power(vdd/vcc/pwr), 2 ground(vss/gnd), 3 input, 4 output, 5 bias, 6 gate, 7 drain, 8 source, 9 bulk/body/substrate, 10 clock.
- **hyperedge_index**: `[2, num_incident]` LongTensor of (node, hyperedge) incidence.
- **hyperedge_attr**: LongTensor of device type codes.  
  - AICircuit uses a richer vocab (resistor/capacitor/inductor/mos/bjt/…); AnalogGenie uses a smaller one {capacitor, nmos4, pmos4, resistor, unknown}.
- **incidence_roles**: 1D LongTensor aligned with `hyperedge_index` (same length), encoding pin roles:  
  - MOS: [drain=1, gate=2, source=3, bulk=4] by order.  
  - BJT: [collector=11, base=12, emitter=13] by order.  
  - Others: 0 (role unknown/generic).
- **hyperedge_params**: float Tensor `[num_edges, max_param_len]`  
  - Per-device numeric parameters parsed from tokens (e.g., W=2.5 → 2.5).  
  - Non-numeric tokens (model names, strings) are ignored. Rows are zero-padded to the max length within the graph; graphs with no numeric params have shape `[num_edges, 0]`.

Graph-level fields
------------------

- **AICircuit only**:
  - `graph_attr`: CSV front 4 columns → design parameters `[Wbias, Rd, Wn1, Wn2]`, padded/truncated to length 4.
  - `y`: CSV remaining columns → performance metrics (3 dims), padded/truncated to length 3.
- **AnalogGenie**: no labels/graph_attr (unsupervised).

Tasks and configs
=================

- AICircuit: regression, hypergraph domain  
  - Config: ``configs/dataset/hypergraph/aicircuit.yaml``  
  - Example run: ``python -m topobench dataset=hypergraph/aicircuit model=hypergraph/unignn``
- AnalogGenie: unsupervised, hypergraph domain  
  - Config: ``configs/dataset/hypergraph/analoggenie.yaml``  
  - Example run: ``python -m topobench dataset=hypergraph/analoggenie model=hypergraph/unignn2``

Notes and limitations
=====================

- **Roles beyond MOS/BJT** (e.g., voltage/current source polarity) are not inferred; they fall back to role=0.
- **Parameters** are best-effort numeric extraction; string/model names are dropped. Further structuring/normalization may be added if needed.
- **Net names** themselves are not stored verbatim—only categorical codes in `x`.
- SPICE coverage is simplified; exotic syntax/macros may be skipped.

Quick data stats (current parsing)
==================================

- AICircuit: 9 graphs (Mixer, LNA, PA, Receiver, CVA, …), each with graph_attr/y, MOS/BJT pin roles, and some numeric params (e.g., W/L).
- AnalogGenie (full repo): ~4,152 graphs, unlabeled, many MOS/BJT devices; most have no numeric params, so `hyperedge_params` is often empty.
