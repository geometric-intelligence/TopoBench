#!/usr/bin/env bash

set -euo pipefail

readonly EXPERIMENTS=(
    example
    graph_synthetic_regression
    heterogeneous_synthetic_hgt_full
    heterogeneous_synthetic_hgt_neighbor
    heterogeneous_synthetic_heterosage_full
    heterogeneous_synthetic_heterosage_neighbor
    heterogeneous_dblp_hgt
    heterogeneous_dblp_heterosage
    heterogeneous_ogb_mag_hgt
    heterogeneous_ogb_mag_heterosage
    hypergraph_synthetic_edgnn
    hypergraph_synthetic_hypergraph_conv
)

for experiment in "${EXPERIMENTS[@]}"; do
    uv run python -m topobench "experiment=${experiment}" "$@"
done
