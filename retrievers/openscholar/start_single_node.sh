#!/bin/bash
set -euo pipefail

# Download the OpenScholar datastore from:
# https://huggingface.co/datasets/OpenSciLM/OpenScholar-DataStore-V3
#
# Run this script from the OpenScholar retrieval-scaling API directory, or set:
#   OPENSCHOLAR_API_ROOT=/path/to/OpenScholar-main/retriever2/retrieval-scaling-main
# The datastore/index paths are read by the Hydra config selected below.


export HYDRA_CONFIG_NAME=ivf_pq
export DS_DOMAIN=pes2o_v3
export NUM_SHARDS=16
export NUM_SHARDS_PER_WORKER=16
export WORKER_ID=0

OPENSCHOLAR_API_ROOT="${OPENSCHOLAR_API_ROOT:-/path/to/OpenScholar-main/retriever2/retrieval-scaling-main}"
cd "$OPENSCHOLAR_API_ROOT"

PYTHONPATH=. python api/serve_worker_node_fastapi.py
