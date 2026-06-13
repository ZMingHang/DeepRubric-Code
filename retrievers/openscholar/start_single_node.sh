#!/bin/bash
set -euo pipefail

# This repository vendors the OpenScholar retrieval-scaling FastAPI server under:
#   retrievers/openscholar/retrieval-scaling-main/
#
# Download the OpenScholar datastore from:
# https://huggingface.co/datasets/OpenSciLM/OpenScholar-DataStore-V3
#
# Set OPENSCHOLAR_DATA_ROOT to the downloaded datastore directory. The script
# passes that directory into the vendored Hydra config through overrides.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HYDRA_CONFIG_NAME=ivf_pq
export DS_DOMAIN=pes2o_v3
export NUM_SHARDS=16
export NUM_SHARDS_PER_WORKER=16
export WORKER_ID=0

OPENSCHOLAR_DATA_ROOT="${OPENSCHOLAR_DATA_ROOT:-/path/to/OpenScholar-DataStore-V3}"
export HYDRA_OVERRIDE_DATASTORE__RAW_DATA_PATH="${HYDRA_OVERRIDE_DATASTORE__RAW_DATA_PATH:-$OPENSCHOLAR_DATA_ROOT}"
export HYDRA_OVERRIDE_DATASTORE__DATASTORE_ROOT_DIR="${HYDRA_OVERRIDE_DATASTORE__DATASTORE_ROOT_DIR:-$OPENSCHOLAR_DATA_ROOT}"

OPENSCHOLAR_SERVER_ROOT="${OPENSCHOLAR_SERVER_ROOT:-${SCRIPT_DIR}/retrieval-scaling-main}"
cd "$OPENSCHOLAR_SERVER_ROOT"

PYTHONPATH=. python api/serve_worker_node_fastapi.py
