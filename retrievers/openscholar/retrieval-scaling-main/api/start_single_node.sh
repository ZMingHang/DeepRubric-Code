#!/bin/bash
set -euo pipefail

# Single-node API startup using api/conf/ivf_pq.yaml
# Assumes conda env "scaling" exists.


export HYDRA_CONFIG_NAME=ivf_pq
export DS_DOMAIN=pes2o_v3
export NUM_SHARDS=16
export NUM_SHARDS_PER_WORKER=16
export WORKER_ID=0

PYTHONPATH=. python api/serve_worker_node_fastapi.py
