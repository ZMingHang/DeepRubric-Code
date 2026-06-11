#!/usr/bin/env bash
set -euo pipefail

# Start both retrieval services used by DeepRubric. Keep this process alive
# while constructing data or training, because the training tool server calls
# these endpoints at rollout time.

export WIKI_CODE_ROOT="${WIKI_CODE_ROOT:-/path/to/ASearcher-main}"
export WIKI_RETRIEVER_ROOT="${WIKI_RETRIEVER_ROOT:-/path/to/ASearcher-Local-Knowledge}"
export OPENSCHOLAR_API_ROOT="${OPENSCHOLAR_API_ROOT:-/path/to/OpenScholar-main/retriever2/retrieval-scaling-main}"
export WIKI_RETRIEVER_URL="${WIKI_RETRIEVER_URL:-http://localhost:8888/retrieve}"
export WIKI_BROWSE_URL="${WIKI_BROWSE_URL:-http://localhost:8888/access}"
export OPENSCHOLAR_RETRIEVER_URL="${OPENSCHOLAR_RETRIEVER_URL:-http://localhost:8000/search}"

mkdir -p outputs/retriever_logs

bash retrievers/wiki/launch_local_server.sh > outputs/retriever_logs/wiki.log 2>&1 &
wiki_pid=$!

bash retrievers/openscholar/start_single_node.sh > outputs/retriever_logs/openscholar.log 2>&1 &
openscholar_pid=$!

cleanup() {
  kill "$wiki_pid" 2>/dev/null || true
  kill "$openscholar_pid" 2>/dev/null || true
}
trap cleanup EXIT

cat <<EOF
Started DeepRubric retrievers:
  Wikipedia   pid=$wiki_pid        endpoint=${WIKI_RETRIEVER_URL:-http://localhost:8888/retrieve}
  Wiki browse pid=$wiki_pid        endpoint=${WIKI_BROWSE_URL:-http://localhost:8888/access}
  OpenScholar pid=$openscholar_pid endpoint=${OPENSCHOLAR_RETRIEVER_URL:-http://localhost:8000/search}

Logs:
  outputs/retriever_logs/wiki.log
  outputs/retriever_logs/openscholar.log

Leave this script running. Press Ctrl-C to stop both retrievers.
EOF

wait
