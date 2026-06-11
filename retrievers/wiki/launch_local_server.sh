#!/usr/bin/env bash
set -euo pipefail
set -x

# Download the Wikipedia retriever assets from:
# https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge
#
# Expected layout:
#   ${WIKI_RETRIEVER_ROOT}/e5.index/e5_Flat.index
#   ${WIKI_RETRIEVER_ROOT}/wiki_corpus.jsonl
#   ${WIKI_RETRIEVER_ROOT}/wiki_webpages.jsonl
#   ${WIKI_RETRIEVER_ROOT}/wikilinks.json
#   ${WIKI_RETRIEVER_ROOT}/e5-base-v2/
#
# The retriever server implementation is provided by ASearcher:
# https://github.com/inclusionAI/ASearcher
# Set WIKI_CODE_ROOT to that checkout. The default is intentionally a virtual
# path for public release reproducibility docs.

WIKI_CODE_ROOT="${WIKI_CODE_ROOT:-/path/to/ASearcher-main}"
WIKI_RETRIEVER_ROOT="${WIKI_RETRIEVER_ROOT:-/path/to/ASearcher-Local-Knowledge}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

server_py="${WIKI_LOCAL_RETRIEVAL_SERVER:-${WIKI_CODE_ROOT}/tools/local_retrieval_server.py}"
index_file="${WIKI_INDEX_FILE:-${WIKI_RETRIEVER_ROOT}/e5.index/e5_Flat.index}"
corpus_file="${WIKI_CORPUS_FILE:-${WIKI_RETRIEVER_ROOT}/wiki_corpus.jsonl}"
pages_file="${WIKI_PAGES_FILE:-${WIKI_RETRIEVER_ROOT}/wiki_webpages.jsonl}"
retriever_name="${WIKI_RETRIEVER_NAME:-e5}"
retriever_path="${WIKI_RETRIEVER_MODEL:-${WIKI_RETRIEVER_ROOT}/e5-base-v2}"
port="${WIKI_RETRIEVER_PORT:-8888}"
address_dir="${WIKI_ADDRESS_DIR:-./outputs/retriever_addresses/wiki}"

if [ ! -f "$server_py" ]; then
  echo "Cannot find local_retrieval_server.py: $server_py" >&2
  echo "Set WIKI_CODE_ROOT=/path/to/ASearcher-main or WIKI_LOCAL_RETRIEVAL_SERVER=/path/to/local_retrieval_server.py." >&2
  exit 1
fi

python3 "$server_py" \
  --index_path "$index_file" \
  --corpus_path "$corpus_file" \
  --pages_path "$pages_file" \
  --topk 3 \
  --retriever_name "$retriever_name" \
  --retriever_model "$retriever_path" \
  --faiss_gpu \
  --port "$port" \
  --save-address-to "$address_dir"
