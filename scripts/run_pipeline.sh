#!/usr/bin/env bash
set -euo pipefail

# End-to-end DeepRubric command template.
# Fill in paths and endpoint URLs before running. By default, this assumes the
# retriever services are already running. Set START_RETRIEVERS=1 to launch them
# from this script as part of the same run.

export WIKI_RETRIEVER_ROOT="${WIKI_RETRIEVER_ROOT:-/path/to/ASearcher-Local-Knowledge}"
export OPENSCHOLAR_DATA_ROOT="${OPENSCHOLAR_DATA_ROOT:-/path/to/OpenScholar-DataStore-V3}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8008/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export WIKI_RETRIEVER_URL="${WIKI_RETRIEVER_URL:-http://localhost:8888/retrieve}"
export WIKI_BROWSE_URL="${WIKI_BROWSE_URL:-http://localhost:8888/access}"
export OPENSCHOLAR_RETRIEVER_URL="${OPENSCHOLAR_RETRIEVER_URL:-http://localhost:8000/search}"
export SEARCH_URL="${SEARCH_URL:-$WIKI_RETRIEVER_URL}"
export BROWSE_URL="${BROWSE_URL:-$WIKI_BROWSE_URL}"
export SCHOLAR_URL="${SCHOLAR_URL:-$OPENSCHOLAR_RETRIEVER_URL}"

WIKI_RETRIEVER_PID=""
OPENSCHOLAR_RETRIEVER_PID=""

cleanup_retrievers() {
  if [ -n "$WIKI_RETRIEVER_PID" ]; then
    kill "$WIKI_RETRIEVER_PID" 2>/dev/null || true
  fi
  if [ -n "$OPENSCHOLAR_RETRIEVER_PID" ]; then
    kill "$OPENSCHOLAR_RETRIEVER_PID" 2>/dev/null || true
  fi
}

if [ "${START_RETRIEVERS:-0}" = "1" ]; then
  mkdir -p outputs/retriever_logs
  bash retrievers/wiki/launch_local_server.sh > outputs/retriever_logs/wiki.log 2>&1 &
  WIKI_RETRIEVER_PID=$!

  bash retrievers/openscholar/start_single_node.sh > outputs/retriever_logs/openscholar.log 2>&1 &
  OPENSCHOLAR_RETRIEVER_PID=$!

  trap cleanup_retrievers EXIT
  sleep "${RETRIEVER_WARMUP_SECONDS:-60}"

  if ! kill -0 "$WIKI_RETRIEVER_PID" 2>/dev/null; then
    echo "Wikipedia retriever exited during warmup. See outputs/retriever_logs/wiki.log." >&2
    exit 1
  fi
  if ! kill -0 "$OPENSCHOLAR_RETRIEVER_PID" 2>/dev/null; then
    echo "OpenScholar retriever exited during warmup. See outputs/retriever_logs/openscholar.log." >&2
    exit 1
  fi
fi

python data_construction/recursive_qa_agent_v4.py \
  --pages "$WIKI_RETRIEVER_ROOT/wiki_webpages.jsonl" \
  --links "$WIKI_RETRIEVER_ROOT/wikilinks.json" \
  --save outputs/wiki_evidence_trees \
  --retriever-url "$WIKI_RETRIEVER_URL" \
  --qa-model "${QA_MODEL:-deepseek/deepseek-chat}" \
  --extract-model "${EXTRACT_MODEL:-Qwen/Qwen3.5-35B-A3B-Instruct}" \
  --n "${WIKI_NUM_SAMPLES:-1000}"

python data_construction/recursive_qa_agent_v44.py \
  --pages "$OPENSCHOLAR_DATA_ROOT/passages/raw_passages-0-of-16.jsonl" \
  --save outputs/openscholar_evidence_trees \
  --retriever-url "$OPENSCHOLAR_RETRIEVER_URL" \
  --qa-model "${QA_MODEL:-deepseek/deepseek-chat}" \
  --extract-model "${EXTRACT_MODEL:-Qwen/Qwen3.5-35B-A3B-Instruct}" \
  --n "${OPENSCHOLAR_NUM_SAMPLES:-1000}"

mkdir -p outputs/verified_trees
python data_construction/recursive_qa_quality_filter.py \
  --input outputs/wiki_evidence_trees \
  --output-dir outputs/wiki_audited \
  --revised-output outputs/verified_trees/wiki_verified.jsonl \
  --include-keep-in-revised \
  --judge-model "${JUDGE_MODEL:-gpt-5.1}"

python data_construction/recursive_qa_quality_filter.py \
  --input outputs/openscholar_evidence_trees \
  --output-dir outputs/openscholar_audited \
  --revised-output outputs/verified_trees/openscholar_verified.jsonl \
  --include-keep-in-revised \
  --judge-model "${JUDGE_MODEL:-gpt-5.1}"

python data_conversion/read_form_tree_revise.py \
  --input-dir outputs/verified_trees \
  --output-dir data \
  --output-file query_1001.jsonl

python data_conversion/preprocess_search_r1_dataset_tool.py \
  --input_file data/query_1001.jsonl \
  --local_dir training/verl-tool/data/deeprubric

(
  cd training/verl-tool
  export DATASET_NAME="${DATASET_NAME:-deeprubric}"
  export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
  export TRAINER_LOGGER="${TRAINER_LOGGER:-['console']}"
  bash examples/train/deepsearch/train_4b_tool.sh
)
