#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 1. begin ray server
host="${TOOL_SERVER_HOST:-0.0.0.0}"
port="${TOOL_SERVER_PORT:-$(shuf -i 30000-31000 -n 1)}"
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host "$host" --port "$port" --tool_type "${TOOL_TYPES:-tool_search_multi,tool_browse,tool_scholar_multi}" --workers_per_tool "${WORKERS_PER_TOOL:-4}" --done_if_invalid True --slient True &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

# 2. start api service
model_path="${MODEL_PATH:-Qwen/Qwen3-8B}"
max_turns="${MAX_TURNS:-20}" # maximum interaction turns between model and tool server
min_turns="${MIN_TURNS:-0}" # minimum action turns
api_host="${API_HOST:-0.0.0.0}"
api_port="${API_PORT:-5000}"
action_stop_tokens='</tool_call>' # stop at this token, then send the output to the tool server, this is a special token that we use to indicate the end of the action, you can change it to any other token that your model will produce when it is asking for a tool calling round
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"
num_models="${NUM_MODELS:-1}" # num_models * tensor_parallel_size should match available GPUs
enable_mtrl="${ENABLE_MTRL:-True}" # whether to evaluate in multi-chat-turn setting
# temp file for action tokens as verl cannot pass special strs as params
action_stop_tokens_file=$(mktemp)
echo "$action_stop_tokens" > "$action_stop_tokens_file"
echo "action_stop_tokens_file=$action_stop_tokens_file"

python eval_service/app.py \
    --host $api_host \
    --port $api_port \
    --tool_server_url $tool_server_url \
    --model $model_path \
    --max_turns $max_turns \
    --min_turns $min_turns \
    --action_stop_tokens $action_stop_tokens_file \
    --tensor_parallel_size $tensor_parallel_size \
    --num_models $num_models \
    --enable_mtrl $enable_mtrl &

api_server_pid=$!
echo "API started at $api_host:$api_port"

# 3. kill all server
cleanup() {
    pkill -P "$server_pid" 2>/dev/null || true
    kill "$server_pid" 2>/dev/null || true
    pkill -P "$api_server_pid" 2>/dev/null || true
    kill "$api_server_pid" 2>/dev/null || true
    rm -f "$action_stop_tokens_file"
}
trap cleanup EXIT
wait "$api_server_pid"
