# 进入仓库根目录
host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve \
  --host $host --port $port \
  --tool_type "browse,search,scholar" \
  --workers_per_tool 4 --use_ray True &

model_path=/path/to/your/deepsearch_checkpoint  # 换成你的 checkpoint
action_stop_tokens='</browse>,</search>,</scholar>'
action_stop_tokens_file=$(mktemp); echo "$action_stop_tokens" > $action_stop_tokens_file

api_host=0.0.0.0
api_port=5000
python eval_service/app.py \
  --host $api_host --port $api_port \
  --tool-server-url $tool_server_url \
  --model $model_path \
  --max_turns 3 \
  --min_turns 0 \
  --action_stop_tokens $action_stop_tokens_file \
  --tensor-parallel-size 1 \
  --num-models 1 \
  --enable_mtrl False &
