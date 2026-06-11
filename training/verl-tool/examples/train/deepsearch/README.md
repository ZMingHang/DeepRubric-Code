# DeepRubric Training

This directory contains the GRPO entrypoint used for the DeepRubric training stage.

Prepare parquet data first:

```bash
python examples/data_preprocess/preprocess_search_r1_dataset_tool.py \
  --input_file ../../data/query_1001.jsonl \
  --local_dir data/deeprubric
```

Start local Wikipedia/OpenScholar retrievers as described in the top-level release README, then launch training:

```bash
export MODEL_PATH=Qwen/Qwen3-8B
export DATASET_NAME=deeprubric
export N_GPUS_PER_NODE=8
export TRAINER_LOGGER="['console']"

bash examples/train/deepsearch/train_4b_tool.sh
```

Useful overrides:

```bash
export TRAIN_DATA=/path/to/train.parquet
export VAL_DATA=/path/to/test.parquet
export CHECKPOINT_DIR=/path/to/checkpoints
export LOG_DIR=/path/to/logs
export TOOL_SERVER_HOST=0.0.0.0
export TOOL_SERVER_PORT=30500
export WORKERS_PER_TOOL=16
```
