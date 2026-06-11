# Anonymous Release Notes

This package is an anonymized code snapshot for the paper submission. Runtime
artifacts, local datasets, local checkpoints, logs, caches, copied debug files,
and machine-specific paths were removed. The default training entrypoint is:

```bash
bash examples/train/deepsearch/train_8b_tool.sh
```

## Data Preparation

Prepare SearchR1-style long-form tool data with:

```bash
python examples/data_preprocess/preprocess_search_r1_dataset_tool.py \
  --input_file /path/to/input.jsonl \
  --local_dir data/searchr1_longform_tool
```

The input can be `.jsonl`, `.json`, or `.parquet`. The script writes:

```text
data/searchr1_longform_tool/train.parquet
data/searchr1_longform_tool/test.parquet
```

You can choose another dataset directory and pass it to training with
`DATASET_NAME`, `TRAIN_DATA`, or `VAL_DATA`.

## Required Services

The training script starts the internal tool server automatically, but the
underlying search/browse/scholar backends and reward judge must be provided by
the user. Configure them with environment variables:

```bash
export SEARCH_URL=http://localhost:18881/retrieve
export BROWSE_URL=http://localhost:18881/access
export SCHOLAR_URL=http://localhost:8001/search

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8000/v1
export RUBRIC_JUDGE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
export CITATION_JUDGE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

If you use multiple judge endpoints, set `RUBRIC_JUDGE_BASE_URLS` or
`CITATION_OPENAI_BASE_URLS` as comma-separated OpenAI-compatible base URLs.

## Training

Set the policy model path or HuggingFace model id, then run:

```bash
export MODEL_PATH=Qwen/Qwen3-8B
export DATASET_NAME=searchr1_longform_tool
export N_GPUS_PER_NODE=8
export TRAIN_BATCH_SIZE=64

bash examples/train/deepsearch/train_8b_tool.sh
```

Useful overrides:

```bash
export TRAIN_DATA=/path/to/train.parquet
export VAL_DATA=/path/to/test.parquet
export CHECKPOINT_DIR=/path/to/checkpoints/deepsearch/run_name
export LOG_DIR=/path/to/logs
export TRAINER_LOGGER="['console']"
```

The default logger is console-only to avoid publishing or requiring any private
W&B configuration.

## Scope of Cleanup

The anonymized copy keeps the code required for data preprocessing, the
`train_8b_tool.sh` training path, `verl_tool/workers/reward_manager`, and
`verl_tool/servers/tools`. It removes local outputs, parquet data files, test
responses, logs, Python caches, copied debug files, and scripts containing
machine-specific local paths.
