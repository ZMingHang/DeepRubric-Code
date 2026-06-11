# DEEPRUBRIC Training Code

This is the anonymous training-code release for:

**DEEPRUBRIC: Evidence-Tree Rubric Supervision for Efficient Reinforcement Learning of Deep Research Agents**

The repository contains the RL training pipeline for the DEEPRUBRIC deep
research agent. It focuses on converting query-rubric supervision into the
training format, launching tool-use GRPO training, and computing rubric,
citation, format, and tool-use rewards during training.

Runtime artifacts, local datasets, checkpoints, logs, caches, copied debug
files, and machine-specific paths have been removed.

## Repository Layout

```text
examples/data_preprocess/preprocess_search_r1_dataset_tool.py
  Converts DEEPRUBRIC/SearchR1-style query-rubric records into train/test
  parquet files consumed by verl-tool.

examples/train/deepsearch/train_4b_tool.sh
  Main DeepRubric training entrypoint. The default policy model is
  Qwen/Qwen3-8B and can be overridden with MODEL_PATH.

verl_tool/workers/reward_manager/
  Reward managers for long-form rubric supervision, citation checking, answer
  format reward, answer length reward, and search-turn reward.

verl_tool/servers/tools/
  Tool-server implementations for search, browse, scholar, and related
  tool-call parsing.

eval_service/
  Optional OpenAI-compatible evaluation service for running a trained model with
  the same tool interface.
```

## What Is Included

- Data preprocessing for query-rubric examples.
- Tool-use RL training based on GRPO.
- The `longform_rubric` reward manager used by the training script.
- Citation and rubric scoring utilities.
- Search, browse, and scholar tool wrappers.
- A vendored minimal `verl` runtime needed by `verl_tool`.

## What Is Not Included

- The paper PDF.
- Raw evidence-tree construction data.
- Large parquet training files.
- Model checkpoints.
- Search, browse, scholar, or judge-model server deployments.
- Private logs, W&B runs, local paths, and machine-specific configuration.

## Environment Setup

Use Python 3.10 or newer. A typical setup is:

```bash
conda create -n deeprubric python=3.10 -y
conda activate deeprubric

pip install -r requirements.txt
pip install -e ./verl
pip install -e .
```

Install optional tool dependencies as needed for your environment. For the
default deepsearch training path, the important external dependencies are the
retrieval/browse/scholar services and an OpenAI-compatible judge model endpoint.

## Data Format

The preprocessing script accepts `.jsonl`, `.json`, or `.parquet` input.
Each input record should contain:

- `question`: the user query for the deep research agent.
- `reward_model.ground_truth`: preferred field containing rubric supervision.
- `golden_answers`: fallback field used as rubric supervision if
  `reward_model.ground_truth` is absent.
- Optional fields such as `data_source`, `ability`, and `metadata`.

The script writes:

```text
data/deeprubric/train.parquet
data/deeprubric/test.parquet
```

Run:

```bash
python examples/data_preprocess/preprocess_search_r1_dataset_tool.py \
  --input_file ../../data/query_1001.jsonl \
  --local_dir data/deeprubric
```

Useful options:

```bash
--test_size 10
--seed 42
--local_dir data/custom_dataset_name
```

## External Services

Training starts the internal verl-tool server automatically, but the underlying
backends must be provided by the user.

### Tool Backends

Set these endpoints before training:

```bash
export SEARCH_URL=http://localhost:18881/retrieve
export BROWSE_URL=http://localhost:18881/access
export SCHOLAR_URL=http://localhost:8001/search
```

The default tool list launched by `train_4b_tool.sh` is:

```text
tool_search_multi,tool_browse,tool_scholar_multi
```

### Reward Judge

Rubric and citation rewards use OpenAI-compatible chat-completion APIs:

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8000/v1
export RUBRIC_JUDGE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
export CITATION_JUDGE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

For multiple judge endpoints:

```bash
export RUBRIC_JUDGE_BASE_URLS=http://host1:8000/v1,http://host2:8000/v1
export CITATION_OPENAI_BASE_URLS=http://host1:8000/v1,http://host2:8000/v1
```

## Training

After data and services are ready, launch the default DEEPRUBRIC-8B training
configuration:

```bash
export MODEL_PATH=Qwen/Qwen3-8B
export DATASET_NAME=deeprubric
export N_GPUS_PER_NODE=8
export TRAIN_BATCH_SIZE=64

bash examples/train/deepsearch/train_4b_tool.sh
```

Common overrides:

```bash
export TRAIN_DATA=/path/to/train.parquet
export VAL_DATA=/path/to/test.parquet
export CHECKPOINT_DIR=/path/to/checkpoints/deepsearch/run_name
export LOG_DIR=/path/to/logs
export TOOL_SERVER_HOST=0.0.0.0
export TOOL_SERVER_PORT=30500
export WORKERS_PER_TOOL=16
export TRAINER_LOGGER="['console']"
```

The default logger is console-only so the release does not require a private
W&B account or expose external run metadata.

## Reward Components

The default training script uses:

```bash
REWARD_MANAGER=longform_rubric
```

The reward manager combines:

- rubric scoring from the provided query-rubric supervision,
- citation quality scoring over cited evidence snippets,
- answer-format checks,
- search/tool-turn behavior,
- answer length regularization.

The active implementation is:

```text
verl_tool/workers/reward_manager/longform_rubric.py
```

Citation utilities used by this reward path are under:

```text
verl_tool/workers/reward_manager/utils/
```

## Optional Evaluation Service

To serve a trained checkpoint with the same tool interface:

```bash
export MODEL_PATH=/path/to/checkpoint/actor/huggingface
export API_PORT=5000
export TENSOR_PARALLEL_SIZE=1
export NUM_MODELS=1

bash eval_service/scripts/start_api_service.sh
```

For remote OpenAI-compatible inference, set:

```bash
export MODEL_PATH=remote/model-name
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
```

## Anonymous Release Notes

Additional cleanup details and the same launch commands are documented in
`ANONYMIZED_RELEASE.md`.
