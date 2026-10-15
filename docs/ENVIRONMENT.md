# Environment Setup

The pipeline has three local components: data processing, retriever serving, and RL training. LLM synthesis and reward judging use external OpenAI-compatible APIs. Separate Python environments can be used for each local component.

The instructions below follow the checked-in source and package declarations. They are not a frozen environment for reproducing the paper: the repository does not contain a complete lock of Python, PyTorch, CUDA, vLLM, FlashAttention, and FAISS versions. A fresh GPU installation and an end-to-end training run have not been validated as part of this documentation update.

## Data Processing

Use Python 3.10 or newer. From the repository root, install the packages used by data construction, verification, and parquet conversion:

```bash
python -m pip install openai aiohttp requests numpy pandas tqdm transformers huggingface-hub pyarrow
```

The construction scripts call retrieval and LLM APIs. They do not start a local policy or judge model. They require the relevant services and downloaded corpus files to be available.

Sources: [`data_construction/`](../data_construction/), [`data_conversion/`](../data_conversion/).

## Retriever Serving

The supplied launchers use GPU-enabled FAISS. Installing only `faiss-cpu` is insufficient for these launchers.

| Service | Packages used by the server | Entry point |
| :--- | :--- | :--- |
| Wikipedia | PyTorch, GPU-enabled FAISS, NumPy, Transformers, Datasets, tqdm, FastAPI, Uvicorn, Pydantic | [`launch_local_server.sh`](../retrievers/wiki/launch_local_server.sh) |
| OpenScholar | PyTorch, GPU-enabled FAISS, Transformers, SentenceTransformers, Hydra/OmegaConf, FastAPI/Uvicorn, and the vendored retrieval/Contriever dependencies | [`start_single_node.sh`](../retrievers/openscholar/start_single_node.sh) |

The upstream [OpenScholar environment file](../retrievers/openscholar/retrieval-scaling-main/environment.yml) declares Python 3.11 and FAISS-GPU 1.8.0. It is a retriever dependency reference, not the RL environment. The FastAPI serving code also requires `fastapi` and `uvicorn`, which are not listed in that file. Verify the PyTorch/CUDA/FAISS combination on the machine used for serving.

The Wikipedia launcher runs `python3`; the OpenScholar launcher runs `python`. Activate the corresponding serving environment before launching each service. If they use different environments, start them in separate terminals instead of using the combined launcher.

Set GPU assignments for the retriever processes separately from the training process. The Wikipedia launcher defaults to `CUDA_VISIBLE_DEVICES=0`; the default training example uses eight GPUs.

For corpus layout and launch commands, see [`retrievers/README.md`](../retrievers/README.md).

## RL Training

Both package manifests declare Python 3.10 or newer. Start with a CUDA/PyTorch environment compatible with the selected vLLM and FlashAttention builds. From the repository root:

```bash
python -m pip install -e './training/verl-tool/verl[vllm]'
python -m pip install -e './training/verl-tool/verl[gpu]' --no-build-isolation
python -m pip install -e ./training/verl-tool
python -m pip install openai aiohttp requests fastapi uvicorn
```

The editable installations read these package declarations:

| Source | Relevant declarations |
| :--- | :--- |
| [`verl/setup.py`](../training/verl-tool/verl/setup.py) | Base training dependencies; `vllm` extra: `vllm>=0.8.5,<=0.11.0`; `gpu` extra: `liger-kernel` and `flash-attn` |
| [`verl/pyproject.toml`](../training/verl-tool/verl/pyproject.toml) | Python version requirement and build configuration |
| [`verl-tool/pyproject.toml`](../training/verl-tool/pyproject.toml) | Tool-runtime dependencies |

These version ranges describe the vendored source's declarations, not a claim that every combination has been tested. The FSDP worker defaults to `flash_attention_2`, and the training entrypoint selects vLLM. Do not replace these backends with unrelated current versions without checking compatibility.

The existing development requirements file is [`training/verl-tool/verl/requirements.txt`](../training/verl-tool/verl/requirements.txt). A file at `training/verl-tool/requirements.txt` does not exist. The editable installations above use the package metadata directly.

The training entrypoint prepends this checkout's `training/verl-tool` and `training/verl-tool/verl` directories to `PYTHONPATH`. It uses `PYTHON_BIN`, defaulting to `python3`, for both the tool server and training process.

## LLM Services

Use an endpoint that serves the model IDs configured for each role. The roles may share a service; they do not require separate local model deployments.

| Variable | Role | Where it is consumed |
| :--- | :--- | :--- |
| `QA_MODEL` | Evidence-tree question answering | `scripts/run_pipeline.sh`, passed as `--qa-model` |
| `EXTRACT_MODEL` | Query/rubric extraction | `scripts/run_pipeline.sh`, passed as `--extract-model` |
| `JUDGE_MODEL` | Offline sample verification | `scripts/run_pipeline.sh`, passed as `--judge-model` |
| `RUBRIC_JUDGE_MODEL` | Rubric reward during RL training | `LongformRubricRewardManager` |
| `CITATION_JUDGE_MODEL` | Citation reward during RL training | Citation scoring utilities |

`JUDGE_MODEL` is a pipeline-wrapper variable. When invoking the verifier directly, pass `--judge-model` as shown in the reproduction commands; its environment fallback is `QUALITY_JUDGE_MODEL`.

The shared API configuration is:

```bash
export OPENAI_BASE_URL=http://localhost:8008/v1
export OPENAI_API_KEY=EMPTY
```

Use the key required by your service. `EMPTY` is a placeholder for services without authentication. Avoid port 8000 for this API when using the default OpenScholar retriever on the same host.

For separate training judge endpoints, set `RUBRIC_JUDGE_BASE_URLS`, `CITATION_OPENAI_BASE_URLS`, and, when using the async citation path, `CITATION_ASYNC_OPENAI_BASE_URLS`. Each accepts a comma-separated list. Otherwise the corresponding clients fall back to `OPENAI_BASE_URL`.

## Running Across Environments

1. Start the Wikipedia and OpenScholar services in their serving environments.
2. In the data-processing environment, configure corpus paths, retriever URLs, and synthesis/verification model IDs.
3. Set `PYTHON_BIN` to the training environment's Python executable and configure the RL judge model IDs.
4. Run `bash scripts/run_pipeline.sh` from the repository root.

The pipeline's data commands use the active `python`. `PYTHON_BIN` affects the training stage only. `START_RETRIEVERS=1` additionally launches both retrievers using the active `python`/`python3`; use it only when that environment contains their dependencies.
