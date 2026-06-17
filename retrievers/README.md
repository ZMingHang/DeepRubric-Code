# Retrievers

DeepRubric uses two local retrieval services in both data construction and RL
training:

1. **Wikipedia retriever**: an ASearcher-style local dense retriever over
   Wikipedia passages. It provides both passage search and page browsing.
2. **OpenScholar retriever**: an OpenScholar/PES2O scientific-paper retriever.
   It provides academic evidence search.

This repository includes the lightweight server code and launch wrappers. It
does **not** include the large corpora, embedding models, FAISS indexes, or
OpenScholar datastore files. Download those assets separately and point the
scripts to your local copies.

## What to Download

| Component | Download source | Local path variable |
| --- | --- | --- |
| Wikipedia retriever assets | <https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge> | `WIKI_RETRIEVER_ROOT` |
| OpenScholar datastore | <https://huggingface.co/datasets/OpenSciLM/OpenScholar-DataStore-V3> | `OPENSCHOLAR_DATA_ROOT` |

Example download commands:

```bash
huggingface-cli download inclusionAI/ASearcher-Local-Knowledge \
  --repo-type dataset \
  --local-dir /path/to/ASearcher-Local-Knowledge

huggingface-cli download OpenSciLM/OpenScholar-DataStore-V3 \
  --repo-type dataset \
  --local-dir /path/to/OpenScholar-DataStore-V3
```

If your machine cannot download from Hugging Face directly, download the same
repositories with your local mirror or cluster transfer workflow. The launch
scripts only require the final local directory layout below.

## Required Files

The Wikipedia retriever directory must contain:

```text
$WIKI_RETRIEVER_ROOT/
  e5.index/
    e5_Flat.index
  wiki_corpus.jsonl
  wiki_webpages.jsonl
  wikilinks.json
  e5-base-v2/
```

How these files are used:

| File or directory | Used for |
| --- | --- |
| `e5.index/e5_Flat.index` | FAISS dense index for `/retrieve` |
| `wiki_corpus.jsonl` | Passage text returned by `/retrieve` |
| `wiki_webpages.jsonl` | Full page content returned by `/access` and used by data construction |
| `wikilinks.json` | Wikipedia link graph used by evidence-tree construction |
| `e5-base-v2/` | Local E5 query encoder used by the retriever |

If `e5-base-v2/` is not included in your local asset copy, download
`intfloat/e5-base-v2` from Hugging Face and place it at
`$WIKI_RETRIEVER_ROOT/e5-base-v2/`.

The OpenScholar datastore directory must contain the released PES2O v3 datastore
files expected by the vendored OpenScholar config. In practice this should
include a `pes2o_v3/` subtree with passage, embedding, and index files, for
example:

```text
$OPENSCHOLAR_DATA_ROOT/
  pes2o_v3/
    passages/
    embeddings/
    index/          # or index files/directories released with the datastore
```

The exact file names inside the OpenScholar datastore are controlled by the
released `OpenScholar-DataStore-V3` package. Keep that directory structure
unchanged after download.

## How the Indexes Were Built

You do not need to rebuild the indexes if you download the two Hugging Face
asset repositories above. The notes below document how the released assets were
encoded.

### Wikipedia

The Wikipedia dense index follows the ASearcher local retriever setup. The
published `e5.index/e5_Flat.index` is built from `wiki_corpus.jsonl` with E5
passage embeddings:

```bash
python3 utils/index_builder.py \
  --retrieval_method e5 \
  --model_path /path/to/e5-base-v2 \
  --corpus_path /path/to/ASearcher-Local-Knowledge/wiki_corpus.jsonl \
  --save_dir /path/to/ASearcher-Local-Knowledge/e5.index \
  --use_fp16 \
  --max_length 256 \
  --batch_size 512 \
  --pooling_method mean \
  --faiss_type Flat \
  --save_embedding
```

Encoding details:

- Each passage is read from the `contents` field in `wiki_corpus.jsonl`.
- For E5, passage text is encoded as `passage: <text>`.
- The encoder uses mean pooling, L2 normalization, and fp16 inference.
- FAISS uses a `Flat` inner-product index, saved as `e5_Flat.index`.
- At serving time, each user query is encoded as `query: <query>` with the same
  E5 model, then searched against `e5_Flat.index`.
- `wiki_webpages.jsonl` and `wikilinks.json` are not embedded. They are used for
  page browsing and evidence-tree expansion.

### OpenScholar

For normal reproduction, use the released
`OpenSciLM/OpenScholar-DataStore-V3` datastore directly. It already contains the
PES2O v3 retrieval assets needed by the OpenScholar server.

The original local rebuild path used the OpenScholar retrieval-scaling pipeline
and split the work into two stages:

1. **Embedding stage**: encode PES2O passages into 16 embedding shards.
2. **Indexing stage**: build FAISS indexes from those embeddings.

The local script used for that rebuild had this shape:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MKL_THREADING_LAYER=GNU
export KMP_DUPLICATE_LIB_OK=TRUE

datastore_raw_data_path=/path/to/OpenScholar-DataStore-V3/passages
num_shards=16

# Embedding, one shard at a time or in parallel if resources allow.
for SLURM_ARRAY_TASK_ID in {0..15}; do
  PYTHONPATH=. python ric/main_ric.py \
    --config-name=pes2o_v3 \
    tasks.datastore.embedding=true \
    tasks.datastore.index=false \
    datastore.raw_data_path=$datastore_raw_data_path \
    datastore.embedding.num_shards=$num_shards \
    datastore.embedding.shard_ids=[$SLURM_ARRAY_TASK_ID]
done

# Indexing, after embeddings are ready.
PYTHONPATH=. python ric/main_ric.py \
  --config-name=ivf_pq \
  tasks.datastore.embedding=false \
  tasks.datastore.index=true \
  datastore.raw_data_path=$datastore_raw_data_path
```

The resource-saving compromises were:

- The corpus was split into `16` shards instead of encoding/indexing all passages
  as one monolithic job.
- The embedding jobs can be run sequentially by removing background parallelism
  when GPU memory is insufficient.
- The embedding code uses a smaller SentenceTransformer batch size (`64`) in the
  OOM-prone path instead of the larger default.
- The index uses `IVFPQ` compression instead of a full flat index.
- The local IVFPQ config used `projection_size=768`, `ncentroids=8192`,
  `n_subquantizers=16`, `n_bits=8`, `probe=512`, and
  `sample_train_size=6000000`.
- Index construction can be split by `index_shard_ids`; for low-resource
  machines, build shard groups sequentially and serve only the shards assigned
  to that worker.

## Repository Layout

```text
retrievers/
  wiki/
    launch_local_server.sh
    code/local_retrieval_server.py

  openscholar/
    start_single_node.sh
    retrieval-scaling-main/
```

The Hugging Face repositories above provide data assets only. The server code
used by DeepRubric is included in this `retrievers/` directory.

## Endpoints

DeepRubric expects these local services:

| Service | Default endpoint | Used by |
| --- | --- | --- |
| Wikipedia search | `http://localhost:8888/retrieve` | data construction and training `search` tool |
| Wikipedia browse | `http://localhost:8888/access` | training `browse` tool |
| OpenScholar search | `http://localhost:8000/search` | data construction and training `scholar` tool |

The training code should point to the same endpoints:

```bash
export SEARCH_URL=http://localhost:8888/retrieve
export BROWSE_URL=http://localhost:8888/access
export SCHOLAR_URL=http://localhost:8000/search
```

## Start Both Retrievers

From the repository root:

```bash
export WIKI_RETRIEVER_ROOT=/path/to/ASearcher-Local-Knowledge
export OPENSCHOLAR_DATA_ROOT=/path/to/OpenScholar-DataStore-V3

bash scripts/start_retrievers.sh
```

Keep this process running while constructing data or training. Logs are written
to:

```text
outputs/retriever_logs/wiki.log
outputs/retriever_logs/openscholar.log
```

## Start Separately

Wikipedia retriever:

```bash
export WIKI_RETRIEVER_ROOT=/path/to/ASearcher-Local-Knowledge
export CUDA_VISIBLE_DEVICES=0

bash retrievers/wiki/launch_local_server.sh
```

OpenScholar retriever:

```bash
export OPENSCHOLAR_DATA_ROOT=/path/to/OpenScholar-DataStore-V3
export CUDA_VISIBLE_DEVICES=1

bash retrievers/openscholar/start_single_node.sh
```

You can change ports with:

```bash
export WIKI_RETRIEVER_PORT=8888
```

OpenScholar uses port `8000` in the vendored FastAPI server by default. Do not
run your LLM API server on port `8000` unless you also change the OpenScholar
service port and update `SCHOLAR_URL`.

## Smoke Test

After startup, test the endpoints:

```bash
curl -s http://localhost:8888/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Where was Marie Curie born?"],"topk":3}'

curl -s http://localhost:8888/access \
  -H 'Content-Type: application/json' \
  -d '{"urls":["https://en.wikipedia.org/wiki/Marie_Curie"]}'

curl -s http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"retrieval augmented generation","n_docs":3,"domains":"pes2o_v3"}'
```

If these requests work, the retriever side of the DeepRubric loop is ready for
data construction and training.
