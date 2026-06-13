# Retrievers

This directory contains the local retriever services used by DeepRubric.

The release vendors the lightweight serving code needed to start the retrievers. It does **not** vendor the large corpora, embedding models, FAISS indexes, or OpenScholar datastore files. Download the external data assets described in the root `README.md`, then point the scripts to your local copies with environment variables.

## Layout

```text
retrievers/
  wiki/
    launch_local_server.sh           # Starts the vendored ASearcher local FastAPI server
    code/local_retrieval_server.py   # Vendored Wikipedia retrieval server code

  openscholar/
    start_single_node.sh             # Starts the vendored OpenScholar FastAPI worker
    retrieval-scaling-main/          # Vendored OpenScholar retrieval-scaling server code
```

## Endpoints

DeepRubric expects the same retrievers during data construction and training:

| Retriever | Default endpoint | Used by |
| --- | --- | --- |
| Wikipedia search | `http://localhost:8888/retrieve` | evidence-tree construction and training `search` tool |
| Wikipedia browse | `http://localhost:8888/access` | training `browse` tool |
| OpenScholar search | `http://localhost:8000/search` | evidence-tree construction and training `scholar` tool |

## Start

Start both retrievers from the repository root:

```bash
bash scripts/start_retrievers.sh
```

Or start each retriever separately:

```bash
bash retrievers/wiki/launch_local_server.sh
bash retrievers/openscholar/start_single_node.sh
```

Override paths as needed:

```bash
WIKI_RETRIEVER_ROOT=/path/to/ASearcher-Local-Knowledge \
bash retrievers/wiki/launch_local_server.sh

OPENSCHOLAR_DATA_ROOT=/path/to/OpenScholar-DataStore-V3 \
bash retrievers/openscholar/start_single_node.sh
```

The Hugging Face links provide data assets only. The server code used by these scripts is included in this directory.
