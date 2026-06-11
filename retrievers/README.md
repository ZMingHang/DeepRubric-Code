# Retrievers

This directory contains launch wrappers for the local retrievers used by DeepRubric.

The code release does not vendor the large retrieval indexes. Download the external assets described in the root `README.md`, then point the scripts to your local copies with environment variables.

## Layout

```text
retrievers/
  wiki/launch_local_server.sh        # Wikipedia retriever wrapper
  openscholar/start_single_node.sh   # OpenScholar retriever wrapper
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
WIKI_CODE_ROOT=/path/to/ASearcher-main \
WIKI_CORPUS_PATH=/path/to/ASearcher-Local-Knowledge/wiki-18.jsonl \
WIKI_INDEX_PATH=/path/to/ASearcher-Local-Knowledge/e5_Flat.index \
bash retrievers/wiki/launch_local_server.sh

OPENSCHOLAR_DATASTORE=/path/to/OpenScholar-DataStore-V3 \
OPENSCHOLAR_MODEL=/path/to/retriever-or-model \
bash retrievers/openscholar/start_single_node.sh
```

