# Data and Retriever Assets

The release does not include large corpora or FAISS indexes.

## Wikipedia

Download:

```text
https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge
```

Expected layout:

```text
/path/to/ASearcher-Local-Knowledge/
  e5.index/e5_Flat.index
  wiki_corpus.jsonl
  wiki_webpages.jsonl
  wikilinks.json
  e5-base-v2/
```

Set:

```bash
export WIKI_RETRIEVER_ROOT=/path/to/ASearcher-Local-Knowledge
```

The Wikipedia serving code is vendored at `retrievers/wiki/code/local_retrieval_server.py`; the corpus, index, webpages, and E5 model are read from `WIKI_RETRIEVER_ROOT`.

## OpenScholar

Download:

```text
https://huggingface.co/datasets/OpenSciLM/OpenScholar-DataStore-V3
```

Expected layout depends on the OpenScholar retrieval-scaling config. The evidence-tree script expects passage JSONL files such as:

```text
/path/to/OpenScholar-DataStore-V3/passages/raw_passages-0-of-16.jsonl
```

Set:

```bash
export OPENSCHOLAR_DATA_ROOT=/path/to/OpenScholar-DataStore-V3
```

The OpenScholar retrieval-scaling serving code is vendored at `retrievers/openscholar/retrieval-scaling-main/`. `retrievers/openscholar/start_single_node.sh` passes `OPENSCHOLAR_DATA_ROOT` into the vendored Hydra config for your datastore and index paths.

## Default Endpoints

```bash
export WIKI_RETRIEVER_URL=http://localhost:8888/retrieve
export WIKI_BROWSE_URL=http://localhost:8888/access
export OPENSCHOLAR_RETRIEVER_URL=http://localhost:8000/search
export SEARCH_URL="$WIKI_RETRIEVER_URL"
export BROWSE_URL="$WIKI_BROWSE_URL"
export SCHOLAR_URL="$OPENSCHOLAR_RETRIEVER_URL"
```

Use a different port for the OpenAI-compatible LLM server, for example `http://localhost:8008/v1`, so it does not conflict with the OpenScholar retriever.
