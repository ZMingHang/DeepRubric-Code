# Data Construction

This directory builds and filters DeepRubric query-rubric samples from retrieved evidence.

The typical flow is:

```text
seed questions
  -> evidence-tree expansion
  -> query and rubric synthesis
  -> LLM-based quality verification
  -> verified JSONL samples for conversion
```

## Main Files

| File | Role |
| --- | --- |
| `recursive_qa_agent_v4.py` | Builds Wikipedia-grounded evidence trees. |
| `recursive_qa_agent_v44.py` | Builds OpenScholar-grounded evidence trees. |
| `qa_synthesis_agent_openai_v9.py` | Synthesizes research queries and rubric criteria from evidence trees. |
| `recursive_qa_quality_filter.py` | Verifies generated samples and emits KEEP/REVISE/DROP decisions. |
| `prompt.py` | Prompt templates used by the construction scripts. |

## Minimal Commands

From the repository root, configure the paths, model IDs, and endpoints in
[Quick Start](../README.md#quick-start), then run after the retrievers are available:

```bash
python data_construction/recursive_qa_agent_v4.py \
  --pages "$WIKI_RETRIEVER_ROOT/wiki_webpages.jsonl" \
  --links "$WIKI_RETRIEVER_ROOT/wikilinks.json" \
  --save outputs/wiki_evidence_trees \
  --retriever-url "$WIKI_RETRIEVER_URL" \
  --qa-model "$QA_MODEL" \
  --extract-model "$EXTRACT_MODEL" \
  --n 1000

python data_construction/recursive_qa_agent_v44.py \
  --pages "$OPENSCHOLAR_DATA_ROOT/passages/raw_passages-0-of-16.jsonl" \
  --save outputs/openscholar_evidence_trees \
  --retriever-url "$OPENSCHOLAR_RETRIEVER_URL" \
  --qa-model "$QA_MODEL" \
  --extract-model "$EXTRACT_MODEL" \
  --n 1000
```

Verify and export retained samples:

```bash
mkdir -p outputs/verified_trees

python data_construction/recursive_qa_quality_filter.py \
  --input outputs/wiki_evidence_trees \
  --output-dir outputs/wiki_audited \
  --revised-output outputs/verified_trees/wiki_verified.jsonl \
  --include-keep-in-revised \
  --judge-model "$JUDGE_MODEL"

python data_construction/recursive_qa_quality_filter.py \
  --input outputs/openscholar_evidence_trees \
  --output-dir outputs/scholar_audited \
  --revised-output outputs/verified_trees/scholar_verified.jsonl \
  --include-keep-in-revised \
  --judge-model "$JUDGE_MODEL"
```

Use `OPENAI_BASE_URL` and `OPENAI_API_KEY` for the shared LLM service. The synthesis client also supports `DEEPRUBRIC_LLM_BASE_URL` and `DEEPRUBRIC_LLM_API_KEY`; the verifier uses its `--judge-base-url` and `--judge-api-key` options or the `OPENAI_*` variables. See [Environment Setup](../docs/ENVIRONMENT.md).
