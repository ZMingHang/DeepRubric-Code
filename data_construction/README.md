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

Run after the retrievers are available:

```bash
python data_construction/recursive_qa_agent_v4.py \
  --input data/seeds/wiki_seed_questions.jsonl \
  --output data/intermediate/wiki_trees.jsonl

python data_construction/recursive_qa_agent_v44.py \
  --input data/seeds/scholar_seed_questions.jsonl \
  --output data/intermediate/scholar_trees.jsonl
```

Verify and export retained samples:

```bash
python data_construction/recursive_qa_quality_filter.py \
  --input data/intermediate/wiki_trees.jsonl \
  --output data/processed/wiki_verified.jsonl

python data_construction/recursive_qa_quality_filter.py \
  --input data/intermediate/scholar_trees.jsonl \
  --output data/processed/scholar_verified.jsonl
```

Use `OPENAI_BASE_URL` and `OPENAI_API_KEY` or the `DEEPRUBRIC_LLM_*` variables to point the construction scripts to your OpenAI-compatible LLM service.

