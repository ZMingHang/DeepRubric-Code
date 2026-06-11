# Data Conversion

This directory converts verified DeepRubric construction outputs into verl-tool training data.

## Main Files

| File | Role |
| --- | --- |
| `read_form_tree_revise.py` | Reads verified tree outputs and normalizes them into training examples. |
| `preprocess_search_r1_dataset_tool.py` | Converts normalized examples into verl-tool parquet files. |

## Inputs and Outputs

Expected input:

```text
data/processed/wiki_verified.jsonl
data/processed/scholar_verified.jsonl
```

Expected output:

```text
training/verl-tool/data/deeprubric/train.parquet
training/verl-tool/data/deeprubric/test.parquet
```

## Commands

From the repository root:

```bash
python data_conversion/read_form_tree_revise.py \
  --input data/processed/wiki_verified.jsonl data/processed/scholar_verified.jsonl \
  --output data/processed/deeprubric_train.jsonl
```

Then build verl-tool parquet data:

```bash
python data_conversion/preprocess_search_r1_dataset_tool.py \
  --input_file data/processed/deeprubric_train.jsonl \
  --local_dir training/verl-tool/data/deeprubric \
  --data_source deeprubric
```

The resulting dataset is consumed by `training/verl-tool/examples/train/deepsearch/train_4b_tool.sh` or `train_8b_tool.sh`.

