#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm


def _is_nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _extract_qa_item(record: Dict[str, Any], include_drop: bool = False) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None

    question = record.get("question", "")
    rubrics = record.get("rubrics", [])

    audit = record.get("_audit")
    if isinstance(audit, dict):
        decision = str(audit.get("decision", "")).strip().upper()
        if not include_drop and decision == "DROP":
            return None

        revised_question = audit.get("revised_question")
        revised_rubrics = audit.get("revised_rubrics")

        # Prefer repaired content for REVISE samples when available.
        if _is_nonempty_str(revised_question):
            question = revised_question
        elif not _is_nonempty_str(question):
            question = audit.get("generated_question", audit.get("question", ""))

        if isinstance(revised_rubrics, list) and revised_rubrics:
            rubrics = revised_rubrics
        elif not isinstance(rubrics, list) or not rubrics:
            rubrics = audit.get("generated_rubrics", [])

    if not _is_nonempty_str(question):
        return None
    if not isinstance(rubrics, list):
        rubrics = []

    return {
        "question": question,
        "answer": "",
        "golden_answers": rubrics,
        "data_source": "long_form",
        "reward_model": "rubric",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert recursive QA tree JSONL (raw or filtered) to QA eval JSONL.")
    parser.add_argument(
        "--input-dir",
        default="outputs/verified_trees",
        help="Directory containing source .jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory.",
    )
    parser.add_argument(
        "--output-file",
        default="question_1001.jsonl",
        help="Output filename under output-dir.",
    )
    parser.add_argument(
        "--include-drop",
        action="store_true",
        help="Include samples with _audit.decision=DROP (default: skip).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file

    jsonl_files = sorted([p for p in input_dir.rglob("*.jsonl") if p.is_file()])
    print(f"Found {len(jsonl_files)} jsonl files")

    total_lines = 0
    total_written = 0
    skipped_drop = 0
    skipped_invalid = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for input_path in jsonl_files:
            print(f"\nProcessing file: {input_path.name}")
            with input_path.open("r", encoding="utf-8") as fin:
                for line in tqdm(fin, desc=f"Processing {input_path.name}"):
                    total_lines += 1
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        skipped_invalid += 1
                        continue

                    item = _extract_qa_item(obj, include_drop=args.include_drop)
                    if item is None:
                        if isinstance(obj, dict) and isinstance(obj.get("_audit"), dict):
                            if str(obj["_audit"].get("decision", "")).strip().upper() == "DROP":
                                skipped_drop += 1
                            else:
                                skipped_invalid += 1
                        else:
                            skipped_invalid += 1
                        continue

                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_written += 1

    print("\nDone")
    print(f"Total input lines: {total_lines}")
    print(f"Total written QA items: {total_written}")
    print(f"Skipped DROP items: {skipped_drop}")
    print(f"Skipped invalid items: {skipped_invalid}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
