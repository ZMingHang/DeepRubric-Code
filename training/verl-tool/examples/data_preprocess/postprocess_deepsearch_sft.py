#!/usr/bin/env python3
"""Post-process deep-search SFT data with GPT-4.1.

This script repairs missing <think> blocks in assistant turns, rewrites the
final answer to improve citation quality and completeness, and writes updated
JSONL/parquet files for downstream SFT.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
CITE_RE = re.compile(r'<cite\s+id="([^"]+)">.*?</cite>', re.DOTALL)
CITE_SPAN_RE = re.compile(r'<cite\s+id="([^"]+)">(.*?)</cite>', re.DOTALL)
EVIDENCE_ID_RE = re.compile(r'<(?:snippet|webpage)\s+id="?([^"\s>]+)"?\s*>', re.DOTALL)

REPAIR_TURN_SYSTEM_PROMPT = """You are repairing one assistant turn in a deep-research tool-use trajectory.

Return JSON only with this schema:
{
  "assistant_message": "...",
  "reason": "..."
}

Rules:
1. Output exactly one standalone <think>...</think> block.
2. Then output exactly one standalone <tool_call>...</tool_call> block.
3. Preserve the tool name and arguments from the draft whenever they are already valid.
4. Do not add any extra text outside those blocks.
"""

REWRITE_FINAL_SYSTEM_PROMPT = """You are improving the final answer in a deep-research trajectory for SFT training quality.

Return JSON only with this schema:
{
  "assistant_message": "...",
  "reason": "..."
}

Rules:
1. Output exactly one standalone <think>...</think> block followed by exactly one standalone <answer>...</answer> block.
2. Do not output any <tool_call>.
3. Use only citation IDs that appear in the provided tool evidence.
4. Add or improve citations for factual claims whenever supported evidence exists.
5. Improve completeness, synthesis, and clarity, but do not invent facts or citations.
6. If evidence is weak or conflicting, explicitly say so in the answer.
"""

UNIFIED_RECORD_EDITOR_SYSTEM_PROMPT = """You are post-processing all assistant responses in one deep-research SFT sample for high-quality supervised fine-tuning.

Background:
- The target model is a tool-using deep-research assistant.
- The training sample is a multi-turn trajectory consisting of system, user, tool-response, and assistant messages.
- The assistant is expected to reason explicitly in a visible <think> block, use tools through <tool_call>, and deliver a final evidence-grounded answer inside <answer>.
- These post-processed samples are intended to teach the model better habits: proper format, better evidence use, stronger citations, and more complete final answers.

Your job:
- Inspect the whole sample carefully.
- Identify what is wrong or weak in each assistant response.
- Rewrite every assistant response, not just the final one.
- Improve intermediate tool-use turns so they have better visible reasoning and cleaner formatting.
- Improve the final answer when it is thin, poorly cited, weakly synthesized, or structurally weak.

Why this matters:
- Missing or malformed <think> blocks make the sample inconsistent for SFT.
- Bad citations are especially harmful: the answer may cite nothing, cite too little, or cite the wrong evidence.
- A weak final answer can teach the model to be shallow, under-supported, or too confident.

Return JSON only with this schema:
{
  "issues": ["..."],
  "attention_points": ["..."],
  "edited_assistant_messages": [
    {
      "assistant_turn_index": 0,
      "assistant_message": "..."
    }
  ]
}

What to check:
1. Rewrite the assistant response.
2. Every assistant response should begin with exactly one standalone <think>...</think> block.
3. Every non-final assistant response should then contain exactly one standalone <tool_call>...</tool_call> block.
4. The final assistant response should then contain exactly one standalone <answer>...</answer> block and no <tool_call>.
5. The <think> block should reflect genuine problem-solving thought: it should explain how the assistant interprets the question, plans the answer, weighs evidence, notices uncertainty or conflicts, and decides what to do next.
6. The final answer should be more complete, better structured, and better synthesized if the current answer is weak.
7. Citation quality is critical:
   - use only citation IDs that appear in the provided tool evidence
   - add citations for nontrivial factual claims when evidence exists
   - never invent IDs
   - if evidence is weak or conflicting, say so instead of fabricating confidence
   - each citation must wrap a meaningful claim span in natural prose, not only punctuation, not only a trailing period, and not an empty filler span
   - avoid bad patterns like <cite id="X">.</cite> or a paragraph that ends with a detached citation marker
   - prefer citations embedded directly around the supported clause or sentence fragment
8. The <think> block should help the model learn how to plan, reflect, and organize evidence before answering, rather than being a generic placeholder.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process deep-search SFT outputs with GPT-4.1.")
    parser.add_argument("--input", required=True, help="Input file or directory.")
    parser.add_argument("--output", required=True, help="Output file or directory.")
    parser.add_argument("--model", default="gpt-4.1", help="Editor model name.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI-compatible API key.")
    parser.add_argument("--concurrency", type=int, default=4, help="How many records to post-process concurrently.")
    parser.add_argument("--request-timeout", type=int, default=300, help="Timeout seconds for model requests.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens for rewrite requests.")
    parser.add_argument("--max-evidence-chars", type=int, default=32000, help="Evidence chars passed to the editor.")
    parser.add_argument("--max-history-chars", type=int, default=12000, help="History chars passed to the editor.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries when edited output is invalid.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Only process the first N records when > 0.")
    parser.add_argument("--debug", action="store_true", help="Print debug information.")
    parser.add_argument(
        "--process-names",
        default="train.jsonl,test.jsonl,usable_records.jsonl,train.parquet,test.parquet",
        help="When --input is a directory, process these comma-separated top-level filenames if present.",
    )
    return parser.parse_args()


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                elif "text" in item:
                    chunks.append(str(item.get("text", "")))
                else:
                    chunks.append(str(item))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(content)


def has_valid_think(text: str) -> bool:
    match = THINK_RE.search(text)
    if match is None:
        return False
    next_positions = [pos for pos in (text.find("<tool_call>"), text.find("<answer>")) if pos >= 0]
    if next_positions and match.end() > min(next_positions):
        return False
    return True


def has_tool_call(text: str) -> bool:
    return TOOL_CALL_RE.search(text) is not None


def has_answer(text: str) -> bool:
    return ANSWER_RE.search(text) is not None


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars - 32
    if keep <= 0:
        return text[:max_chars]
    return text[:keep] + "\n\n[TRUNCATED]\n"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = stripped.strip()
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first < 0 or last < 0 or last <= first:
        raise ValueError("No JSON object found in editor output.")
    return json.loads(stripped[first : last + 1])


def extract_available_ids_from_messages(messages: list[dict[str, Any]]) -> set[str]:
    available: set[str] = set()
    for message in messages:
        if message.get("role") != "user":
            continue
        content = normalize_message_content(message.get("content", ""))
        for match in EVIDENCE_ID_RE.findall(content):
            available.add(match.strip())
    return available


def extract_cited_ids(text: str) -> set[str]:
    cited: set[str] = set()
    for group in CITE_RE.findall(text):
        for item in group.split(","):
            item = item.strip()
            if item:
                cited.add(item)
    return cited


def has_meaningful_citation_spans(text: str) -> bool:
    spans = CITE_SPAN_RE.findall(text)
    if not spans:
        return False

    for _, span_text in spans:
        span_text = span_text.strip()
        # Reject citations that wrap only punctuation / symbols / whitespace.
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", span_text):
            return False
    return True


def validate_tool_turn(text: str) -> bool:
    return has_valid_think(text) and has_tool_call(text) and not has_answer(text)


def validate_final_answer(text: str, available_ids: set[str]) -> bool:
    if not has_valid_think(text):
        return False
    if not has_answer(text):
        return False
    if has_tool_call(text):
        return False
    cited_ids = extract_cited_ids(text)
    if not has_meaningful_citation_spans(text):
        return False
    if available_ids:
        if not cited_ids:
            return False
        if not cited_ids.issubset(available_ids):
            return False
    return True


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "records", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    raise ValueError(f"Unsupported JSON structure in {path}")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Reading parquet requires pandas.") from exc
    dataframe = pd.read_parquet(path)
    return dataframe.to_dict(orient="records")


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Writing parquet requires pandas.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".json":
        return read_json(path)
    if path.suffix.lower() == ".parquet":
        return read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def dump_records(path: Path, records: list[dict[str, Any]]) -> None:
    if path.suffix.lower() == ".jsonl":
        write_jsonl(path, records)
        return
    if path.suffix.lower() == ".json":
        write_json(path, records)
        return
    if path.suffix.lower() == ".parquet":
        write_parquet(path, records)
        return
    raise ValueError(f"Unsupported file type: {path}")


def resolve_paths(input_path: Path, output_path: Path, process_names: list[str]) -> list[tuple[Path, Path]]:
    if input_path.is_file():
        return [(input_path, output_path)]

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    pairs: list[tuple[Path, Path]] = []
    for name in process_names:
        source = input_path / name
        if source.exists():
            pairs.append((source, output_path / name))
    if not pairs:
        raise ValueError(f"No matching files found in {input_path}")
    return pairs


class DeepSearchPostprocessor:
    def __init__(self, args: argparse.Namespace):
        if not args.api_key:
            raise ValueError("Missing API key. Set --api-key or OPENAI_API_KEY.")
        client_kwargs: dict[str, Any] = {"api_key": args.api_key}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        self.args = args
        self.client = AsyncOpenAI(**client_kwargs)
        self.semaphore = asyncio.Semaphore(max(1, args.concurrency))

    def _debug(self, message: str) -> None:
        if self.args.debug:
            print(f"[DEBUG] {message}", flush=True)

    async def process_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tasks = [asyncio.create_task(self.process_record(record)) for record in records]
        return await asyncio.gather(*tasks)

    async def process_record(self, record: dict[str, Any]) -> dict[str, Any]:
        async with self.semaphore:
            return await self._process_record(record)

    async def _process_record(self, record: dict[str, Any]) -> dict[str, Any]:
        new_record = copy.deepcopy(record)
        messages = new_record.get("messages")
        if not isinstance(messages, list) or not messages:
            new_record["_postprocess"] = {"status": "skipped", "reason": "missing_messages"}
            return new_record

        assistant_indices = [idx for idx, msg in enumerate(messages) if msg.get("role") == "assistant"]
        if not assistant_indices:
            new_record["_postprocess"] = {"status": "skipped", "reason": "no_assistant_turns"}
            return new_record

        question = str(new_record.get("question") or "")
        if not question:
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if user_messages:
                question = normalize_message_content(user_messages[0].get("content", ""))

        final_index = assistant_indices[-1]
        available_ids = extract_available_ids_from_messages(messages)
        self._debug(
            f"sample_id={new_record.get('sample_id', '')} assistant_turns={len(assistant_indices)} "
            f"available_ids={len(available_ids)}"
        )
        rewrite_result = await self._rewrite_record(question, messages, assistant_indices, available_ids)

        issues: list[str] = []
        attention_points: list[str] = []
        final_rewritten = False
        repaired_turns: list[int] = []
        if rewrite_result is not None:
            issues = [str(item) for item in rewrite_result.get("issues", []) if str(item).strip()]
            attention_points = [str(item) for item in rewrite_result.get("attention_points", []) if str(item).strip()]
            edited_items = rewrite_result.get("edited_assistant_messages", [])
            if isinstance(edited_items, list):
                for item in edited_items:
                    try:
                        assistant_turn_index = int(item.get("assistant_turn_index"))
                    except Exception:
                        continue
                    if assistant_turn_index < 0 or assistant_turn_index >= len(assistant_indices):
                        continue

                    message_index = assistant_indices[assistant_turn_index]
                    revised = normalize_message_content(item.get("assistant_message", "")).strip()
                    if not revised:
                        continue

                    is_final_turn = message_index == final_index
                    is_valid = (
                        validate_final_answer(revised, available_ids)
                        if is_final_turn
                        else validate_tool_turn(revised)
                    )
                    if not is_valid:
                        continue

                    if normalize_message_content(messages[message_index].get("content", "")) != revised:
                        messages[message_index]["content"] = revised
                        repaired_turns.append(message_index)
                        if is_final_turn:
                            final_rewritten = True
                        self._debug(
                            f"sample_id={new_record.get('sample_id', '')} patched_turn={assistant_turn_index} "
                            f"is_final={is_final_turn}"
                        )

        final_text = normalize_message_content(messages[final_index].get("content", ""))
        new_record["_postprocess"] = {
            "status": "ok",
            "model": self.args.model,
            "repaired_turn_indices": repaired_turns,
            "final_rewritten": final_rewritten,
            "issues": issues,
            "attention_points": attention_points,
            "available_ids": sorted(available_ids),
            "cited_ids": sorted(extract_cited_ids(final_text)),
            "final_valid": validate_final_answer(final_text, available_ids),
            "all_assistant_turns_have_think": all(
                has_valid_think(normalize_message_content(messages[idx].get("content", ""))) for idx in assistant_indices
            ),
        }
        return new_record

    async def _rewrite_record(
        self,
        question: str,
        messages: list[dict[str, Any]],
        assistant_indices: list[int],
        available_ids: set[str],
    ) -> dict[str, Any] | None:
        assistant_turn_blocks: list[str] = []
        for assistant_turn_index, message_index in enumerate(assistant_indices):
            role_name = "final_answer_turn" if assistant_turn_index == len(assistant_indices) - 1 else "tool_use_turn"
            content = normalize_message_content(messages[message_index].get("content", ""))
            assistant_turn_blocks.append(
                f"Assistant turn {assistant_turn_index} ({role_name}):\n{content}"
            )

        evidence_blocks = [
            normalize_message_content(msg.get("content", ""))
            for msg in messages
            if msg.get("role") == "user" and normalize_message_content(msg.get("content", "")).startswith("<tool_response>")
        ]
        assistant_turns_text = truncate_text("\n\n".join(assistant_turn_blocks), self.args.max_history_chars)
        evidence_text = truncate_text("\n\n".join(evidence_blocks), self.args.max_evidence_chars)

        user_prompt = (
            f"Question:\n{question}\n\n"
            "Assistant turns:\n"
            f"{assistant_turns_text or '[NONE]'}\n\n"
            "Tool evidence:\n"
            f"{evidence_text}\n\n"
            "Available citation IDs:\n"
            f"{', '.join(sorted(available_ids)) if available_ids else '[NONE]'}\n\n"
            "Check the whole sample, list the main issues and attention points, then return corrected replacements for every assistant response."
        )
        self._debug(f"editor_prompt:\n{user_prompt}")

        for _ in range(self.args.max_retries):
            text = await self._chat_json(UNIFIED_RECORD_EDITOR_SYSTEM_PROMPT, user_prompt)
            if text is None:
                continue
            self._debug(f"editor_raw_output:\n{text}")
            try:
                obj = extract_json_object(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                edited_items = obj.get("edited_assistant_messages", [])
                if not isinstance(edited_items, list):
                    continue
                returned_indices = set()
                for item in edited_items:
                    try:
                        returned_indices.add(int(item.get("assistant_turn_index")))
                    except Exception:
                        pass
                if returned_indices != set(range(len(assistant_indices))):
                    self._debug(
                        f"editor_output_missing_turns expected={list(range(len(assistant_indices)))} "
                        f"got={sorted(returned_indices)}"
                    )
                    continue
                return obj
        return None

    async def _repair_tool_turn(self, question: str, draft_turn: str) -> str | None:
        user_prompt = (
            f"Question:\n{question}\n\n"
            "Draft assistant tool-use turn:\n"
            f"{draft_turn}\n\n"
            "Repair this turn."
        )
        for _ in range(self.args.max_retries):
            text = await self._chat_json(REPAIR_TURN_SYSTEM_PROMPT, user_prompt)
            if text is None:
                continue
            try:
                obj = extract_json_object(text)
            except Exception:
                continue
            revised = normalize_message_content(obj.get("assistant_message", "")).strip()
            if validate_tool_turn(revised):
                return revised
        return None

    async def _rewrite_final_answer(
        self,
        question: str,
        messages: list[dict[str, Any]],
        final_index: int,
        available_ids: set[str],
        draft_answer: str,
    ) -> str | None:
        evidence_blocks = [
            normalize_message_content(msg.get("content", ""))
            for msg in messages
            if msg.get("role") == "user" and normalize_message_content(msg.get("content", "")).startswith("<tool_response>")
        ]
        history_blocks = [
            normalize_message_content(messages[idx].get("content", ""))
            for idx in range(final_index)
            if messages[idx].get("role") == "assistant"
        ]
        evidence_text = truncate_text("\n\n".join(evidence_blocks), self.args.max_evidence_chars)
        history_text = truncate_text("\n\n".join(history_blocks), self.args.max_history_chars)
        available_ids_text = ", ".join(sorted(available_ids)) if available_ids else "[NONE]"

        user_prompt = (
            f"Question:\n{question}\n\n"
            "Existing final assistant turn:\n"
            f"{draft_answer}\n\n"
            "Previous assistant turns:\n"
            f"{history_text or '[NONE]'}\n\n"
            "Tool evidence:\n"
            f"{evidence_text or '[NONE]'}\n\n"
            "Available citation IDs:\n"
            f"{available_ids_text}\n\n"
            "Rewrite the final assistant turn to maximize quality."
        )
        for _ in range(self.args.max_retries):
            text = await self._chat_json(REWRITE_FINAL_SYSTEM_PROMPT, user_prompt)
            if text is None:
                continue
            try:
                obj = extract_json_object(text)
            except Exception:
                continue
            revised = normalize_message_content(obj.get("assistant_message", "")).strip()
            if validate_final_answer(revised, available_ids):
                return revised
        return None

    async def _chat_json(self, system_prompt: str, user_prompt: str) -> str | None:
        response = await self.client.chat.completions.create(
            model=self.args.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=self.args.max_tokens,
            timeout=self.args.request_timeout,
        )
        return normalize_message_content(response.choices[0].message.content).strip()


async def process_file(processor: DeepSearchPostprocessor, input_file: Path, output_file: Path) -> dict[str, Any]:
    records = load_records(input_file)
    if processor.args.max_samples > 0:
        records = records[: processor.args.max_samples]
    updated_records = await processor.process_records(records)
    dump_records(output_file, updated_records)

    valid_count = 0
    for record in updated_records:
        post = record.get("_postprocess", {})
        if post.get("final_valid"):
            valid_count += 1

    summary = {
        "input": str(input_file),
        "output": str(output_file),
        "records": len(updated_records),
        "final_valid_records": valid_count,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


async def async_main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    input_is_dir = input_path.is_dir()
    process_names = [name.strip() for name in args.process_names.split(",") if name.strip()]

    pairs = resolve_paths(input_path, output_path, process_names)
    processor = DeepSearchPostprocessor(args)

    summaries = []
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        summaries.append(await process_file(processor, source, target))

    meta_path = output_path / "postprocess_summary.json" if input_is_dir or len(pairs) > 1 else output_path.with_suffix(".summary.json")
    meta_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
