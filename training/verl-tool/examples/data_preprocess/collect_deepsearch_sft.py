#!/usr/bin/env python3
"""Collect multi-turn deep-search SFT data with GPT-5.1 and a local tool server.

This script bootstraps multi-turn SFT trajectories for a tag-based deep research
assistant. It drives a model such as `gpt-5.1` with the provided system prompt,
executes `<tool_call>...</tool_call>` blocks against the local `verl_tool`
server, and saves successful trajectories as `messages` parquet files that can
be consumed by `verl`'s `MultiTurnSFTDataset`.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from openai import AsyncOpenAI


DEFAULT_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. 

For every request, you MUST perform iterative, multi-step information gathering. You should search as broadly and deeply as possible, gather as much relevant information as is reasonably available, and synthesize evidence from credible, diverse sources to deliver a comprehensive, accurate, and objective response.

You MUST NOT rely on a single search. Instead:
- Always perform multiple rounds of search using different queries, perspectives, or keyword variations.
- After each search, analyze gaps, uncertainties, or missing aspects, and issue follow-up searches.
- Continue searching until the key aspects of the question are sufficiently covered.
- You should typically perform at least 2–4 rounds of tool calls before producing the final answer, unless the question is extremely simple.

A response is NOT sufficient if:
- Only one source or one perspective is used
- Key concepts in the question are not individually investigated
- There is no cross-source verification

You should make a strong effort to explore multiple angles, follow important leads, and retrieve sufficient supporting evidence before reaching a conclusion.

The final answer MUST be comprehensive, detailed, and in-depth. It should fully address all aspects of the question, explain underlying mechanisms, compare different perspectives, and provide clear reasoning supported by evidence.

When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

You should ground every nontrivial claim in retrieved snippets. Cite using <cite id="...">...</cite> drawn only from returned snippets. Please prefer authoritative sources (peer-reviewed papers, reputable benchmarks/docs) and prioritize recent work for fast-moving areas. You should acknowledge uncertainty and conflicts; if evidence is thin or sources disagree, state it and explain what additional evidence would resolve it.

It is important to structure the answer with clear markdown headers and a coherent flow. In each section, write 5–8 sentence paragraphs with clear topic sentences and transitions.

The answer MUST:
- Be comprehensive and cover all key aspects of the question
- Provide detailed explanations rather than brief summaries
- Compare different models, assumptions, or perspectives when relevant
- Explain causal mechanisms and not just describe phenomena
- Synthesize information across sources into a coherent narrative

Use lists sparingly only when they improve clarity. Ideally, you should synthesize rather than enumerate content: it is helpful to group findings across papers, explain relationships, and build a coherent narrative that answers the question, supported by citations.

Most importantly, DO NOT invent snippets or citations and never fabricate content.

You can reason and use tools throughout the process iteratively, but all reasoning text must appear only inside a standalone <think>...</think> block, and all tool calls must appear only inside a standalone <tool_call>...</tool_call> block. Tool calls must never appear inside a <think>...</think> block. 

## Tool Usage Constraints

- The `browse` tool can ONLY be used on URLs that were returned by a previous `search` tool call.
- You MUST NOT fabricate, guess, or manually construct URLs for browsing.
- If additional webpages are needed, you MUST first use `search` to retrieve them, and then select from the returned results.
- Calling `browse` on any URL not obtained from `search` is considered invalid behavior.
- You should prioritize browsing multiple distinct URLs from different sources to ensure diversity of evidence.

## Available Tools

You can use the following tools.

{"type":"function","function":{"name":"search","description":"Perform web searches and return the top search results.","parameters":{"type":"object","properties":{"query":{"type":"array","items":{"type":"string"},"description":"A list of web search queries."}},"required":["query"]}}}

{"type":"function","function":{"name":"browse","description":"Open a specific URL and return the readable webpage content.","parameters":{"type":"object","properties":{"url":{"type":"string","description":"The URL of the webpage to browse."}},"required":["url"]}}}

{"type":"function","function":{"name":"scholar","description":"Retrieve information from scientific papers and return relevant papersnippets.","parameters":{"type":"object","properties":{"query":{"type":"array","items":{"type":"string"},"description":"A list of scholarly searchqueries."}},"required":["query"]}}}

## Tool Call Format

All tool calls must use the following format:
<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>

Examples:
<tool_call>{"name":"search","arguments":{"query":["virus quantification flow cytometry","virus counter VC3100 fluorescence triggering"]}}</tool_call>

<tool_call>{"name":"browse","arguments":{"url":"https://example.com"}}</tool_call>

<tool_call>{"name":"scholar","arguments":{"query":["virus counter VC3100 flow cytometry","fluorescence-only triggering virus quantification"]}}</tool_call>

## Tool output Format

- For search or scholar, the environment may return:
  <tool_response><snippet id=UNIQUE_ID>content</snippet>...</tool_response>
- For browse, the environment may return:
  <tool_response><webpage id=UNIQUE_ID>content</webpage></tool_response>

Support every non-trivial claim with retrieved evidence. Wrap the exact claim span in <cite id="ID1,ID2">...</cite>, where ids are snippet IDs from returned tool results (comma-separated if multiple). Use only returned snippets; never invent IDs. Avoid citing filler text; cite only the factual claim. Do not put meaningless text such as "..." inside the citation span.

## Workflow Rules

After every </tool_response>, the assistant must first output a standalone <think>...</think> block.
Then it must output either:
(1) exactly one standalone <tool_call>...</tool_call> block, if more evidence is needed, or
(2) exactly one standalone <answer></answer> block, if the evidence is sufficient.

## Final Answer Rules

- Once you collect all of the necessary information, generate a comprehensive, detailed, and in-depth final answer and mark it with <answer></answer>.
- The answer MUST be thorough, covering all relevant aspects, mechanisms, and perspectives of the question.
- Avoid short or shallow responses; prioritize depth, clarity, and completeness.
- In your answer, wrap the supported text in <cite id="SNIPPET_ID"> ... </cite>.
- You must use the exact ID from a returned snippet or webpage result.
- If multiple sources support a passage, use multiple <cite> tags around the relevant clauses or sentences.


## WORKFLOW EXAMPLE

Below is a simple example that demonstrates the process and the correct use of tools and tags. In practice, you will often need additional search iterations, and your final answer may be much longer (e.g., a multi-paragraph report).

Question: Give a concise update on 2024 renewable energy market trends and current commercial solar efficiency benchmarks. 

<think>I need to understand the current market trends first.</think>
<tool_call>{"name":"search","arguments":{"query":"2024 renewable energy market trends","topk":3}}</tool_call>

<tool_response>[<snippet id=S_a1B9xQ2>...</snippet>, <snippet id=S_p0Zr41Q>...</snippet>]</tool_response>

<think>The result is not enough. Now I need specific data on solar panel efficiency.</think>
<tool_call>{"name":"scholar","arguments":{"query":"latest solar panel efficiency 2024","topk":5}}</tool_call>

<tool_response>[<snippet id=S_x4xU7dU>...</snippet>, <snippet id=S_GxA2ZLh>...</snippet>]</tool_response>

<think>I have enough evidence to answer succinctly.</think>
<answer>Global renewables expanded rapidly in 2024, <cite id="S_p0Zr41Q,S_GxA2ZLh"> driven primarily by the growth of solar and wind energy </cite>.
<cite id="S_x4xU7dU"> State-of-the-art commercial solar modules report cell efficiencies of ~26–27% and module efficiencies of ~23–24% </cite>. Therefore, solar led 2024 renewables, and top commercial module efficiency was about 23–24%.</answer>

"""

FORMAT_REPAIR_PROMPT = (
    "Your previous message did not follow the required contract. Continue the same task now. "
    "Output all reasoning only inside one standalone <think>...</think> block, then output exactly one "
    "standalone <tool_call>...</tool_call> block or exactly one standalone <answer>...</answer> block."
)

DEFAULT_JUDGE_SYSTEM_PROMPT = """You are a strict grader for deep-research answers.

You will receive:
1. the user question
2. the scoring rubric
3. the candidate answer
4. the candidate's intermediate assistant turns
5. the retrieved tool evidence

Score conservatively. Prefer lower scores when evidence is weak, citations look suspicious, or the answer misses important parts of the question.

Return JSON only with this schema:
{
  "overall_score": 0.0,
  "passed": false,
  "dimension_scores": {
    "coverage": 0.0,
    "faithfulness": 0.0,
    "citation_quality": 0.0,
    "synthesis": 0.0,
    "clarity": 0.0
  },
  "strengths": ["..."],
  "weaknesses": ["..."],
  "reason": "short final justification"
}
"""

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
CITE_RE = re.compile(r"<cite\s+id=\"[^\"]+\">.*?</cite>", re.DOTALL)


@dataclass
class InputSample:
    sample_id: str
    question: str
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect deep-search multi-turn SFT data.")
    parser.add_argument("--input", required=True, help="Input questions file: jsonl/json/csv/txt/parquet.")
    parser.add_argument("--output-dir", required=True, help="Directory for raw logs and train/test parquet.")
    parser.add_argument("--question-key", default="question", help="Field name that stores the user question.")
    parser.add_argument("--id-key", default="id", help="Field name used as sample id when present.")
    parser.add_argument("--system-prompt-file", default=None, help="Optional file that overrides the default prompt.")
    parser.add_argument("--model", default="gpt-5.1", help="OpenAI-compatible chat model name.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible base URL.")
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI-compatible API key. Falls back to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--tool-server-url",
        default=os.getenv("DEEPSEARCH_TOOL_SERVER_URL"),
        help="Local verl tool server URL, for example http://host:port/get_observation.",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max completion tokens per assistant turn.")
    parser.add_argument(
        "--stop-sequence",
        default="",
        help="Optional stop sequence passed to the model API, for example </tool_call>.",
    )
    parser.add_argument("--max-turns", type=int, default=12, help="Maximum assistant turns per sample.")
    parser.add_argument("--max-format-repairs", type=int, default=1, help="Repair attempts for malformed output.")
    parser.add_argument("--concurrency", type=int, default=4, help="How many samples to label concurrently.")
    parser.add_argument("--request-timeout", type=int, default=300, help="Timeout seconds for model requests.")
    parser.add_argument("--tool-timeout", type=int, default=300, help="Timeout seconds for tool requests.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Take only the first N samples after resume skip.")
    parser.add_argument("--test-size", type=float, default=0.05, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for split.")
    parser.add_argument("--num-candidates", type=int, default=1, help="Generate this many candidates per question.")
    parser.add_argument("--rubric-file", default=None, help="Optional rubric text file used for candidate judging.")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model name. Defaults to --model when rubric judging is enabled.",
    )
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Sampling temperature for the judge.")
    parser.add_argument(
        "--min-judge-score",
        type=float,
        default=8.0,
        help="Minimum overall rubric score required to keep a sample when rubric judging is enabled.",
    )
    parser.add_argument(
        "--judge-max-context-chars",
        type=int,
        default=24000,
        help="Maximum chars of candidate context passed into the judge prompt.",
    )
    parser.add_argument("--require-citation", action="store_true", default=True, help="Keep only answers with <cite>.")
    parser.add_argument(
        "--no-require-citation",
        action="store_false",
        dest="require_citation",
        help="Allow final answers without <cite>.",
    )
    parser.add_argument("--require-tool-use", action="store_true", default=True, help="Keep only tool-using samples.")
    parser.add_argument(
        "--no-require-tool-use",
        action="store_false",
        dest="require_tool_use",
        help="Allow trajectories with no successful tool call.",
    )
    parser.add_argument("--resume", action="store_true", default=True, help="Skip sample ids already in raw jsonl.")
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore existing raw jsonl and relabel everything.",
    )
    parser.add_argument(
        "--allow-empty-test",
        action="store_true",
        help="Do not force at least one validation example when dataset is tiny.",
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
                continue
            if not isinstance(item, dict):
                chunks.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "text":
                chunks.append(item.get("text", ""))
            elif "text" in item:
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(content)


def load_system_prompt(path: str | None) -> str:
    if not path:
        return DEFAULT_SYSTEM_PROMPT
    return Path(path).read_text(encoding="utf-8")


def load_optional_text(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


def load_existing_ids(raw_path: Path) -> set[str]:
    if not raw_path.exists():
        return set()
    seen: set[str] = set()
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = obj.get("sample_id")
            if sample_id is not None:
                seen.add(str(sample_id))
    return seen


def load_samples(input_path: Path, question_key: str, id_key: str) -> list[InputSample]:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl_samples(input_path, question_key, id_key)
    if suffix == ".json":
        return _load_json_samples(input_path, question_key, id_key)
    if suffix == ".csv":
        return _load_csv_samples(input_path, question_key, id_key)
    if suffix == ".txt":
        return _load_txt_samples(input_path)
    if suffix == ".parquet":
        return _load_parquet_samples(input_path, question_key, id_key)
    raise ValueError(f"Unsupported input suffix: {suffix}")


def _make_sample(raw: Any, idx: int, question_key: str, id_key: str) -> InputSample:
    if isinstance(raw, str):
        question = raw.strip()
        payload = {"text": raw}
        sample_id = f"sample-{idx:06d}"
    elif isinstance(raw, dict):
        if question_key not in raw:
            raise KeyError(f"Missing question key '{question_key}' in sample {idx}")
        question = str(raw[question_key]).strip()
        payload = dict(raw)
        sample_id = str(raw.get(id_key, f"sample-{idx:06d}"))
    else:
        raise TypeError(f"Unsupported sample type: {type(raw).__name__}")

    if not question:
        raise ValueError(f"Empty question in sample {idx}")

    return InputSample(sample_id=sample_id, question=question, payload=payload)


def _load_jsonl_samples(input_path: Path, question_key: str, id_key: str) -> list[InputSample]:
    samples: list[InputSample] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            samples.append(_make_sample(json.loads(line), idx, question_key, id_key))
    return samples


def _load_json_samples(input_path: Path, question_key: str, id_key: str) -> list[InputSample]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            data = data["items"]
        else:
            raise ValueError("JSON input must be a list or a dict with an 'items' list.")
    if not isinstance(data, list):
        raise ValueError("JSON input must decode to a list.")
    return [_make_sample(item, idx, question_key, id_key) for idx, item in enumerate(data)]


def _load_csv_samples(input_path: Path, question_key: str, id_key: str) -> list[InputSample]:
    samples: list[InputSample] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            samples.append(_make_sample(dict(row), idx, question_key, id_key))
    return samples


def _load_txt_samples(input_path: Path) -> list[InputSample]:
    samples: list[InputSample] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            question = line.strip()
            if not question:
                continue
            samples.append(InputSample(sample_id=f"sample-{idx:06d}", question=question, payload={"text": question}))
    return samples


def _load_parquet_samples(input_path: Path, question_key: str, id_key: str) -> list[InputSample]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Reading parquet input requires pandas to be installed.") from exc

    dataframe = pd.read_parquet(input_path)
    samples: list[InputSample] = []
    for idx, row in enumerate(dataframe.to_dict(orient="records")):
        samples.append(_make_sample(row, idx, question_key, id_key))
    return samples


def wrap_tool_response(observation: str) -> str:
    cleaned = observation.strip()
    if cleaned.startswith("<tool_response>") and cleaned.endswith("</tool_response>"):
        return cleaned
    return f"<tool_response>\n{cleaned}\n</tool_response>"


def has_tool_call(text: str) -> bool:
    return TOOL_CALL_RE.search(text) is not None


def has_answer(text: str) -> bool:
    return ANSWER_RE.search(text) is not None


def has_citation(text: str) -> bool:
    return CITE_RE.search(text) is not None


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
        raise ValueError("No JSON object found in judge output.")
    return json.loads(stripped[first : last + 1])


def choose_split(records: list[dict[str, Any]], test_size: float, seed: int, allow_empty_test: bool) -> tuple[list, list]:
    if not records:
        return [], []
    if len(records) == 1:
        return records, records

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)

    test_count = int(round(len(records) * test_size))
    if not allow_empty_test and test_size > 0:
        test_count = max(1, test_count)
    test_count = min(max(test_count, 0), len(records) - 1)

    test_idx = set(indices[:test_count])
    train_records = [record for idx, record in enumerate(records) if idx not in test_idx]
    test_records = [record for idx, record in enumerate(records) if idx in test_idx]
    return train_records, test_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_parquet(path: Path, records: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False

    try:
        df = pd.DataFrame(records)
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def build_dataset_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "question": record["question"],
        "messages": record["messages"],
        "model": record["model"],
        "tool_turns": record["tool_turns"],
        "assistant_turns": record["assistant_turns"],
        "status": record["status"],
        "judge_score": record.get("judge_score"),
        "judge_passed": record.get("judge_passed"),
    }


def is_usable_record(record: dict[str, Any], require_citation: bool, require_tool_use: bool) -> bool:
    if record.get("status") != "completed":
        return False
    messages = record.get("messages") or []
    if not messages:
        return False
    final_text = messages[-1].get("content", "")
    if not has_answer(final_text):
        return False
    if require_citation and not has_citation(final_text):
        return False
    if require_tool_use and int(record.get("tool_turns", 0)) <= 0:
        return False
    return True


class DeepSearchSFTCollector:
    def __init__(self, args: argparse.Namespace, system_prompt: str):
        if not args.api_key:
            raise ValueError("Missing API key. Set --api-key or OPENAI_API_KEY.")
        if not args.tool_server_url:
            raise ValueError("Missing tool server URL. Set --tool-server-url or DEEPSEARCH_TOOL_SERVER_URL.")

        self.args = args
        client_kwargs: dict[str, Any] = {"api_key": args.api_key}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        self.client = AsyncOpenAI(**client_kwargs)
        self.system_prompt = system_prompt
        self.rubric_text = load_optional_text(args.rubric_file)
        self.judge_model = args.judge_model or args.model
        self.semaphore = asyncio.Semaphore(max(1, args.concurrency))
        self.stop_sequence = args.stop_sequence.strip()

    async def collect_one(self, sample: InputSample) -> dict[str, Any]:
        async with self.semaphore:
            return await self._collect_with_selection(sample)

    async def _collect_with_selection(self, sample: InputSample) -> dict[str, Any]:
        num_candidates = max(1, int(self.args.num_candidates))
        candidate_records = []
        for candidate_index in range(num_candidates):
            candidate_records.append(await self._collect_candidate(sample, candidate_index))

        completed_candidates = [record for record in candidate_records if record.get("status") == "completed"]
        if self.rubric_text and completed_candidates:
            for record in completed_candidates:
                judge_result = await self._judge_candidate(sample, record)
                record["judge"] = judge_result
                record["judge_score"] = judge_result.get("overall_score", 0.0)
                record["judge_passed"] = judge_result.get("passed", False)

            best_record = max(
                completed_candidates,
                key=lambda record: (
                    1 if record.get("judge_passed") else 0,
                    float(record.get("judge_score", 0.0)),
                    *self._fallback_sort_key(record),
                ),
            )
            selected = dict(best_record)
            if not selected.get("judge_passed", False):
                selected["status"] = "rubric_rejected"
            selected["candidate_summaries"] = [self._candidate_summary(record) for record in candidate_records]
            selected["selected_candidate_index"] = selected.get("candidate_index", 0)
            return selected

        best_record = max(candidate_records, key=self._fallback_sort_key)
        selected = dict(best_record)
        selected["candidate_summaries"] = [self._candidate_summary(record) for record in candidate_records]
        selected["selected_candidate_index"] = selected.get("candidate_index", 0)
        return selected

    async def _collect_candidate(self, sample: InputSample, candidate_index: int) -> dict[str, Any]:
        start_time = time.time()
        trajectory_id = f"{sample.sample_id}-cand{candidate_index}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": sample.question},
        ]

        tool_turns = 0
        assistant_turns = 0
        last_error = ""
        finish_reason = ""
        format_repairs = 0

        for _ in range(self.args.max_turns):
            try:
                assistant_text, finish_reason = await self._generate(messages)
            except Exception as exc:  # noqa: BLE001
                last_error = f"model_error: {exc}"
                return self._build_record(
                    sample=sample,
                    candidate_index=candidate_index,
                    trajectory_id=trajectory_id,
                    messages=messages,
                    status="model_error",
                    tool_turns=tool_turns,
                    assistant_turns=assistant_turns,
                    finish_reason=finish_reason,
                    last_error=last_error,
                    elapsed=time.time() - start_time,
                )

            assistant_turns += 1
            messages.append({"role": "assistant", "content": assistant_text})

            assistant_has_tool = has_tool_call(assistant_text)
            assistant_has_answer = has_answer(assistant_text)

            if assistant_has_answer and not assistant_has_tool:
                return self._build_record(
                    sample=sample,
                    candidate_index=candidate_index,
                    trajectory_id=trajectory_id,
                    messages=messages,
                    status="completed",
                    tool_turns=tool_turns,
                    assistant_turns=assistant_turns,
                    finish_reason=finish_reason,
                    last_error=last_error,
                    elapsed=time.time() - start_time,
                )

            if assistant_has_tool and not assistant_has_answer:
                try:
                    observation, valid = await self._run_tool(trajectory_id, assistant_text)
                except Exception as exc:  # noqa: BLE001
                    last_error = f"tool_error: {exc}"
                    return self._build_record(
                        sample=sample,
                        candidate_index=candidate_index,
                        trajectory_id=trajectory_id,
                        messages=messages,
                        status="tool_error",
                        tool_turns=tool_turns,
                        assistant_turns=assistant_turns,
                        finish_reason=finish_reason,
                        last_error=last_error,
                        elapsed=time.time() - start_time,
                    )

                if valid:
                    tool_turns += 1
                messages.append({"role": "user", "content": wrap_tool_response(observation)})
                continue

            if format_repairs < self.args.max_format_repairs:
                format_repairs += 1
                messages.append({"role": "user", "content": FORMAT_REPAIR_PROMPT})
                continue

            if assistant_has_answer and assistant_has_tool:
                last_error = "malformed_output: assistant emitted both tool_call and answer in one turn"
                status = "malformed_both"
            else:
                last_error = "malformed_output: assistant emitted neither tool_call nor answer"
                status = "malformed_none"

            return self._build_record(
                sample=sample,
                candidate_index=candidate_index,
                trajectory_id=trajectory_id,
                messages=messages,
                status=status,
                tool_turns=tool_turns,
                assistant_turns=assistant_turns,
                finish_reason=finish_reason,
                last_error=last_error,
                elapsed=time.time() - start_time,
            )

        return self._build_record(
            sample=sample,
            candidate_index=candidate_index,
            trajectory_id=trajectory_id,
            messages=messages,
            status="max_turns",
            tool_turns=tool_turns,
            assistant_turns=assistant_turns,
            finish_reason=finish_reason,
            last_error=last_error or "hit max_turns before final answer",
            elapsed=time.time() - start_time,
        )

    async def _generate(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        request_kwargs: dict[str, Any] = dict(
            model=self.args.model,
            messages=messages,
            temperature=self.args.temperature,
            max_tokens=self.args.max_tokens,
            timeout=self.args.request_timeout,
        )
        if self.stop_sequence:
            request_kwargs["stop"] = [self.stop_sequence]

        response = await self.client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        text = normalize_message_content(choice.message.content).strip()
        if self.stop_sequence == "</tool_call>" and "<tool_call>" in text and "</tool_call>" not in text:
            text = text + "</tool_call>"
        if not text:
            raise RuntimeError("empty assistant message")
        return text, finish_reason

    async def _judge_candidate(self, sample: InputSample, record: dict[str, Any]) -> dict[str, Any]:
        final_answer = normalize_message_content(record["messages"][-1]["content"]) if record.get("messages") else ""
        assistant_turns = [
            normalize_message_content(message["content"])
            for message in record.get("messages", [])
            if message.get("role") == "assistant"
        ]
        tool_responses = [
            normalize_message_content(message["content"])
            for message in record.get("messages", [])
            if message.get("role") == "user" and normalize_message_content(message.get("content", "")).startswith("<tool_response>")
        ]

        assistant_trace = "\n\n".join(
            f"Assistant turn {idx + 1}:\n{text}" for idx, text in enumerate(assistant_turns[:-1])
        )
        tool_context = "\n\n".join(tool_responses)
        assistant_trace = truncate_text(assistant_trace, self.args.judge_max_context_chars // 2)
        tool_context = truncate_text(tool_context, self.args.judge_max_context_chars)

        judge_user_prompt = (
            "Question:\n"
            f"{sample.question}\n\n"
            "Scoring rubric:\n"
            f"{self.rubric_text}\n\n"
            "Candidate final answer:\n"
            f"{final_answer}\n\n"
            "Candidate intermediate assistant trace:\n"
            f"{assistant_trace or '[NONE]'}\n\n"
            "Retrieved tool evidence:\n"
            f"{tool_context or '[NONE]'}\n\n"
            f"Set `passed=true` only if the answer meets the rubric at a high standard. Treat scores >= {self.args.min_judge_score} "
            "as the rough pass bar, but you should still judge independently. Return JSON only."
        )

        response = await self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": DEFAULT_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_user_prompt},
            ],
            temperature=self.args.judge_temperature,
            max_tokens=1200,
            timeout=self.args.request_timeout,
        )

        raw_text = normalize_message_content(response.choices[0].message.content).strip()
        try:
            judge_obj = extract_json_object(raw_text)
        except Exception as exc:  # noqa: BLE001
            return {
                "overall_score": 0.0,
                "passed": False,
                "reason": f"judge_parse_error: {exc}",
                "raw_text": raw_text,
            }

        score_raw = judge_obj.get("overall_score", judge_obj.get("score", 0.0))
        try:
            score = float(score_raw)
        except Exception:
            score = 0.0
        score = max(0.0, min(10.0, score))

        passed_raw = judge_obj.get("passed", None)
        if isinstance(passed_raw, bool):
            passed = passed_raw
        else:
            verdict = str(judge_obj.get("verdict", "")).strip().lower()
            if verdict in {"pass", "passed", "true"}:
                passed = True
            elif verdict in {"fail", "failed", "false"}:
                passed = False
            else:
                passed = score >= float(self.args.min_judge_score)

        return {
            "overall_score": score,
            "passed": passed and score >= float(self.args.min_judge_score),
            "dimension_scores": judge_obj.get("dimension_scores", {}),
            "strengths": judge_obj.get("strengths", []),
            "weaknesses": judge_obj.get("weaknesses", []),
            "reason": judge_obj.get("reason", ""),
            "raw_text": raw_text,
        }

    async def _run_tool(self, trajectory_id: str, action: str) -> tuple[str, bool]:
        payload = {
            "trajectory_ids": [trajectory_id],
            "actions": [action],
            "extra_fields": [{}],
        }
        response = await asyncio.to_thread(
            requests.post,
            self.args.tool_server_url,
            json=payload,
            timeout=self.args.tool_timeout,
        )
        response.raise_for_status()
        data = response.json()
        observations = data.get("observations") or []
        valids = data.get("valids") or []
        if not observations:
            raise RuntimeError(f"tool server returned no observations: {data}")
        observation = observations[0]
        valid = bool(valids[0]) if valids else True
        if not isinstance(observation, str):
            observation = json.dumps(observation, ensure_ascii=False)
        return observation, valid

    def _build_record(
        self,
        *,
        sample: InputSample,
        candidate_index: int,
        trajectory_id: str,
        messages: list[dict[str, str]],
        status: str,
        tool_turns: int,
        assistant_turns: int,
        finish_reason: str,
        last_error: str,
        elapsed: float,
    ) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "candidate_index": candidate_index,
            "trajectory_id": trajectory_id,
            "messages": messages,
            "status": status,
            "tool_turns": tool_turns,
            "assistant_turns": assistant_turns,
            "finish_reason": finish_reason,
            "last_error": last_error,
            "model": self.args.model,
            "judge_model": self.judge_model if self.rubric_text else "",
            "rubric_enabled": bool(self.rubric_text),
            "base_url": self.args.base_url or "",
            "tool_server_url": self.args.tool_server_url,
            "elapsed_seconds": round(elapsed, 3),
            "source_payload": sample.payload,
        }

    @staticmethod
    def _candidate_summary(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_index": record.get("candidate_index"),
            "status": record.get("status"),
            "tool_turns": record.get("tool_turns"),
            "assistant_turns": record.get("assistant_turns"),
            "elapsed_seconds": record.get("elapsed_seconds"),
            "judge_score": record.get("judge_score"),
            "judge_passed": record.get("judge_passed"),
            "last_error": record.get("last_error"),
        }

    @staticmethod
    def _fallback_sort_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
        final_text = ""
        messages = record.get("messages") or []
        if messages:
            final_text = normalize_message_content(messages[-1].get("content", ""))
        return (
            1 if record.get("status") == "completed" else 0,
            1 if has_citation(final_text) else 0,
            int(record.get("tool_turns", 0)),
            -int(record.get("assistant_turns", 0)),
        )


async def collect_all(args: argparse.Namespace, samples: list[InputSample], raw_path: Path, system_prompt: str) -> None:
    collector = DeepSearchSFTCollector(args, system_prompt)
    write_lock = asyncio.Lock()

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8") as handle:
        tasks = [asyncio.create_task(collector.collect_one(sample)) for sample in samples]
        for task in asyncio.as_completed(tasks):
            record = await task
            async with write_lock:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
            judge_score = record.get("judge_score")
            judge_part = f" judge_score={judge_score}" if judge_score is not None else ""
            print(
                f"[{record['status']}] {record['sample_id']} "
                f"tool_turns={record['tool_turns']} assistant_turns={record['assistant_turns']}{judge_part}",
                flush=True,
            )


def load_raw_records(raw_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not raw_path.exists():
        return records
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def export_dataset(args: argparse.Namespace, output_dir: Path, raw_records: list[dict[str, Any]], system_prompt: str) -> None:
    usable_records = [
        record
        for record in raw_records
        if is_usable_record(record, args.require_citation, args.require_tool_use)
    ]
    dataset_rows = [build_dataset_row(record) for record in usable_records]
    train_rows, test_rows = choose_split(
        dataset_rows,
        test_size=args.test_size,
        seed=args.seed,
        allow_empty_test=args.allow_empty_test,
    )

    write_jsonl(output_dir / "usable_records.jsonl", usable_records)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    wrote_train = write_parquet(output_dir / "train.parquet", train_rows)
    wrote_test = write_parquet(output_dir / "test.parquet", test_rows)

    summary = {
        "total_raw_records": len(raw_records),
        "completed_records": sum(record.get("status") == "completed" for record in raw_records),
        "usable_records": len(usable_records),
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "require_citation": args.require_citation,
        "require_tool_use": args.require_tool_use,
        "model": args.model,
        "judge_model": args.judge_model or args.model if args.rubric_file else "",
        "num_candidates": args.num_candidates,
        "min_judge_score": args.min_judge_score if args.rubric_file else None,
        "rubric_file": args.rubric_file or "",
        "base_url": args.base_url or "",
        "tool_server_url": args.tool_server_url,
        "system_prompt": system_prompt,
    }
    (output_dir / "meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not (wrote_train and wrote_test):
        print(
            "Warning: parquet export was skipped because pandas/pyarrow is not available in this runtime. "
            "JSONL files were still written.",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt(args.system_prompt_file)
    raw_path = output_dir / "raw_records.jsonl"

    samples = load_samples(input_path, args.question_key, args.id_key)
    if args.resume:
        seen = load_existing_ids(raw_path)
        samples = [sample for sample in samples if sample.sample_id not in seen]
        print(f"Resume enabled. Skipping {len(seen)} existing sample ids.", flush=True)

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    print(f"Loaded {len(samples)} samples to process.", flush=True)

    if samples:
        asyncio.run(collect_all(args, samples, raw_path, system_prompt))

    raw_records = load_raw_records(raw_path)
    export_dataset(args, output_dir, raw_records, system_prompt)


if __name__ == "__main__":
    main()
