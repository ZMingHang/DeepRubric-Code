#!/usr/bin/env python3
"""
LLM-based quality filter for outputs produced by recursive_qa_agent_v3.py.

Input:
- A single .jsonl/.json file, or
- A directory containing .jsonl/.json files.

For each sample, an LLM judge returns:
- decision (KEEP / REVISE / DROP)
- major issues and concrete repair suggestions
- optional repaired question/rubrics

Outputs:
- Per-sample JSONL (original record fields + `_audit` block)
- Aggregate summary JSON
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


JUDGE_SYSTEM_PROMPT = """
You are an expert auditor for tree-rubric data used in long-form answer evaluation and deep research QA.
You must audit one JSON sample.

Goal:
- Decide if the sample should be KEEP / REVISE / DROP.
- If REVISE, provide a repaired question and/or repaired rubrics.
- Keep the original topic unless it is structurally unsalvageable.

Deep research standards:
- Requires synthesis across multiple evidence sources.
- Requires multi-step reasoning, not shallow fact lookup.
- Connects mechanisms, evidence, limitations, uncertainty, and conclusions.
- Rewards structured, evidence-grounded comparative reasoning.

Audit dimensions (apply all, do not skip):
1) Question-rubric alignment
- Rubrics must stay centered on the main question.
- Rubric granularity should match the question.
- Do not elevate a narrow technical detail into a mandatory rubric unless it is truly central.

2) Deep research suitability
- Rubrics should evaluate deep reasoning, not just surface coverage.
- They should support comparison, synthesis, causal or mechanistic explanation, evidence qualification, and structured conclusions when appropriate.
- They should not reduce a deep research question into a checklist of isolated facts.
- They should reward answers that integrate multiple dimensions of the problem.

3) Rubric quality
- Each rubric should be atomic, specific, and independently judgeable.
- Rubrics should not substantially overlap or score the same thing twice.
- Avoid vague or non-judgeable wording.

4) Evidence sufficiency
- source_leaf_ids must sufficiently support each rubric.
- A detail appearing in one leaf does not automatically justify making it a required rubric.
- Flag claims that are stronger than cited support.
- Note: Leaf nodes may be truncated due to length constraints. If you do not see the full leaf node content during your audit, you may temporarily ignore this issue and judge solely based on the visible information.

5) Merge faithfulness
- The final question should reflect the semantic union of selected leaves and intermediate merges when trace exists.
- Flag cases where the final question becomes broader, stronger, or more specific than justified.

6) Scoring validity
- Weights should be reasonable.
- Core factual rubrics should not be dominated by vague logical rubrics.
- Negative-weight rubrics should usually be treated as problematic unless explicitly justified.

A good sample is:
- centered on the question
- aligned in granularity
- suitable for deep research evaluation
- atomic and judgeable
- non-redundant
- evidence-grounded
- faithful to the available evidence/merge context
- useful for distinguishing strong research-grade answers from weak or shallow ones

Decision policy:
- KEEP: usable as-is
- REVISE: valuable but needs targeted repair
- DROP: severe structural or semantic failure
Prefer REVISE over DROP when topic remains valuable.
Do not invent unsupported claims.

If decision = REVISE, follow this hard rewrite template:
- revised_question:
  - Keep the topic, but fix overreach/under-specification.
  - Must stay faithful to selected leaves + merge trace when available; otherwise stay faithful to tree_repr/search_tree_text + statements.
  - Must remain a deep research question.
- revised_rubrics:
  - Output 5 to 8 rubric objects.
  - Every rubric MUST contain: id (e.g., R1, R2), type, description, weight, source_leaf_ids.
  - type must be logical or factual.
  - source_leaf_ids must be non-empty.
  - logical rubrics do NOT need an evidence field.
  - factual rubrics MUST include a non-empty evidence list grounded in selected-leaf statements.
  - Here, evidence means supporting statement text from selected leaves (verbatim or tightly faithful short paraphrase), not generic reasoning notes.
  - If selected_leaf_ids are provided, source_leaf_ids should be a subset of selected_leaf_ids.
  - If selected_leaf_ids are missing, source_leaf_ids should refer to provided evidence item ids (e.g., E1, E2).
  - Avoid semantic duplicates and overlap.
  - For factual rubrics, include concise evidence grounded in leaf statements (i.e., statement text); do not add evidence to logical rubrics unless strictly necessary.
  - Use tree information as much as possible: selected leaves, leaf paths, merge trace, tree_repr/search_tree_text, and statements.
  - Prefer rubrics that evaluate synthesis, evidence-based comparison, causal/mechanistic reasoning, limitations, and conclusion quality.


Input may be partial by design. If a field is missing, note uncertainty in major_issues, but still provide the best audit possible.

Return VALID JSON ONLY with this exact top-level shape:
{
  "decision": "KEEP | REVISE | DROP",
  "error_tags": [],
  "major_issues": [
    {
      "type": "",
      "severity": "low|medium|high",
      "description": ""
    }
  ],
  "justification": "",
  "fix_plan": [],
  "revised_question": null,
  "revised_rubrics": null,
  "implementation_guidance": {
    "rule_checks": [],
    "llm_checks": [],
    "revision_logic": []
  }
}

Rules:
- If decision is KEEP, revised_question and revised_rubrics should usually be null.
- If decision is REVISE, provide at least one concrete fix in fix_plan and revision_logic.
- revised_rubrics must be judgeable, non-redundant, and evidence-grounded.
- Output JSON only; no markdown, no prose outside JSON.
""".strip()


def _is_nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _clip_text(text: Any, max_chars: int) -> str:
    s = str(text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + " ..."


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    first_obj = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if first_obj:
        candidates.append(first_obj.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _clip_list(items: Any, max_items: int = 12, max_chars: int = 220) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        if not _is_nonempty_str(item):
            continue
        out.append(_clip_text(item, max_chars))
        if len(out) >= max_items:
            break
    return out


def _extract_tree_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    tree_json = record.get("tree_json")
    if isinstance(tree_json, str) and tree_json.strip():
        try:
            parsed = json.loads(tree_json)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed.get("tree_payload")
            if isinstance(payload, dict):
                return payload
    elif isinstance(tree_json, dict):
        payload = tree_json.get("tree_payload")
        if isinstance(payload, dict):
            return payload

    payload = record.get("tree_payload")
    if isinstance(payload, dict):
        return payload
    return {}


def _extract_tree_json_blob(record: Dict[str, Any]) -> Dict[str, Any]:
    tree_json = record.get("tree_json")
    if isinstance(tree_json, dict):
        return tree_json
    if isinstance(tree_json, str) and tree_json.strip():
        try:
            parsed = json.loads(tree_json)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return {}


def _build_evidence_items(
    statements: List[str],
    max_items: int,
    max_chars: int,
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for idx, s in enumerate(statements[:max_items], start=1):
        if not _is_nonempty_str(s):
            continue
        items.append({"id": f"E{idx}", "text": _clip_text(s, max_chars)})
    return items


PAYLOAD_FIELDS_ALL = {
    "uid",
    "root_query",
    "generated_question",
    "generated_rubrics",
    "query",
    "question",
    "tree_depth",
    "trace_mode",
    "data_availability",
    "selected_leaf_ids",
    "selection_trace",
    "merge_trace",
    "statements_flat",
    "evidence_items",
    "rubrics",
    "selected_leaf_evidence",
    "tree_repr",
    "tree_repr_with_reason",
    "search_tree_text",
    "generation_input_content",
    "tree_meta",
}

PAYLOAD_FIELDS_BY_MODE = {
    "minimal": {
        "generated_question",
        "generated_rubrics",
        "question",
        "rubrics",
        "statements_flat",
        "tree_repr_with_reason",
        "generation_input_content",
        "trace_mode",
        "data_availability",
    },
    "generator_like": {
        "uid",
        "root_query",
        "generated_question",
        "generated_rubrics",
        "question",
        "rubrics",
        "statements_flat",
        "tree_repr",
        "tree_repr_with_reason",
        "search_tree_text",
        "generation_input_content",
        "tree_depth",
        "trace_mode",
        "data_availability",
    },
    "balanced": {
        "uid",
        "root_query",
        "generated_question",
        "generated_rubrics",
        "query",
        "question",
        "tree_depth",
        "trace_mode",
        "data_availability",
        "selected_leaf_ids",
        "selection_trace",
        "merge_trace",
        "statements_flat",
        "evidence_items",
        "rubrics",
        "selected_leaf_evidence",
        "generation_input_content",
        "tree_repr_with_reason",
        "search_tree_text",
        "tree_meta",
    },
    "full": PAYLOAD_FIELDS_ALL,
}


def _parse_csv_set(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def _resolve_payload_fields(mode: str, include_fields: Optional[str], exclude_fields: Optional[str]) -> set[str]:
    base = set(PAYLOAD_FIELDS_BY_MODE.get(mode, PAYLOAD_FIELDS_BY_MODE["balanced"]))
    include_set = _parse_csv_set(include_fields)
    explicit_include = include_set is not None and len(include_set) > 0
    if include_set is not None and len(include_set) > 0:
        # explicit include acts as whitelist
        base = {x for x in include_set if x in PAYLOAD_FIELDS_ALL}
    exclude_set = _parse_csv_set(exclude_fields)
    if exclude_set:
        base -= exclude_set
    # For mode-driven defaults, keep a minimal meaningful core.
    # For explicit include, respect user selection as-is.
    if not explicit_include:
        essentials = {"question", "rubrics", "statements_flat", "tree_repr_with_reason", "trace_mode"}
        base |= essentials
    return {x for x in base if x in PAYLOAD_FIELDS_ALL}


def _prepare_judge_payload(
    record: Dict[str, Any],
    payload_fields: set[str],
    max_leaves: int,
    max_statements_per_leaf: int,
    max_statement_chars: int,
    max_rubric_evidence_chars: int,
    max_flat_statements: int,
    max_tree_repr_chars: int,
) -> Dict[str, Any]:
    tree_payload = _extract_tree_payload(record)
    tree_blob = _extract_tree_json_blob(record)
    selected_leaf_ids = record.get("selected_leaf_ids", [])
    selected_leaf_ids_set = set(selected_leaf_ids) if isinstance(selected_leaf_ids, list) else set()

    raw_leaves = tree_payload.get("leaves", []) if isinstance(tree_payload, dict) else []
    leaf_objects = []
    leaf_limit = max(max_leaves, len(selected_leaf_ids_set)) if selected_leaf_ids_set else max_leaves

    for leaf in raw_leaves:
        if not isinstance(leaf, dict):
            continue
        leaf_id = leaf.get("id")
        if not _is_nonempty_str(leaf_id):
            continue
        if selected_leaf_ids_set and leaf_id not in selected_leaf_ids_set:
            continue

        statements = leaf.get("statements", [])
        if not isinstance(statements, list):
            statements = []
        clipped_statements = [
            _clip_text(s, max_statement_chars)
            for s in statements[:max_statements_per_leaf]
            if _is_nonempty_str(s)
        ]
        leaf_objects.append(
            {
                "id": leaf_id,
                "path": leaf.get("path", []),
                "statements": clipped_statements,
            }
        )
        # if len(leaf_objects) >= max_leaves:
        if len(leaf_objects) >= leaf_limit:

            break

    if not leaf_objects:
        for leaf in raw_leaves[:max_leaves]:
            if not isinstance(leaf, dict):
                continue
            statements = leaf.get("statements", [])
            if not isinstance(statements, list):
                statements = []
            leaf_objects.append(
                {
                    "id": leaf.get("id"),
                    "path": leaf.get("path", []),
                    "statements": [
                        _clip_text(s, max_statement_chars)
                        for s in statements[:max_statements_per_leaf]
                        if _is_nonempty_str(s)
                    ],
                }
            )

    rubrics = record.get("rubrics", [])
    compact_rubrics = []
    if isinstance(rubrics, list):
        for r in rubrics:
            if not isinstance(r, dict):
                compact_rubrics.append(r)
                continue
            ev = r.get("evidence", [])
            if isinstance(ev, list):
                ev = [_clip_text(x, max_rubric_evidence_chars) for x in ev[:4] if _is_nonempty_str(x)]
            else:
                ev = []
            compact_rubrics.append(
                {
                    "id": r.get("id"),
                    "type": r.get("type"),
                    "description": _clip_text(r.get("description", ""), 350),
                    "weight": r.get("weight"),
                    "source_leaf_ids": r.get("source_leaf_ids", []),
                    "evidence": ev,
                }
            )

    flat_statements: List[str] = []
    raw_record_statements = record.get("statements", [])
    if isinstance(raw_record_statements, list):
        flat_statements.extend([s for s in raw_record_statements if _is_nonempty_str(s)])
    raw_tree_statements = tree_blob.get("statements", [])
    if isinstance(raw_tree_statements, list):
        for s in raw_tree_statements:
            if _is_nonempty_str(s):
                flat_statements.append(s)
    # de-dup preserve order
    seen_stmt = set()
    dedup_flat_statements = []
    for s in flat_statements:
        s_norm = s.strip()
        if s_norm not in seen_stmt:
            dedup_flat_statements.append(s_norm)
            seen_stmt.add(s_norm)
    dedup_flat_statements = dedup_flat_statements[:max_flat_statements]

    tree_repr = _clip_text(record.get("tree_repr", ""), max_tree_repr_chars)
    tree_repr_with_reason = _clip_text(record.get("tree_repr_with_reason", ""), max_tree_repr_chars)
    search_tree_text = _clip_text(tree_blob.get("search_tree_text", ""), max_tree_repr_chars)

    has_selected_leaf_ids = isinstance(selected_leaf_ids, list) and len(selected_leaf_ids) > 0
    has_selection_trace = isinstance(record.get("selection_trace"), list) and len(record.get("selection_trace", [])) > 0
    has_merge_trace = isinstance(record.get("merge_trace"), list) and len(record.get("merge_trace", [])) > 0
    trace_mode = "trace_rich" if (has_selected_leaf_ids and (has_selection_trace or has_merge_trace)) else "trace_light"

    if isinstance(tree_payload, dict) and tree_payload:
        # Put selected leaves first so long-tree truncation does not remove the
        # exact nodes used by the generated question/rubrics.
        selected_first_payload = {
            "root_query": tree_payload.get("root_query", record.get("root_query")),
            "root_id": tree_payload.get("root_id"),
            "selected_leaf_ids": selected_leaf_ids if isinstance(selected_leaf_ids, list) else [],
            "selected_leaf_context": leaf_objects,
            "selection_trace": record.get("selection_trace", []),
            "merge_trace": record.get("merge_trace", []),
            "tree_meta": {
                "tree_depth": record.get("tree_depth"),
                "leaf_count_total": len(raw_leaves) if isinstance(raw_leaves, list) else 0,
                "selected_leaf_count_in_context": len(leaf_objects),
            },
            "full_tree_payload": tree_payload,
        }
        generation_input_content = _clip_text(
            json.dumps(selected_first_payload, ensure_ascii=False),
            max_tree_repr_chars * 2,
        )
    else:
        # Fallback for legacy samples without tree_payload.
        generation_input_content = _clip_text(
            "# Search Tree (structure)\n"
            + tree_repr
            + "\n\n# Evidence (flat statements)\n"
            + (
                "\n".join([_clip_text(s, max_statement_chars) for s in dedup_flat_statements])
                if dedup_flat_statements
                else _clip_text(record.get("root_query", ""), 600)
            ),
            max_tree_repr_chars * 2,
        )

    full_payload = {
        "uid": record.get("uid"),
        "root_query": record.get("root_query"),
        "generated_question": _clip_text(record.get("question", ""), 600),
        "generated_rubrics": compact_rubrics,
        "query": _clip_text(record.get("question", record.get("root_query", "")), 600),
        "question": _clip_text(record.get("question", ""), 600),
        "tree_depth": record.get("tree_depth"),
        "trace_mode": trace_mode,
        "data_availability": {
            "has_selected_leaf_ids": has_selected_leaf_ids,
            "has_selection_trace": has_selection_trace,
            "has_merge_trace": has_merge_trace,
            "has_tree_repr": _is_nonempty_str(tree_repr),
            "has_tree_repr_with_reason": _is_nonempty_str(tree_repr_with_reason),
            "has_search_tree_text": _is_nonempty_str(search_tree_text),
            "has_flat_statements": len(dedup_flat_statements) > 0,
            "has_tree_payload_leaves": len(raw_leaves) > 0,
        },
        "selected_leaf_ids": selected_leaf_ids if isinstance(selected_leaf_ids, list) else [],
        "selection_trace": record.get("selection_trace", []),
        "merge_trace": record.get("merge_trace", []),
        "statements_flat": [_clip_text(s, max_statement_chars) for s in dedup_flat_statements],
        "evidence_items": _build_evidence_items(dedup_flat_statements, max_flat_statements, max_statement_chars),
        "rubrics": compact_rubrics,
        "selected_leaf_evidence": leaf_objects,
        "tree_repr": tree_repr,
        "tree_repr_with_reason": tree_repr_with_reason,
        "search_tree_text": search_tree_text,
        # Mirror recursive_qa_agent_v3.generate_single input to BASE_QA_TREE_PROMPT:
        # enriched_content = json.dumps(tree_payload, ensure_ascii=False)
        "generation_input_content": generation_input_content,
        "tree_meta": {
            "tree_depth": record.get("tree_depth"),
            "leaf_count_total": len(raw_leaves) if isinstance(raw_leaves, list) else 0,
        },
    }
    return {k: v for k, v in full_payload.items() if k in payload_fields}


class LLMJudge:
    def __init__(self, model: str, base_url: Optional[str], api_key: Optional[str], timeout_s: float) -> None:
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8008/v1")
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        self.model = model
        self.timeout_s = timeout_s

    def evaluate(self, judge_payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        user_prompt = "Sample to audit (JSON):\n" + json.dumps(judge_payload, ensure_ascii=False, indent=2)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            timeout=self.timeout_s,
        )
        text = resp.choices[0].message.content or ""
        return _extract_json_obj(text), text


def _normalize_revised_rubrics(raw: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(raw, list):
        return None
    cleaned: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rid = item.get("id")
        rtype = item.get("type")
        desc = item.get("description")
        weight = item.get("weight")
        source_leaf_ids = item.get("source_leaf_ids", [])
        evidence = item.get("evidence", [])

        row: Dict[str, Any] = {
            "id": _clip_text(rid, 24) if rid is not None else None,
            "type": _clip_text(rtype, 24) if rtype is not None else None,
            "description": _clip_text(desc, 360) if desc is not None else "",
            "weight": weight,
            "source_leaf_ids": source_leaf_ids if isinstance(source_leaf_ids, list) else [],
        }
        if isinstance(evidence, list):
            row["evidence"] = [_clip_text(x, 280) for x in evidence[:6] if _is_nonempty_str(x)]
        cleaned.append(row)
        if len(cleaned) >= 20:
            break

    return cleaned or None


def _severity_to_issue_level(sev: str) -> str:
    s = (sev or "").strip().lower()
    if s in {"high", "error", "critical"}:
        return "error"
    if s in {"medium", "warning", "warn"}:
        return "warning"
    return "info"


def _normalize_judge_output(
    parsed: Optional[Dict[str, Any]],
    selected_leaf_ids: Optional[List[str]] = None,
    allowed_source_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    
    if not isinstance(parsed, dict):
        print(parsed)
        return {
            "decision": "DROP",
            "error_tags": ["judge_parse_error"],
            "major_issues": [
                {
                    "type": "judge_output",
                    "severity": "high",
                    "description": "Failed to parse judge JSON output.",
                }
            ],
            "justification": "Judge response was not parseable JSON.",
            "fix_plan": ["Retry audit with stricter formatting and shorter payload."],
            "revised_question": None,
            "revised_rubrics": None,
            "implementation_guidance": {
                "rule_checks": [],
                "llm_checks": ["Validate that judge returns strict JSON before parsing."],
                "revision_logic": ["Retry with reduced leaf/rubric context."],
            },
            "issues": [
                {
                    "severity": "error",
                    "code": "judge_parse_error",
                    "message": "Failed to parse judge JSON output",
                    "suggestion": "Retry with stricter response formatting or shorter input payload.",
                }
            ],
            "improvement_actions": ["Retry audit with a shorter payload."],
            "rationale": "Judge response was not parseable JSON.",
        }

    decision = str(parsed.get("decision", "")).strip().upper()
    if decision not in {"KEEP", "REVISE", "DROP"}:
        decision = ""

    if decision == "":
        # Fallback only if model fails to return a decision.
        if parsed.get("revised_question") is not None or parsed.get("revised_rubrics") is not None:
            decision = "REVISE"
        else:
            decision = "DROP"

    error_tags = []
    raw_error_tags = parsed.get("error_tags", [])
    if isinstance(raw_error_tags, list):
        error_tags = [_clip_text(x, 64) for x in raw_error_tags if _is_nonempty_str(x)][:20]

    major_issues_raw = parsed.get("major_issues", [])
    if not isinstance(major_issues_raw, list):
        major_issues_raw = []
    major_issues: List[Dict[str, Any]] = []
    for issue in major_issues_raw:
        if not isinstance(issue, dict):
            continue
        issue_type = _clip_text(issue.get("type", "general_issue"), 80)
        severity = str(issue.get("severity", "medium")).lower().strip()
        if severity not in {"low", "medium", "high"}:
            severity = "medium"
        description = _clip_text(issue.get("description", ""), 360)
        major_issues.append(
            {
                "type": issue_type or "general_issue",
                "severity": severity,
                "description": description,
            }
        )
        if len(major_issues) >= 20:
            break

    justification = _clip_text(parsed.get("justification", parsed.get("rationale", "")), 700)
    fix_plan = _clip_list(parsed.get("fix_plan", parsed.get("improvement_actions", [])), max_items=12, max_chars=240)

    revised_question = parsed.get("revised_question")
    if not _is_nonempty_str(revised_question):
        revised_question = None
    else:
        revised_question = _clip_text(revised_question, 700)

    revised_rubrics = _normalize_revised_rubrics(parsed.get("revised_rubrics"))
    selected_leaf_id_set = set(selected_leaf_ids or [])
    allowed_source_id_set = set(allowed_source_ids or [])
    if not allowed_source_id_set and selected_leaf_id_set:
        allowed_source_id_set = set(selected_leaf_id_set)

    ig_raw = parsed.get("implementation_guidance", {})
    if not isinstance(ig_raw, dict):
        ig_raw = {}
    implementation_guidance = {
        "rule_checks": _clip_list(ig_raw.get("rule_checks", []), max_items=20, max_chars=240),
        "llm_checks": _clip_list(ig_raw.get("llm_checks", []), max_items=20, max_chars=240),
        "revision_logic": _clip_list(ig_raw.get("revision_logic", []), max_items=20, max_chars=240),
    }

    normalized_issues = []
    for tag in error_tags:
        normalized_issues.append(
            {
                "severity": "warning",
                "code": tag,
                "message": f"Error tag: {tag}",
                "suggestion": "Check corresponding fix_plan and implementation_guidance actions.",
            }
        )
    for i, issue in enumerate(major_issues):
        sev = _severity_to_issue_level(issue["severity"])
        code = issue["type"] if issue["type"] else f"major_issue_{i+1}"
        suggestion = fix_plan[i] if i < len(fix_plan) else "Apply targeted revision from revision_logic."
        normalized_issues.append(
            {
                "severity": sev,
                "code": _clip_text(code, 80),
                "message": _clip_text(issue["description"], 300),
                "suggestion": _clip_text(suggestion, 300),
            }
        )
    if not normalized_issues and decision != "KEEP":
        normalized_issues.append(
            {
                "severity": "warning",
                "code": "needs_revision",
                "message": f"Decision is {decision} but no explicit issues returned.",
                "suggestion": "Review justification and add explicit actionable issues.",
            }
        )

    if decision == "REVISE":
        if revised_rubrics is None or len(revised_rubrics) == 0:
            normalized_issues.append(
                {
                    "severity": "error",
                    "code": "revise_missing_rubrics",
                    "message": "Decision is REVISE but revised_rubrics is empty.",
                    "suggestion": "Provide 5-8 revised rubrics with source_leaf_ids.",
                }
            )
        else:
            if len(revised_rubrics) < 5 or len(revised_rubrics) > 8:
                normalized_issues.append(
                    {
                        "severity": "warning",
                        "code": "revise_rubric_count_out_of_range",
                        "message": f"revised_rubrics count is {len(revised_rubrics)} (expected 5-8).",
                        "suggestion": "Rewrite to 5-8 non-overlapping rubrics.",
                    }
                )
            seen_desc = set()
            for idx, rr in enumerate(revised_rubrics, start=1):
                desc_norm = _clip_text((rr.get("description") or "").strip().lower(), 260)
                if desc_norm and desc_norm in seen_desc:
                    normalized_issues.append(
                        {
                            "severity": "warning",
                            "code": "revise_duplicate_rubrics",
                            "message": f"Revised rubric {idx} appears semantically duplicated.",
                            "suggestion": "Merge or rewrite overlapping rubrics.",
                        }
                    )
                    break
                if desc_norm:
                    seen_desc.add(desc_norm)

                src = rr.get("source_leaf_ids", [])
                if not isinstance(src, list) or len(src) == 0:
                    normalized_issues.append(
                        {
                            "severity": "error",
                            "code": "revise_missing_source_leaf_ids",
                            "message": f"Revised rubric {idx} has empty source_leaf_ids.",
                            "suggestion": "Attach non-empty source_leaf_ids to each revised rubric.",
                        }
                    )
                    continue

                rtype = str(rr.get("type", "")).strip().lower()
                evidence = rr.get("evidence", [])
                has_evidence = isinstance(evidence, list) and any(_is_nonempty_str(x) for x in evidence)
                if rtype == "factual" and not has_evidence:
                    normalized_issues.append(
                        {
                            "severity": "error",
                            "code": "revise_factual_missing_evidence",
                            "message": f"Revised factual rubric {idx} is missing evidence.",
                            "suggestion": "Add a non-empty evidence list for each factual rubric.",
                        }
                    )
                if allowed_source_id_set:
                    extra = [x for x in src if x not in allowed_source_id_set]
                    if extra:
                        normalized_issues.append(
                            {
                                "severity": "warning",
                                "code": "revise_source_leaf_out_of_selected",
                                "message": f"Revised rubric {idx} includes source_leaf_ids outside allowed evidence anchors: {extra}",
                                "suggestion": "Align revised source_leaf_ids with selected_leaf_ids or provided evidence item ids.",
                            }
                        )
                        break

    return {
        "decision": decision,
        "error_tags": error_tags,
        "major_issues": major_issues,
        "justification": justification,
        "fix_plan": fix_plan,
        "revised_question": revised_question,
        "revised_rubrics": revised_rubrics,
        "implementation_guidance": implementation_guidance,
        "issues": normalized_issues,
        "improvement_actions": fix_plan,
        "rationale": justification,
    }


def _iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*.jsonl")):
        if p.is_file():
            yield p
    for p in sorted(path.rglob("*.json")):
        if p.is_file():
            yield p


def _read_records(path: Path) -> Iterable[Tuple[Dict[str, Any], Path, int]]:
    if path.suffix.lower() == ".jsonl":
        # Standard JSONL reader: one JSON object per line.
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj, path, i
        return

    if path.suffix.lower() == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            yield {"_parse_error": True, "_raw_line": f"<json parse failed: {e}>"}, path, 1
            return
        if isinstance(obj, list):
            for i, item in enumerate(obj, start=1):
                if isinstance(item, dict):
                    yield item, path, i
                else:
                    yield {"_parse_error": True, "_raw_line": str(item)}, path, i
        elif isinstance(obj, dict):
            yield obj, path, 1
        else:
            yield {"_parse_error": True, "_raw_line": str(obj)}, path, 1


def _build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "decision_counts": {},
            "top_issue_codes": [],
        }

    def _audit_view(r: Dict[str, Any]) -> Dict[str, Any]:
        audit = r.get("_audit")
        if isinstance(audit, dict):
            return audit
        return r

    passed = sum(1 for r in results if _audit_view(r).get("pass"))

    issue_counter = Counter()
    decision_counter = Counter()
    for r in results:
        rv = _audit_view(r)
        decision_counter[str(rv.get("decision", "UNKNOWN"))] += 1
        for issue in rv.get("issues", []):
            issue_counter[issue.get("code", "unknown")] += 1

    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "decision_counts": dict(decision_counter),
        "top_issue_codes": issue_counter.most_common(20),
    }


def _to_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except Exception:
        return path.name


def _make_cache_key(src: Path, line_no: int, input_root: Path) -> str:
    rel = _to_relpath(src, input_root)
    return f"{rel}#{line_no}"


def _load_processed_keys_from_cache(cache_path: Path) -> set[str]:
    keys: set[str] = set()
    if not cache_path.exists():
        return keys
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                k = obj.get("key")
                if _is_nonempty_str(k):
                    keys.add(k)
    except Exception:
        return keys
    return keys


def _load_processed_keys_from_outputs(output_files: List[Path]) -> set[str]:
    keys: set[str] = set()
    for p in output_files:
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    k = obj.get("_cache_key")
                    if _is_nonempty_str(k):
                        keys.add(k)
        except Exception:
            continue
    return keys


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM quality filter for recursive_qa_agent_v3 outputs.")
    parser.add_argument("--input", required=True, help="Input file or directory (.jsonl/.json).")
    parser.add_argument("--output", default=None, help="Output JSONL path for single-file mode.")
    parser.add_argument("--output-dir", default=None, help="Output directory for folder mode (mirrors input filenames).")
    parser.add_argument("--summary-output", default=None, help="Output summary JSON path.")
    parser.add_argument("--cache-path", default=None, help="Cache checkpoint JSONL path for resume.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel judge workers.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar display.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Only process first N unprocessed samples (global across input files).",
    )
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from cache/output.")
    parser.add_argument("--cache-flush-every", type=int, default=1, help="Flush cache every N written records.")
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Allow pass even if issue severity includes error.",
    )
    parser.add_argument(
        "--allow-revise-pass",
        action="store_true",
        help="Allow REVISE samples to pass threshold check (default requires KEEP).",
    )
    parser.add_argument("--judge-model", default=os.getenv("QUALITY_JUDGE_MODEL", "gpt-5.1"))
    parser.add_argument("--judge-base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8008/v1"))
    parser.add_argument("--judge-api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))

    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument(
        "--payload-mode",
        choices=["minimal", "generator_like", "balanced", "full"],
        default="balanced",
        help="Input payload richness for the judge.",
    )
    parser.add_argument(
        "--payload-fields",
        default=None,
        help=(
            "Optional comma-separated whitelist of payload fields. Overrides --payload-mode. "
            "Available: uid,root_query,generated_question,generated_rubrics,query,question,tree_depth,trace_mode,data_availability,"
            "selected_leaf_ids,selection_trace,merge_trace,statements_flat,evidence_items,rubrics,"
            "selected_leaf_evidence,tree_repr,tree_repr_with_reason,search_tree_text,generation_input_content,tree_meta"
        ),
    )
    parser.add_argument(
        "--exclude-fields",
        default=None,
        help="Optional comma-separated payload fields to remove from mode/default selection.",
    )
    parser.add_argument("--max-leaves", type=int, default=16)
    parser.add_argument("--max-statements-per-leaf", type=int, default=4)
    parser.add_argument("--max-statement-chars", type=int, default=260)
    parser.add_argument("--max-rubric-evidence-chars", type=int, default=260)
    parser.add_argument("--max-flat-statements", type=int, default=40)
    parser.add_argument("--max-tree-repr-chars", type=int, default=5000)
    parser.add_argument(
        "--save-raw-judge-text",
        action="store_true",
        help="Save raw LLM output text in each record for debugging.",
    )
    parser.add_argument(
        "--revised-output",
        default=None,
        help="Optional path to export revised samples as JSONL.",
    )
    parser.add_argument(
        "--include-keep-in-revised",
        action="store_true",
        help="When set with --revised-output, include KEEP samples unchanged.",
    )
    args = parser.parse_args()
    if args.max_records is not None and args.max_records < 0:
        raise ValueError("--max-records must be >= 0")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input path not found: {input_path}")

    input_is_dir = input_path.is_dir()
    folder_mode = input_is_dir or _is_nonempty_str(args.output_dir)

    if folder_mode:
        input_root = input_path if input_is_dir else input_path.parent
        output_root = Path(args.output_dir) if _is_nonempty_str(args.output_dir) else (Path.cwd() / "quality_filter_results")
        summary_path = Path(args.summary_output) if _is_nonempty_str(args.summary_output) else (output_root / "quality_filter_summary.json")
        cache_path = Path(args.cache_path) if _is_nonempty_str(args.cache_path) else (output_root / ".audit_cache.jsonl")
        output_file = None
    else:
        input_root = input_path.parent
        output_file = Path(args.output) if _is_nonempty_str(args.output) else (Path.cwd() / "quality_filter_report.jsonl")
        output_root = output_file.parent
        summary_path = Path(args.summary_output) if _is_nonempty_str(args.summary_output) else (output_root / "quality_filter_summary.json")
        cache_path = Path(args.cache_path) if _is_nonempty_str(args.cache_path) else (output_root / ".audit_cache.jsonl")

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    input_files = list(_iter_input_files(input_path))
    if folder_mode:
        filtered_files = []
        for p in input_files:
            # Avoid recursively re-processing generated outputs if output dir sits under input dir.
            if output_root == p or output_root in p.parents:
                continue
            filtered_files.append(p)
        input_files = filtered_files
    if not input_files:
        raise FileNotFoundError(f"no .jsonl/.json files found under: {input_path}")

    output_targets: Dict[Path, Path] = {}
    for src_file in input_files:
        if folder_mode:
            rel = Path(_to_relpath(src_file, input_root))
            output_targets[src_file] = output_root / rel
        else:
            output_targets[src_file] = output_file

    payload_fields = _resolve_payload_fields(
        mode=args.payload_mode,
        include_fields=args.payload_fields,
        exclude_fields=args.exclude_fields,
    )

    processed_keys: set[str] = set()
    if not args.no_resume:
        processed_keys |= _load_processed_keys_from_cache(cache_path)
        processed_keys |= _load_processed_keys_from_outputs(list(set(output_targets.values())))

    tasks: List[Dict[str, Any]] = []
    reached_max_records = False
    for src_file in input_files:
        out_file = output_targets[src_file]
        for record, src, line_no in _read_records(src_file):
            key = _make_cache_key(src, line_no, input_root)
            if key in processed_keys:
                continue
            tasks.append(
                {
                    "key": key,
                    "record": record,
                    "src": src,
                    "line_no": line_no,
                    "out_file": out_file,
                }
            )
            if args.max_records is not None and len(tasks) >= args.max_records:
                reached_max_records = True
                break
        if reached_max_records:
            break

    thread_local = threading.local()

    def _get_thread_judge() -> LLMJudge:
        judge = getattr(thread_local, "judge", None)
        if judge is None:
            judge = LLMJudge(
                model=args.judge_model,
                base_url=args.judge_base_url,
                api_key=args.judge_api_key,
                timeout_s=args.judge_timeout,
            )
            thread_local.judge = judge
        return judge

    def _process_task(task: Dict[str, Any]) -> Dict[str, Any]:
        record = task["record"]
        src = task["src"]
        line_no = task["line_no"]
        out_file = task["out_file"]
        key = task["key"]

        parse_err = 0
        judge_parse_err = 0

        if record.get("_parse_error"):
            parse_err = 1
            normalized = _normalize_judge_output(None, selected_leaf_ids=[], allowed_source_ids=[])
            normalized["issues"] = [
                {
                    "severity": "error",
                    "code": "input_parse_error",
                    "message": "Failed to parse input record",
                    "suggestion": "Check that the source file is valid JSON/JSONL.",
                }
            ]
            normalized["decision"] = "DROP"
        else:
            judge_payload = _prepare_judge_payload(
                record=record,
                payload_fields=payload_fields,
                max_leaves=args.max_leaves,
                max_statements_per_leaf=args.max_statements_per_leaf,
                max_statement_chars=args.max_statement_chars,
                max_rubric_evidence_chars=args.max_rubric_evidence_chars,
                max_flat_statements=args.max_flat_statements,
                max_tree_repr_chars=args.max_tree_repr_chars,
            )
            try:
                parsed, raw_text = _get_thread_judge().evaluate(judge_payload)
            except Exception as e:
                parsed, raw_text = None, f"<judge_error: {e}>"

            allowed_source_ids: List[str] = []
            selected_ids_for_check = record.get("selected_leaf_ids", []) if isinstance(record, dict) else []
            if isinstance(selected_ids_for_check, list) and selected_ids_for_check:
                allowed_source_ids.extend([str(x) for x in selected_ids_for_check])
            evidence_items = judge_payload.get("evidence_items", [])
            if isinstance(evidence_items, list):
                allowed_source_ids.extend(
                    [str(item.get("id")) for item in evidence_items if isinstance(item, dict) and _is_nonempty_str(item.get("id"))]
                )
            selected_leaf_evidence = judge_payload.get("selected_leaf_evidence", [])
            if isinstance(selected_leaf_evidence, list):
                allowed_source_ids.extend(
                    [str(item.get("id")) for item in selected_leaf_evidence if isinstance(item, dict) and _is_nonempty_str(item.get("id"))]
                )

            normalized = _normalize_judge_output(
                parsed,
                selected_leaf_ids=selected_ids_for_check if isinstance(selected_ids_for_check, list) else [],
                allowed_source_ids=allowed_source_ids,
            )
            if parsed is None:
                judge_parse_err = 1
            if args.save_raw_judge_text:
                normalized["raw_judge_text"] = raw_text

        issue_counts = {
            "error": sum(1 for x in normalized.get("issues", []) if x.get("severity") == "error"),
            "warning": sum(1 for x in normalized.get("issues", []) if x.get("severity") == "warning"),
            "info": sum(1 for x in normalized.get("issues", []) if x.get("severity") == "info"),
        }
        has_error = issue_counts["error"] > 0
        decision = str(normalized.get("decision", "REVISE")).upper()
        decision_ok = decision in {"KEEP", "REVISE"} if args.allow_revise_pass else decision == "KEEP"
        passed = bool(decision_ok and (args.allow_errors or not has_error))

        audit_payload = {
            "_cache_key": key,
            "uid": record.get("uid") if isinstance(record, dict) else None,
            "generated_question": record.get("question", "") if isinstance(record, dict) else "",
            "generated_rubrics": record.get("rubrics", []) if isinstance(record, dict) else [],
            "question": record.get("question", "") if isinstance(record, dict) else "",
            "decision": decision,
            "error_tags": normalized.get("error_tags", []),
            "major_issues": normalized.get("major_issues", []),
            "justification": normalized.get("justification", normalized.get("rationale", "")),
            "fix_plan": normalized.get("fix_plan", normalized.get("improvement_actions", [])),
            "revised_question": normalized.get("revised_question"),
            "revised_rubrics": normalized.get("revised_rubrics"),
            "implementation_guidance": normalized.get("implementation_guidance", {}),
            "issue_counts": issue_counts,
            "issues": normalized.get("issues", []),
            "improvement_actions": normalized.get("improvement_actions", []),
            "rationale": normalized.get("rationale", ""),
            "source_file": str(src),
            "line_no": line_no,
            "pass": passed,
        }
        if "raw_judge_text" in normalized:
            audit_payload["raw_judge_text"] = normalized["raw_judge_text"]

        # Keep original sample fields and append audit outputs in a dedicated block.
        if isinstance(record, dict):
            result = dict(record)
        else:
            result = {"_raw_record": record}
        result["_cache_key"] = key
        result["_audit"] = audit_payload

        revised_obj = None
        if isinstance(record, dict):
            if decision == "REVISE":
                patched = dict(record)
                revised_question = normalized.get("revised_question")
                revised_rubrics = normalized.get("revised_rubrics")
                if _is_nonempty_str(revised_question):
                    patched["question"] = revised_question
                if isinstance(revised_rubrics, list) and revised_rubrics:
                    patched["rubrics"] = revised_rubrics
                patched["_audit"] = {
                    "decision": decision,
                    "fix_plan": normalized.get("fix_plan", []),
                }
                revised_obj = patched
            elif decision == "KEEP" and args.include_keep_in_revised:
                revised_obj = dict(record)

        return {
            "key": key,
            "result": result,
            "out_file": out_file,
            "cache_entry": {
                "key": key,
                "source_file": str(src),
                "line_no": line_no,
                "output_file": str(out_file),
                "decision": decision,
                "ts": int(time.time()),
            },
            "revised_obj": revised_obj,
            "parse_error": parse_err,
            "judge_parse_error": judge_parse_err,
        }

    results: List[Dict[str, Any]] = []
    parse_error_count = 0
    judge_parse_error_count = 0
    revised_count = 0
    written = 0

    revised_path = Path(args.revised_output) if _is_nonempty_str(args.revised_output) else None
    if revised_path is not None:
        revised_path.parent.mkdir(parents=True, exist_ok=True)

    if args.workers <= 1:
        iterator = (_process_task(t) for t in tasks)
    else:
        max_workers = max(1, int(args.workers))
        pool = ThreadPoolExecutor(max_workers=max_workers)
        futures = [pool.submit(_process_task, t) for t in tasks]
        iterator = (f.result() for f in as_completed(futures))

    cache_flush_every = max(1, int(args.cache_flush_every))
    total_tasks = len(tasks)
    pbar = None
    if not args.no_progress and tqdm is not None:
        pbar = tqdm(total=total_tasks, desc="quality-filter", unit="sample", dynamic_ncols=True)
    elif not args.no_progress and total_tasks > 0:
        print(f"[quality-filter] processing {total_tasks} samples...")

    try:
        with cache_path.open("a", encoding="utf-8") as cache_fh:
            for item in iterator:
                result = item["result"]
                out_file: Path = item["out_file"]
                _append_jsonl(out_file, result)

                cache_fh.write(json.dumps(item["cache_entry"], ensure_ascii=False) + "\n")
                written += 1
                if written % cache_flush_every == 0:
                    cache_fh.flush()

                parse_error_count += int(item["parse_error"])
                judge_parse_error_count += int(item["judge_parse_error"])
                results.append(result)

                if revised_path is not None and item["revised_obj"] is not None:
                    _append_jsonl(revised_path, item["revised_obj"])
                    revised_count += 1

                if pbar is not None:
                    pbar.update(1)

            cache_fh.flush()
    finally:
        if pbar is not None:
            pbar.close()

    if args.workers > 1:
        pool.shutdown(wait=True)

    summary = _build_summary(results)
    summary["input"] = str(input_path)
    summary["output_root"] = str(output_root)
    summary["cache_path"] = str(cache_path)
    summary["workers"] = max(1, int(args.workers))
    summary["max_records"] = args.max_records
    summary["resume_enabled"] = not args.no_resume
    summary["skipped_from_cache_or_existing"] = len(processed_keys)
    summary["queued_tasks"] = len(tasks)
    if output_file is not None:
        summary["output_file"] = str(output_file)
    summary["parse_error_count"] = parse_error_count
    summary["judge_parse_error_count"] = judge_parse_error_count
    summary["allow_errors"] = args.allow_errors
    summary["allow_revise_pass"] = args.allow_revise_pass
    summary["judge_model"] = args.judge_model
    summary["payload_mode"] = args.payload_mode
    summary["payload_fields"] = sorted(list(payload_fields))
    summary["revised_records"] = revised_count

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[quality-filter] processed={summary['total']} passed={summary['passed']} failed={summary['failed']}")
    print(f"[quality-filter] decision_counts={summary['decision_counts']}")
    print(f"[quality-filter] skipped_from_cache_or_existing={summary['skipped_from_cache_or_existing']}")
    print(f"[quality-filter] queued_tasks={summary['queued_tasks']}")
    print(f"[quality-filter] max_records={summary['max_records']}")
    print(f"[quality-filter] parse_error_count={summary['parse_error_count']}")
    print(f"[quality-filter] judge_parse_error_count={summary['judge_parse_error_count']}")
    if output_file is not None:
        print(f"[quality-filter] output_file={output_file}")
    else:
        print(f"[quality-filter] output_root={output_root}")
    print(f"[quality-filter] cache_path={cache_path}")
    print(f"[quality-filter] summary={summary_path}")
    if revised_path is not None:
        print(f"[quality-filter] revised_output={revised_path} count={revised_count}")


if __name__ == "__main__":
    main()
