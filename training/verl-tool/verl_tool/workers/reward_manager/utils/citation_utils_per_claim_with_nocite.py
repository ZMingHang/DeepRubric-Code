import re
import os
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

# from open_instruct.search_rewards.utils.run_utils import run_litellm, run_litellm_async
from openai import OpenAI, AsyncOpenAI
import random


def _parse_base_urls(raw_value: Optional[str], default_urls: List[str]) -> List[str]:
    if raw_value:
        urls = [url.strip() for url in raw_value.split(",") if url.strip()]
        if urls:
            return urls
    return default_urls


_citation_base_urls = _parse_base_urls(
    os.getenv("CITATION_OPENAI_BASE_URLS") or os.getenv("OPENAI_BASE_URL"),
    ["http://localhost:8000/v1"],
)
_async_citation_base_urls = _parse_base_urls(
    os.getenv("CITATION_ASYNC_OPENAI_BASE_URLS") or os.getenv("OPENAI_BASE_URL"),
    ["http://localhost:8000/v1"],
)

citation_clients = [
    OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=base_url)
    for base_url in _citation_base_urls
]
async_citation_clients = [
    AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=base_url)
    for base_url in _async_citation_base_urls
]

DEFAULT_CITATION_MODEL = os.getenv("CITATION_JUDGE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")


def _choose_citation_client() -> OpenAI:
    return random.choice(citation_clients)


def _choose_async_citation_client() -> AsyncOpenAI:
    return random.choice(async_citation_clients)


def _call_chat(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    max_tokens: int = 512,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    resp = _choose_citation_client().chat.completions.create(
        model=model_name or DEFAULT_CITATION_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
        top_p=1.0,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
    )
    return resp.choices[0].message.content or ""


citation_recall_has_citation_prompt = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether this statement is supported by the snippet. Please use the following scale to generate your rating:
- [[Fully supported]] - Most information in the statement is supported by or extracted from the snippet. This applies only to cases where the statement and parts of the snippet are almost identical.
- [[Partially supported]] - More than half of the content in the statement is supported by the snippet, but a small portion is either not mentioned or contradicts the snippet. For example, if the statement has two key points and the snippet supports only one of them, it should be considered [Partially supported].
- [[No support]] - The statement is largely unrelated to the snippet, or most key points in the statement do not align with the content of the snippet.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<statement>
{statement}
</statement>

<snippet>
{concatenated_cited_snippets}
</snippet>"""


# citation_recall_no_citation_prompt = """You are an expert in evaluating text quality. You will receive a user's question regarding their uploaded document (due to the length of the document, it is not shown to you), an AI assistant's response based on the document, and a sentence from the response. Your task is to determine whether this sentence is a factual statement made based on the information in the document that requires citation, rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.
# Ensure that you do not use any other external information during your evaluation.
# Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".

# <question>
# {question}
# </question>

# <response>
# {full_response}
# </response>

# <statement>
# {statement}
# </statement>"""
citation_recall_no_citation_prompt = """You are an expert in evaluating citation coverage for source-grounded QA.

You will receive:
1. the user's question,
2. the assistant's final answer,
3. one sentence from the final answer.

The external source material used by the assistant is not shown to you.
Your task is to determine whether the sentence is a factual claim that should be grounded with an explicit citation to external sources.

Answer [[Yes]] if the sentence introduces a concrete factual claim, empirical result, dataset/model detail, number, comparison, attribution, or document-specific statement that requires citation.
Answer [[No]] if the sentence is only an introductory phrase, transition, high-level organization sentence, restatement of the user's question, or a reasoning/summary sentence that directly follows from already cited claims in the final answer.

Do not use external knowledge. Judge only whether the sentence needs citation in the final answer.

<question>
{question}
</question>

<final_answer>
{full_response}
</final_answer>

<statement>
{statement}
</statement>

Please output in the format:
Need Citation: [[Yes/No]] Analysis: ...
"""

citation_precision_prompt = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether the snippet contains some key information of the statement. Please use the following grades to generate the rating:
- [[Relevant]] - Some key points of the statement are supported by the snippet or extracted from it.
- [[Unrelevant]] - The statement is almost unrelated to the snippet, or all key points of the statement are inconsistent with the snippet content.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<statement>
{statement}
</statement>

<snippet>
{concatenated_cited_snippets}
</snippet>"""


def extract_claims_and_corresponding_citation_ids(
    response: str,
    split_non_cited_parts_by_newlines: bool = False,
    split_non_cited_parts_by_sentences: bool = False,
) -> Dict[str, List[str]]:
    claims = {}

    cite_pattern = r"<cite id=([\"\']?)([^\"\'>\s]+)\1[^>]*>([^<]+)</cite>"
    cite_matches = re.findall(cite_pattern, response)

    cite_tag_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>"
    non_cited_parts = re.split(cite_tag_pattern, response)

    if split_non_cited_parts_by_newlines:
        further_split_parts = []
        for part in non_cited_parts:
            further_split_parts.extend(re.split(r"\n", part))
        non_cited_parts = further_split_parts

    if split_non_cited_parts_by_sentences:
        further_split_parts = []
        for part in non_cited_parts:
            further_split_parts.extend(re.split(r"[.!?]", part))
        non_cited_parts = further_split_parts

    for part in non_cited_parts:
        part = part.strip()
        if part:
            claims[part] = []

    for _, citation_ids, cited_text in cite_matches:
        cited_text = cited_text.strip()
        if cited_text:
            citation_id_list = [
                citation_id.strip()
                for citation_id in citation_ids.split(",")
                if citation_id.strip()
            ]
            if citation_id_list:
                claims[cited_text] = citation_id_list

    return claims


def _is_meaningful_cited_text(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False

    if re.fullmatch(r"[^\w\u4e00-\u9fff]+", cleaned):
        return False

    non_space = re.sub(r"\s+", "", cleaned)
    if not non_space:
        return False

    alpha_num_or_cjk = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", non_space))
    symbol_ratio = 1.0 - (alpha_num_or_cjk / len(non_space))
    lexical_units = len(
        re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]", cleaned)
    )

    if alpha_num_or_cjk < 2 and len(non_space) < 6:
        return False
    if symbol_ratio > 0.7 and alpha_num_or_cjk < 4:
        return False
    if lexical_units < 4:
        return False
    if lexical_units < 6 and len(non_space) < 18:
        return False

    return True


def _extract_valid_citation_positions(response: str, citations: Dict[str, str]) -> List[float]:
    if not response:
        return []

    cite_tag_pattern = r"<cite id=([\"\']?)([^\"\'>\s]+)\1[^>]*>[^<]+</cite>"
    response_len = max(1, len(response))
    positions_by_citation_id: Dict[str, List[float]] = {}

    for match in re.finditer(cite_tag_pattern, response):
        citation_ids_raw = match.group(2) or ""
        citation_ids = [
            citation_id.strip()
            for citation_id in citation_ids_raw.split(",")
            if citation_id.strip()
        ]
        if not citation_ids:
            continue
        valid_citation_ids = sorted(
            {citation_id for citation_id in citation_ids if citation_id in citations}
        )
        if not valid_citation_ids:
            continue
        position = ((match.start() + match.end()) / 2.0) / response_len
        normalized_position = max(0.0, min(1.0, position))
        for citation_id in valid_citation_ids:
            positions_by_citation_id.setdefault(citation_id, []).append(
                normalized_position
            )

    positions: List[float] = []
    for citation_id in sorted(positions_by_citation_id):
        citation_positions = positions_by_citation_id[citation_id]
        positions.append(sum(citation_positions) / len(citation_positions))

    return positions


def _score_citation_distribution(positions: List[float]) -> float:
    if len(positions) == 0:
        return 0.0
    if len(positions) == 1:
        return 0.3

    ordered_positions = sorted(positions)
    span = ordered_positions[-1] - ordered_positions[0]
    span_score = max(0.0, min(1.0, span / 0.6))

    mean_position = sum(ordered_positions) / len(ordered_positions)
    center_score = max(0.0, 1.0 - abs(mean_position - 0.5) / 0.5)

    boundaries = [0.0] + ordered_positions + [1.0]
    gaps = [
        boundaries[idx + 1] - boundaries[idx]
        for idx in range(len(boundaries) - 1)
    ]
    ideal_gap = 1.0 / (len(ordered_positions) + 1)
    avg_gap_deviation = sum(abs(gap - ideal_gap) for gap in gaps) / len(gaps)
    uniformity_score = max(0.0, 1.0 - (avg_gap_deviation / ideal_gap))

    return 0.4 * span_score + 0.4 * uniformity_score + 0.2 * center_score


def _score_citation_count(
    valid_unique_citation_count: int, max_count_for_full_score: int = 6
) -> float:
    if max_count_for_full_score <= 0:
        return 0.0
    capped_count = max(0, min(valid_unique_citation_count, max_count_for_full_score))
    return capped_count / max_count_for_full_score


def score_citation_format(
    claims: Dict[str, List[str]], citations: Dict[str, str], response: str = ""
) -> float:
    all_citations = []
    cited_claim_texts = []
    for claim_text, citation_ids in claims.items():
        if citation_ids:
            cited_claim_texts.append(claim_text)
            all_citations.extend(citation_ids)
    all_citations = list(set(all_citations))
    if len(all_citations) == 0:
        return 0
    valid_citations = [citation for citation in all_citations if citation in citations]
    citation_id_validity = len(valid_citations) / len(all_citations)

    if len(cited_claim_texts) == 0:
        return 0
    meaningful_claims = [
        claim_text
        for claim_text in cited_claim_texts
        if _is_meaningful_cited_text(claim_text)
    ]
    meaningful_claim_ratio = len(meaningful_claims) / len(cited_claim_texts)

    base_quality = citation_id_validity * meaningful_claim_ratio

    count_reward = _score_citation_count(
        len(valid_citations), max_count_for_full_score=6
    )

    if response:
        valid_citation_positions = _extract_valid_citation_positions(response, citations)
        distribution_reward = _score_citation_distribution(valid_citation_positions)
    else:
        distribution_reward = 1.0

    format_multiplier = 0.7 + 0.1 * distribution_reward + 0.2 * count_reward
    return base_quality * format_multiplier


def score_citation_f1(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    recall = score_citation_recall(question, claim, concatenated_citations, full_response)
    precision = score_citation_precision(question, claim, concatenated_citations)
    if recall + precision == 0:
        return 0
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


def score_citation_recall(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    if len(concatenated_citations) == 0:
        return score_no_citation_recall(question, claim, full_response)
    else:
        return score_with_citation_recall(question, claim, concatenated_citations)


def score_with_citation_recall(question: str, claim: str, concatenated_citations: str) -> float:
    user_prompt = citation_recall_has_citation_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    response = _call_chat(
        user_prompt=user_prompt,
        system_prompt=None,
        model_name=os.environ.get("CITATION_JUDGE_MODEL", DEFAULT_CITATION_MODEL),
        max_tokens=512,
    )
    return extract_recall_rating_from_response(response)


def score_no_citation_recall(question: str, claim: str, full_response: str) -> float:
    user_prompt = citation_recall_no_citation_prompt.format(
        question=question, statement=claim, full_response=full_response
    )
    response = _call_chat(
        user_prompt=user_prompt,
        system_prompt=None,
        model_name=os.environ.get("CITATION_JUDGE_MODEL", DEFAULT_CITATION_MODEL),
        max_tokens=512,
    )
    return 1 - extract_yes_no_from_response(response)


def score_citation_precision(question: str, claim: str, concatenated_citations: str) -> float:
    if len(concatenated_citations) == 0:
        return 1
    user_prompt = citation_precision_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    response = _call_chat(
        user_prompt=user_prompt,
        system_prompt=None,
        model_name=os.environ.get("CITATION_JUDGE_MODEL", DEFAULT_CITATION_MODEL),
        max_tokens=512,
    )
    return extract_relevant_rating_from_response(response)


def extract_recall_rating_from_response(response: str) -> float:
    rating = re.search(r"Rating: \[\[(.*)\]\]", response)
    if rating:
        extracted_text = rating.group(1).strip().lower()
        if extracted_text == "fully supported":
            return 1.0
        elif extracted_text == "partially supported":
            return 0.5
        elif extracted_text == "no support":
            return 0.0
        else:
            return 0.0
    else:
        return 0.0


def extract_yes_no_from_response(response: str) -> int:
    yes_no = re.search(r"Need Citation: \[\[(.*)\]\]", response)
    if yes_no:
        extracted_text = yes_no.group(1).strip().lower()
        if extracted_text == "yes":
            return 1
        elif extracted_text == "no":
            return 0
        else:
            return 0
    else:
        return 0


def extract_relevant_rating_from_response(response: str) -> int:
    rating = re.search(r"Rating: \[\[(.*)\]\]", response)
    if rating:
        extracted_text = rating.group(1).strip().lower()
        if extracted_text == "relevant":
            return 1
        elif extracted_text == "unrelevant":
            return 0
        else:
            return 0
    else:
        return 0


def _score_in_context_citations_per_claim(
    question: str,
    response: str,
    citations: Dict[str, str],
    claims: Dict[str, List[str]],
) -> float:
    def concatenate_citations(citation_ids: List[str], citations: Dict[str, str]) -> str:
        if len(citation_ids) == 0:
            return ""
        return "\n\n".join(
            [citations[citation_id] for citation_id in citation_ids if citation_id in citations]
        )

    avg_f1 = 0.0
    for claim_text, citation_ids in claims.items():
        concatenated_citations = concatenate_citations(citation_ids, citations)
        avg_f1 += score_citation_f1(
            question, claim_text, concatenated_citations, response
        )
    if len(claims) > 0:
        avg_f1 /= len(claims)
    return avg_f1


def score_in_context_citations(question: str, response: str, citations: Dict[str, str]) -> float:
    """
    Variant based on the current citation_utils.py, but:
    1. always uses per-claim scoring
    2. no-citation claims are also scored
    3. keeps the new multi-citation parsing and the new citation_format reward
    """
    if not citations:
        return 0

    claims = extract_claims_and_corresponding_citation_ids(response)
    citation_format_reward = score_citation_format(claims, citations, response=response)
    avg_f1 = _score_in_context_citations_per_claim(question, response, citations, claims)
    return 0.6 * avg_f1 + 0.4 * citation_format_reward
