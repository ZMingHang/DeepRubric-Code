import re
import os
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

# from open_instruct.search_rewards.utils.run_utils import run_litellm, run_litellm_async
from openai import OpenAI, AsyncOpenAI

citation_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
)
async_citation_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
)
DEFAULT_CITATION_MODEL = os.getenv("CITATION_JUDGE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")


def _call_chat(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    max_tokens: int = 800,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    resp = citation_client.chat.completions.create(
        model=model_name or DEFAULT_CITATION_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
        top_p=1.0,
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


citation_recall_no_citation_prompt = """You are an expert in evaluating text quality. You will receive a user's question regarding their uploaded document (due to the length of the document, it is not shown to you), an AI assistant's response based on the document, and a sentence from the response. Your task is to determine whether this sentence is a factual statement made based on the information in the document that requires citation, rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.
Ensure that you do not use any other external information during your evaluation.
Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".

<question>
{question}
</question>

<response>
{full_response}
</response>

<statement>
{statement}
</statement>"""


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

citation_single_pass_prompt = """You are an expert in evaluating citation quality for document-grounded QA.
You will receive:
1) the user question,
2) only the cited claims extracted from the assistant response (each has claim_id + citation ids),
3) citation snippets keyed by citation id (only ids used by those cited claims).

For EACH claim, score two metrics using the SAME rubric as per-claim mode:
- Recall must be one of {{0, 0.5, 1}}:
  - 1: Fully supported
  - 0.5: Partially supported
  - 0: No support
- Precision must be one of {{0, 1}}:
  - 1: Relevant
  - 0: Unrelevant

Do not output any values outside these discrete sets.

Do not use any external knowledge.
Output ONLY a JSON object (no markdown, no extra text):
{{"claims":[{{"claim_id":"1","recall":1,"precision":1}},{{"claim_id":"2","recall":0.5,"precision":0}}]}}

Score every provided claim_id exactly once.

<question>
{question}
</question>

<cited_claims>
{cited_claims_text}
</cited_claims>

<citations>
{citations_text}
</citations>"""


def extract_claims_and_corresponding_citation_ids(
    response: str, 
    split_non_cited_parts_by_newlines: bool = False,
    split_non_cited_parts_by_sentences: bool = False,
    ) -> Dict[str, List[str]]:
    """
    Example response:
    "The Great Wall of China, one of the most iconic structures in human history, stretches approximately 13,000 miles across northern China. <cite id=a1b2c3d4>Construction of the Great Wall began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644).</cite> The wall was primarily constructed as a defensive fortification to protect Chinese states from invasions by nomadic groups from the north. <cite id=e5f6g7h8>The wall incorporates various materials including stone, brick, tamped earth, wood, and other materials, with different sections built using locally available resources.</cite> Contrary to popular belief, <cite id=i9j0k1l2>the Great Wall is not visible from space with the naked eye, a myth that has been debunked by astronauts and satellite imagery.</cite>"
    """
    claims = {}

    # Use findall to get all cite tags and their content
    cite_pattern = r"<cite id=([\"\']?)([^\"\'>\s]+)\1[^>]*>([^<]+)</cite>"
    cite_matches = re.findall(cite_pattern, response)
    
    # Split the response by cite tags to get non-cited text
    cite_tag_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>"
    non_cited_parts = re.split(cite_tag_pattern, response)
    
    # Further split non-cited parts by newlines if requested
    if split_non_cited_parts_by_newlines:
        further_split_parts = []
        for part in non_cited_parts:
            further_split_parts.extend(re.split(r"\n", part))
        non_cited_parts = further_split_parts
    
    # Further split non-cited parts by sentences if requested
    if split_non_cited_parts_by_sentences:
        further_split_parts = []
        for part in non_cited_parts:
            further_split_parts.extend(re.split(r"[.!?]", part))
        non_cited_parts = further_split_parts
    
    # Add non-cited text (parts between cite tags)
    for part in non_cited_parts:
        part = part.strip()
        if part:
            claims[part] = []
    
    # Add cited text with their citation IDs
    for _, citation_ids, cited_text in cite_matches:
        cited_text = cited_text.strip()
        if cited_text:
            citation_id_list = [citation_id.strip() for citation_id in citation_ids.split(",") if citation_id.strip()]
            if citation_id_list:
                claims[cited_text] = citation_id_list

    return claims
    

def _format_citations_for_single_pass_prompt(citations: Dict[str, str], max_chars_per_snippet: int = 1200, max_total_chars: int = 12000) -> str:
    chunks = []
    total_chars = 0
    for citation_id, citation_text in citations.items():
        cleaned = (citation_text or "").strip()
        if len(cleaned) > max_chars_per_snippet:
            cleaned = cleaned[:max_chars_per_snippet] + " ..."
        chunk = f"[{citation_id}] {cleaned}"
        if total_chars + len(chunk) > max_total_chars:
            break
        chunks.append(chunk)
        total_chars += len(chunk)
    return "\n".join(chunks)


def _extract_cited_claims(claims: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {claim_text: citation_ids for claim_text, citation_ids in claims.items() if len(citation_ids) > 0}


def _format_cited_claims_for_single_pass_prompt(
    cited_claims: Dict[str, List[str]],
    max_claim_chars: int = 800,
    max_total_chars: int = 6000,
) -> Tuple[str, List[str]]:
    chunks = []
    claim_ids = []
    total_chars = 0
    for idx, (claim_text, citation_ids) in enumerate(cited_claims.items(), start=1):
        claim = (claim_text or "").strip()
        if len(claim) > max_claim_chars:
            claim = claim[:max_claim_chars] + " ..."
        claim_id = str(idx)
        citation_ids_text = ",".join(citation_ids)
        chunk = f"- claim_id={claim_id} ids=[{citation_ids_text}] claim={claim}"
        if total_chars + len(chunk) > max_total_chars:
            break
        chunks.append(chunk)
        claim_ids.append(claim_id)
        total_chars += len(chunk)
    return "\n".join(chunks), claim_ids


def _to_bounded_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _extract_recall_precision_from_json_response(
    response: str,
    expected_claim_ids: Optional[List[str]] = None,
) -> Tuple[float, float]:
    text = (response or "").strip()
    if not text:
        return 0.0, 0.0

    candidates = [text]

    fenced_json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_json_match:
        candidates.append(fenced_json_match.group(1).strip())

    json_object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_object_match:
        candidates.append(json_object_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue

        claims_items = parsed.get("claims")
        if isinstance(claims_items, list):
            per_claim_scores = {}
            valid_scores_in_order = []
            for item in claims_items:
                if not isinstance(item, dict):
                    continue
                recall_value = item.get("recall", item.get("Recall"))
                precision_value = item.get("precision", item.get("Precision"))
                if recall_value is None or precision_value is None:
                    continue
                recall_score = _to_bounded_float(recall_value)
                precision_score = _to_bounded_float(precision_value)
                valid_scores_in_order.append((recall_score, precision_score))

                claim_id_raw = item.get("claim_id", item.get("claimId"))
                if claim_id_raw is not None:
                    per_claim_scores[str(claim_id_raw).strip()] = (recall_score, precision_score)

            if expected_claim_ids:
                matched_scores = [per_claim_scores[claim_id] for claim_id in expected_claim_ids if claim_id in per_claim_scores]
                if matched_scores:
                    missing_count = len(expected_claim_ids) - len(matched_scores)
                    recalls = [score[0] for score in matched_scores] + [0.0] * missing_count
                    precisions = [score[1] for score in matched_scores] + [0.0] * missing_count
                    return sum(recalls) / len(expected_claim_ids), sum(precisions) / len(expected_claim_ids)
            elif valid_scores_in_order:
                recalls = [score[0] for score in valid_scores_in_order]
                precisions = [score[1] for score in valid_scores_in_order]
                return sum(recalls) / len(recalls), sum(precisions) / len(precisions)

        recall_value = parsed.get("recall", parsed.get("Recall"))
        precision_value = parsed.get("precision", parsed.get("Precision"))
        if recall_value is None or precision_value is None:
            continue

        return _to_bounded_float(recall_value), _to_bounded_float(precision_value)

    # Safe fallback if the model violates format.
    recall_fallback = _to_bounded_float(re.search(r"recall[^0-9]*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE).group(1)) if re.search(r"recall[^0-9]*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE) else 0.0
    precision_fallback = _to_bounded_float(re.search(r"precision[^0-9]*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE).group(1)) if re.search(r"precision[^0-9]*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE) else 0.0
    return recall_fallback, precision_fallback


def score_citation_f1_single_pass(question: str, cited_claims: Dict[str, List[str]], citations: Dict[str, str]) -> float:
    if len(cited_claims) == 0:
        return 0.0

    used_citation_ids = list({citation_id for citation_ids in cited_claims.values() for citation_id in citation_ids if citation_id in citations})
    used_citations = {citation_id: citations[citation_id] for citation_id in used_citation_ids}
    citations_text = _format_citations_for_single_pass_prompt(used_citations)
    cited_claims_text, claim_ids = _format_cited_claims_for_single_pass_prompt(cited_claims)
    if len(claim_ids) == 0:
        return 0.0
    user_prompt = citation_single_pass_prompt.format(
        question=question,
        cited_claims_text=cited_claims_text,
        citations_text=citations_text,
    )
    judge_response = _call_chat(
        user_prompt=user_prompt,
        system_prompt=None,
        model_name=os.environ.get("CITATION_JUDGE_MODEL", DEFAULT_CITATION_MODEL),
        max_tokens=min(2000, max(300, 120 + 45 * len(claim_ids))),
    )
    recall, precision = _extract_recall_precision_from_json_response(judge_response, expected_claim_ids=claim_ids)
    if recall + precision == 0:
        return 0.0
    return 2 * (recall * precision) / (recall + precision)


def _score_in_context_citations_per_claim(question: str, response: str, citations: Dict[str, str], cited_claims: Dict[str, List[str]]) -> float:
    def concatenate_citations(citation_ids: List[str], citations: Dict[str, str]) -> str:
        if len(citation_ids) == 0:
            return ""
        return "\n\n".join([citations[citation_id] for citation_id in citation_ids if citation_id in citations])

    avg_f1 = 0.0
    for claim_text, citation_ids in cited_claims.items():
        concatenated_citations = concatenate_citations(citation_ids, citations)
        avg_f1 += score_citation_f1(question, claim_text, concatenated_citations, response)
    if len(cited_claims) > 0:
        avg_f1 /= len(cited_claims)
    return avg_f1


def score_in_context_citations(question: str, response: str, citations: Dict[str, str]) -> float:
    """
    Compute the cumulative weighted score for a system response such that the final score is between 0 and 1.
    :param response:
    :param citations:
    :return: final weighted score with and without the static components
    """
    if not citations:
        return 0

    claims = extract_claims_and_corresponding_citation_ids(response)
    cited_claims = _extract_cited_claims(claims)

    citation_format_reward = score_citation_format(claims, citations)

    score_mode = os.environ.get("CITATION_SCORE_MODE", "single_pass").strip().lower()
    if score_mode == "per_claim":
        avg_f1 = _score_in_context_citations_per_claim(question, response, citations, cited_claims)
    else:
        avg_f1 = score_citation_f1_single_pass(question, cited_claims, citations)

    return 0.6 * avg_f1 + 0.4 * citation_format_reward


# def score_in_context_citations(question: str, response: str, citations: Dict[str, str]) -> Dict[str, float]:
#     """
#     Compute the cumulative weighted score for a system response such that the final score is between 0 and 1.
#     :param response:
#     :param citations:
#     :return: final weighted score with and without the static components
#     """
#     if not citations:
#         return 0
    
#     def concatenate_citations(citation_ids: List[str], citations: Dict[str, str]) -> str:
#         if len(citation_ids) == 0:
#             return ""
#         return "\n\n".join([citations[citation_id] for citation_id in citation_ids if citation_id in citations])

#     claims = extract_claims_and_corresponding_citation_ids(response)

#     citation_format_reward = score_citation_format(claims, citations)

#     total_claims = len(claims)
#     matched_claims = 0
#     for _, citation_ids in claims.items():
#         if any(c_id in citations for c_id in citation_ids):
#             matched_claims += 1

#     recall = matched_claims / total_claims if total_claims else 0.0
#     precision = matched_claims / len(citations) if citations else 0.0

#     if recall + precision == 0:
#         f1 = 0.0
#     else:
#         f1 = 2 * (recall * precision) / (recall + precision)

#     return 0.6 * f1 + 0.4 * citation_format_reward



def _is_meaningful_cited_text(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False

    # Pure symbols/punctuation like "...", "???", "---" are meaningless as cited claims.
    if re.fullmatch(r"[^\w\u4e00-\u9fff]+", cleaned):
        return False

    non_space = re.sub(r"\s+", "", cleaned)
    if not non_space:
        return False

    alpha_num_or_cjk = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", non_space))
    symbol_ratio = 1.0 - (alpha_num_or_cjk / len(non_space))

    # Very short snippets with almost no lexical content are likely citation spam.
    if alpha_num_or_cjk < 2 and len(non_space) < 6:
        return False
    if symbol_ratio > 0.7 and alpha_num_or_cjk < 4:
        return False

    return True


def score_citation_format(claims: Dict[str, List[str]], citations: Dict[str, str]) -> float:
    """
    Check if the model has hallucinated citations.
    """
    all_citations = []
    cited_claim_texts = []
    for claim_text, citation_ids in claims.items():
        if citation_ids:
            cited_claim_texts.append(claim_text)
            all_citations.extend(citation_ids)
    all_citations = list(set(all_citations))
    if len(all_citations) == 0:
        # If there are no citations, return 0
        return 0
    valid_citations = [citation for citation in all_citations if citation in citations]
    citation_id_validity = len(valid_citations) / len(all_citations)

    if len(cited_claim_texts) == 0:
        return 0
    meaningful_claims = [claim_text for claim_text in cited_claim_texts if _is_meaningful_cited_text(claim_text)]
    meaningful_claim_ratio = len(meaningful_claims) / len(cited_claim_texts)

    # Apply quality penalty on top of citation-id validity.
    return citation_id_validity * meaningful_claim_ratio


def score_citation_f1(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    recall = score_citation_recall(question, claim, concatenated_citations, full_response)
    precision = score_citation_precision(question, claim, concatenated_citations)
    # avoid division by zero
    if recall + precision == 0:
        return 0
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


def score_citation_recall(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    if len(concatenated_citations) == 0:
        # return score_no_citation_recall(question, claim, full_response)
        return 0
    else:
        return score_with_citation_recall(question, claim, concatenated_citations)


def score_with_citation_recall(question: str, claim: str, concatenated_citations: str) -> float:
    user_prompt = citation_recall_has_citation_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    # response = run_litellm(
    #     model_name=os.environ.get("CITATION_JUDGE_MODEL", "gpt-4o-mini"),
    #     system_prompt=None,
    #     user_prompt=user_prompt,
    #     max_tokens=800,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0,
    # )
    response = _call_chat(
        user_prompt=user_prompt,
        system_prompt=None,
        model_name=os.environ.get("CITATION_JUDGE_MODEL", DEFAULT_CITATION_MODEL),
        max_tokens=800,
    )
    return extract_recall_rating_from_response(response)


def score_no_citation_recall(question: str, claim: str, full_response: str) -> float:
    user_prompt = citation_recall_no_citation_prompt.format(
        question=question, statement=claim, full_response=full_response
    )
    # response = run_litellm(
    #     model_name=os.environ.get("CITATION_JUDGE_MODEL", "gpt-4o-mini"),
    #     system_prompt=None,
    #     user_prompt=user_prompt,
    #     max_tokens=800,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0,
    # )
    response = _call_chat(
        user_prompt=user_prompt,
        system_prompt=None,
        model_name=os.environ.get("CITATION_JUDGE_MODEL", DEFAULT_CITATION_MODEL),
        max_tokens=800,
    )
    # "yes" means it is a factual claim, but no citation is provided.
    return 1 - extract_yes_no_from_response(response)


def score_citation_precision(question: str, claim: str, concatenated_citations: str) -> float:
    if len(concatenated_citations) == 0:
        return 1
    user_prompt = citation_precision_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    # response = run_litellm(
    #     model_name=os.environ.get("CITATION_JUDGE_MODEL", "gpt-4o-mini"),
    #     system_prompt=None,
    #     user_prompt=user_prompt,
    #     max_tokens=800,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0,
    # )
    response = _call_chat(
        user_prompt=user_prompt,
        system_prompt=None,
        model_name=os.environ.get("CITATION_JUDGE_MODEL", DEFAULT_CITATION_MODEL),
        max_tokens=800,
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
            # If the extracted text doesn't match any expected values, return 0 as default
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
            # If the extracted text is neither "yes" nor "no", return 0 as default
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
            # If the extracted text is neither "relevant" nor "unrelevant", return 0 as default
            return 0
    else:
        return 0
