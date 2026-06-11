# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A reward manager that reuses the long-form rubric scoring logic from
`search_rewards/longform_rubric_rewards.py`.

It computes a scalar reward per trajectory by:
1) Decoding the model response from `DataProto`.
2) Calling `compute_weighted_rubric_reward_with_citation_and_format_reward`
   to get rubric/citation/format/search-turn based scores.
3) Writing the resulting reward to the last generated token position and
   exposing detailed log values in `reward_extra_info`.
"""

from collections import defaultdict
from typing import Any, Dict, List
import os
import re
import json
import random

import torch
from openai import OpenAI
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _parse_base_urls(raw_value: str | None, default_urls: List[str]) -> List[str]:
    if raw_value:
        urls = [url.strip() for url in raw_value.split(",") if url.strip()]
        if urls:
            return urls
    return default_urls



@register("longform_rubric")
class LongformRubricRewardManager(AbstractRewardManager):
    """
    Reward manager that wraps the longform rubric scoring function.

    The ground_truth is expected to be a dict with rubrics, citations, etc.,
    as used by search_rewards/longform_rubric_rewards.py.
    """

    name = "longform_rubric"
    TOOL_CALL_NAMES = {"search", "browse", "scholar", "tool_search", "tool_browse", "tool_scholar"}

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        mcp_parser_name: str | None = None,
        use_general_rubric: bool = False,
        no_citation_reward: bool = False,
        use_likert_rubric: bool = False,
        judge_model: str | None = None,
        score_scale: float = 4.0,
        rubric_weight: float = 0.5,
        format_weight: float = 0.15,
        search_turn_weight: float = 0.1,
        citation_reward_weight: float = 0.2,
        answer_length_weight: float = 0.05,
        answer_length_upper_bound: int = 3000,
        # rubric_weight: float = 0.6,
        # format_weight: float = 0.1,
        # search_turn_weight: float = 0.15,
        # citation_reward_weight: float = 0.15,
        api_timeout: float = 30.0,
        max_concurrent_api: int = 1,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.mcp_parser_name = mcp_parser_name
        self.use_general_rubric = use_general_rubric
        self.no_citation_reward = no_citation_reward
        self.use_likert_rubric = use_likert_rubric
        self.judge_model = judge_model or os.getenv("RUBRIC_JUDGE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

        self.score_scale = score_scale
        self.format_weight = format_weight
        self.rubric_weight = rubric_weight
        self.search_turn_weight = search_turn_weight
        self.citation_reward_weight = citation_reward_weight
        self.answer_length_weight = answer_length_weight
        self.answer_length_upper_bound = max(1, int(answer_length_upper_bound))
        self.api_timeout = api_timeout
        self.max_concurrent_api = max(1, max_concurrent_api)
        self._judge_base_urls = _parse_base_urls(
            os.getenv("RUBRIC_JUDGE_BASE_URLS") or os.getenv("OPENAI_BASE_URL"),
            ["http://localhost:8000/v1"],
        )
        self._clients = [
            OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=base_url)
            for base_url in self._judge_base_urls
        ]

    def _choose_client(self) -> OpenAI:
        return random.choice(self._clients)
        
    @staticmethod
    def _normalize_tool_tags(text: str) -> str:
        """
        Normalize our tool tags to what longform_rubric_rewards expects.
        - map <browse>/<scholar> to <search> so format/turn rewards pick them up
        """
        replacements = {
            "<browse>": "<search>",
            "</browse>": "</search>",
            "<scholar>": "<search>",
            "</scholar>": "</search>",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    @staticmethod
    def _extract_answer(response: str) -> tuple[str | None, str | None]:
        """Return (context_before_answer, answer_text)"""
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not match:
            return None, None
        start = match.start()
        context = response[:start].strip()
        answer = match.group(1).strip()
        return context, answer

    @staticmethod
    def _count_clean_thinks(text: str) -> int:
        think_block = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        bad_inside = re.compile(r"</?(think|answer|tool_call)\b")
        clean = 0
        for content in think_block.findall(text or ""):
            if not content or not content.strip():
                continue
            if bad_inside.search(content):
                continue
            clean += 1
        return clean

    @classmethod
    def _extract_query_like_calls(cls, text: str) -> List[str]:
        """Extract query-bearing tool calls from <tool_call> JSON blocks only."""
        if not text:
            return []

        queries: List[str] = []
        for raw_payload in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
            try:
                payload = json.loads(raw_payload.strip())
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            tool_name = payload.get("name")

            if not isinstance(tool_name, str):
                continue
            if tool_name not in cls.TOOL_CALL_NAMES:
                continue

            arguments = payload.get("arguments", {}) or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) or {}
                except Exception:
                    arguments = {"query": arguments}
            if not isinstance(arguments, dict):
                continue

            if tool_name in {"browse", "tool_browse"}:
                query = arguments.get("url") or arguments.get("query") or ""
            else:
                query = arguments.get("query") or arguments.get("q") or ""

            if isinstance(query, str):
                query = query.strip()
            else:
                query = str(query).strip()

            if query:
                queries.append(query)

        return queries
    # def _score_format_reward(self, response: str) -> float:
    #     """A looser format reward: answer/query/cite + think/query(+1) gap signal."""
    #     has_answer = bool(re.search(r"<answer>.*?</answer>", response or "", re.DOTALL))
    #     queries = re.findall(r"<(?:search|browse|scholar)>(.*?)</(?:search|browse|scholar)>", response or "", re.DOTALL)
    #     num_valid_queries = sum(1 for q in queries if q.strip())
    #     has_query = num_valid_queries > 0
    #     has_cite = bool(re.search(r"<cite id=[\"']?[^\"'>\s]+[\"']?[^>]*>[^<]+</cite>", response or "", re.DOTALL))

    #     num_thinks = self._count_clean_thinks(response or "")
    #     target_thinks = num_valid_queries + 1
    #     think_query_gap = abs(num_thinks - target_thinks)
    #     think_query_gap_reward = (1.0 / (1.0 + think_query_gap)) if has_answer else 0.0

    #     # Keep total reward in [0, 1].
    #     return (
    #         0.4 * float(has_answer)
    #         + 0.2 * float(has_cite)
    #         + 0.1 * float(has_query)
    #         + 0.3 * think_query_gap_reward
    #     )
    def _score_format_reward(self, response: str) -> float:
        """Format reward with explicit penalty from missing-think ratio."""
        has_answer = bool(re.search(r"<answer>.*?</answer>", response or "", re.DOTALL))
        queries = self._extract_query_like_calls(response or "")
        num_valid_queries = sum(1 for q in queries if q.strip())
        has_query = num_valid_queries > 0
        
        has_cite = bool(re.search(r"<cite id=[\"']?[^\"'>\s]+[\"']?[^>]*>[^<]+</cite>", response or "", re.DOTALL))

        num_thinks = self._count_clean_thinks(response or "")
        has_think = num_thinks > 1
        target_thinks = num_valid_queries + 1
        think_query_gap = abs(num_thinks - target_thinks)
        missing_think_ratio = max(float(target_thinks - num_thinks), 0.0) / float(max(target_thinks, 1))
        think_coverage = 1.0 - missing_think_ratio

        base_score = (
            0.5 * float(has_answer)
            + 0.2 * float(has_cite)
            + 0.1 * float(has_query)
            + 0.2 * float(has_think)
        )
        # Couple format score directly with how many required thinks are present.
        return base_score 
        # return base_score * think_coverage

        
    def _score_search_turns(self, context: str, upper_bound: int = 6) -> tuple[float, int]:
        if not context:
            return 0.0, 0
        queries = self._extract_query_like_calls(context)
        num_valid = sum(1 for q in queries if q.strip())

        # Robust think counting: a clean <think> only counts if the next non-space
        # content right after </think> is a <tool_call>.
        think_block = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        bad_inside = re.compile(r"</?(think|answer|tool_call)\b")
        next_query_pattern = re.compile(r"^\s*<tool_call>")
        num_thinks = 0
        for match in think_block.finditer(context):
            content = match.group(1)
            if not content or not content.strip():
                continue
            if bad_inside.search(content):
                continue
            tail = context[match.end():]
            if not next_query_pattern.search(tail):
                continue
            num_thinks += 1
        return min(float(num_valid) / upper_bound, 1.0) , num_valid

        # return (min(fZloat(num_valid) / upper_bound, 1.0) + min(float(num_thinks) / upper_bound, 1.0))/2, num_valid



    def _score_with_openai(self, question: str, answer: str, rubric_text: str) -> float:
#         system_prompt = """You will be given a question (in <question></question> tags) and an answer (in <response></response> tags).
# You will also be given a criterion (in <criterion></criterion> tags).

# The criterion can be of different types:
# - Factual criterion: evaluates whether the answer provides sufficient, correct, and relevant factual information. For factual criteria, judge whether the claims in the answer are supported by adequate and appropriate evidence. Missing, incomplete, weak, or unsupported evidence should lower the score.
# - Logical criterion: evaluates whether the answer is logically sound. For logical criteria, judge whether the reasoning is coherent, internally consistent, and whether the conclusions follow from the stated premises.

# Return ONLY a JSON object {"score": x} where x is an integer from 0 to 4 indicating how well the answer satisfies the criterion.

# Scoring guidelines:
# - 4: Fully satisfies the criterion. Evidence (for factual criteria) or reasoning (for logical criteria) is complete, correct, and clearly supports the conclusion.
# - 3: Largely satisfies the criterion, with minor omissions, ambiguities, or weak points that do not undermine the overall validity.
# - 2: Partially satisfies the criterion. Key evidence or reasoning steps are present but incomplete, underdeveloped, or only partially convincing.
# - 1: Minimally satisfies the criterion. Evidence or reasoning is mostly insufficient, flawed, or poorly connected to the conclusion.
# - 0: Fails to satisfy the criterion. Evidence is missing or incorrect, or the reasoning is invalid or incoherent.

# Judge only the specified criterion. Do not evaluate aspects that are not required by the criterion."""
        system_prompt = """You will be given a question (in <question></question> tags), an answer (in <response></response> tags), and a single criterion (in <criterion></criterion> tags).

Your job is to judge only how well the answer satisfies that specific criterion.

The criterion can be of different types:
- Factual criterion: judge whether the answer provides the specific, correct, and relevant factual content required by the criterion. Check whether the required facts, mechanisms, distinctions, conditions, or relationships are actually present and adequately supported in the answer. Generic background discussion or loosely related statements should not receive high scores.
- Logical criterion: judge whether the answer performs the specific reasoning required by the criterion. Check whether the answer actually makes the required comparison, distinction, synthesis, qualification, or conclusion. Surface-level coherence alone is not enough.

Important instructions:
- Do not reward an answer merely for mentioning related topics or keywords.
- Judge whether the answer directly addresses the specific requirement in the criterion.
- If the criterion contains multiple required components, high scores require covering all of the important components.
- If one or more essential components are missing, misstated, or only vaguely implied, do not give a high score.
- Generic, high-level, or background-only discussion should score low if it does not directly satisfy the criterion.

Return ONLY a JSON object {"score": x} where x is an integer from 0 to 4.

Scoring guidelines:
- 4: Fully satisfies the criterion. All essential components required by the criterion are explicitly, correctly, and sufficiently addressed.
- 3: Mostly satisfies the criterion. Most essential components are addressed, but there are minor omissions, ambiguities, or small weaknesses.
- 2: Partially satisfies the criterion. Some important components are present, but one or more essential parts are missing, underdeveloped, vague, or weakly supported.
- 1: Minimally satisfies the criterion. Only a small portion of the criterion is addressed, or the discussion is mostly generic, weak, flawed, or poorly connected to the required point.
- 0: Does not satisfy the criterion. The required content or reasoning is missing, incorrect, or irrelevant.

Judge only the specified criterion. Do not evaluate anything else."""
        user_prompt = f"<question>{question}</question>\n<response>{answer}</response>\n<criterion>{rubric_text}</criterion>"
        try:
            # resp = self.client.chat.completions.create(
            resp = self._choose_client().chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                timeout=self.api_timeout,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
            )
            content = resp.choices[0].message.content or ""
            score = 0.0
            try:
                obj = json.loads(content)
                if isinstance(obj, dict) and "score" in obj:
                    score = float(obj["score"])
            except Exception:
                # 2. fallback 正则（兼容单双引号）
                m = re.search(r'["\']score["\']\s*:\s*([0-9]+(?:\.[0-9]+)?)', content)
                if m:
                    score = float(m.group(1))
            return max(0.0, min(score / self.score_scale, 1.0))
        except Exception as e:
            print(f"[longform_rubric] scoring error: {e}")
            return 0.0

    def _score_likert(self, question: str, answer: str) -> float:
        system_prompt = """You are an expert evaluator. Given a user prompt and a generated response, please rate the overall quality of the response on a scale of 1 to 10, where 1 is very poor and 10 is excellent.
Return JSON: {"score": integer_between_1_and_10}."""
        user_prompt = f"<prompt>{question}</prompt>\n<response>{answer}</response>"
        try:
            # resp = self.client.chat.completions.create(
            resp = self._choose_client().chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                timeout=self.api_timeout,
                extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
            )
            content = resp.choices[0].message.content or ""
            m = re.search(r'"score"\s*:\s*([0-9]+)', content)
            score = float(m.group(1)) if m else 0.0
            return max(0.0, min(score / 10.0, 1.0))
        except Exception as e:
            print(f"[longform_rubric] likert scoring error: {e}")
            return 0.0

    def _compute_rubric_reward(self, answer: str, ground_truth: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
        rubric_scores: Dict[str, float] = {}

        question = ground_truth.get("query") or ground_truth.get("question", "")

        if self.use_likert_rubric:
            score = self._score_likert(question, answer)
            rubric_scores["likert"] = score
            return score, rubric_scores

        if self.use_general_rubric:
            rubric_text = """(1) Overall Comprehensiveness: The report should cover content as comprehensively as possible
(2) Thoroughness of Discussion: Each section should be discussed thoroughly, not just superficially
(3) Factuality: There should be minimal factual errors
(4) Coherence: The discussion should stay focused and relevant to the topic"""
            score = self._score_with_openai(question, answer, rubric_text)
            rubric_scores["general"] = score
            return score, rubric_scores

        rubrics: List[Dict[str, Any]] = ground_truth.get("rubrics") or []

        def _extract(r):
            desc = r.get("description","") 
            weight = r.get("weight", 0.0001)
            title = r.get("id","")  
            typep = r.get("type","")
            evidence =  r.get("evidence",[])
            lines = []
            lines.append(f"Criterion ID: {title}")
            lines.append(f"Criterion Type: {typep}")
            lines.append(f"Criterion Description: {desc}")

            if typep == "factual" and r.get("evidence"):
                lines.append("Supporting Evidence:")
                for i, ev in enumerate(evidence, 1):
                    lines.append(f"- Evidence {i}: {ev}")


            return "\n".join(lines), weight, title

        def build_rubric_text(rubric):
            lines = []
            lines.append(f"Criterion ID: {rubric['id']}")
            lines.append(f"Criterion Type: {rubric['type']}")
            lines.append(f"Criterion Description: {rubric['description']}")

            if rubric["type"] == "factual" and rubric.get("evidence"):
                lines.append("Supporting Evidence:")
                for i, ev in enumerate(rubric["evidence"], 1):
                    lines.append(f"- Evidence {i}: {ev}")

            return "\n".join(lines)

        total_weight = 0.0
        weighted_sum = 0.0
        for r in rubrics:
            desc, weight, title = _extract(r)
            score = self._score_with_openai(question, answer, desc)
            rubric_scores[title] = score
            weighted_sum += score * weight
            total_weight += abs(weight)

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        return overall, rubric_scores

    def _compute_reward_components(self, response: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom wrapper to align with our tag usage (<browse>/<scholar>/<search>, <answer>).
        Citations are not used; citation_reward set to 0.
        """
        from .utils.format_utils import extract_answer_context_citations, compute_format_reward
        # from .utils.citation_utils import score_in_context_citations
        from .utils.citation_utils_per_claim_with_nocite import score_in_context_citations

        extracted_context, extracted_answer, extracted_citations = extract_answer_context_citations(response)

        ctx, ans = self._extract_answer(response)
        num_search_turns_reward, num_search_turns = self._score_search_turns(ctx)
        format_reward = self._score_format_reward(response)

        answer_length = 0.0
        if ans is not None:
            try:
                answer_length = float(len(self.tokenizer.encode(ans, add_special_tokens=False)))
            except Exception:
                answer_length = float(len(re.findall(r"\S+", ans)))

        answer_length_reward = min(answer_length / float(self.answer_length_upper_bound), 1.0)

        result = {
            "reward": 0.0,
            "log_values": {
                "format_reward": format_reward,
                "num_search_turns_reward": num_search_turns_reward,
                "num_search_turns": num_search_turns,
                "citation_reward": 0.0,
                "answer_length": answer_length,
                "answer_length_reward": answer_length_reward,
                "format_correct_has_answer": 0.0,
                "rubric_reward": 0.0,
                "rubric_scores_by_title": {},
            },
            "error": None,
        }

        if ans is None:
            result["error"] = "Failed to extract answer with <answer></answer> tags"
            # reward_val = self.format_weight * format_reward + self.search_turn_weight * num_search_turns_reward
            reward_val = self.format_weight * format_reward
            # reward_val += self.answer_length_weight * answer_length_reward
            result["reward"] = reward_val
            return result
        # if  format_reward <= 0.0:
        #     result["error"] = "Format invalid: hard-gated reward to 0"
        #     result["reward"] = 0.0
        #     return result

        rubric_reward, rubric_scores = self._compute_rubric_reward(ans, ground_truth)
        question = ground_truth.get("query") or ground_truth.get("question", "")

        # citation_reward = score_in_context_citations(question, response, extracted_citations)
        citation_reward = score_in_context_citations(question, extracted_answer, extracted_citations)

        # citation_reward = 0

        result["log_values"].update(
            {
                "rubric_reward": rubric_reward,
                "rubric_scores_by_title": rubric_scores,
                "citation_reward": citation_reward,
                "format_correct_has_answer": 1.0,
            }
        )
        

        reward_val = self.rubric_weight * rubric_reward 
        reward_val += self.format_weight * format_reward
        reward_val += self.search_turn_weight * num_search_turns_reward

        reward_val += self.citation_reward_weight * citation_reward
        reward_val += self.answer_length_weight * answer_length_reward
        result["reward"] = reward_val
        return result

    def __call__(self, data: DataProto, return_dict: bool = False):
        # Reuse pre-computed rollout rewards when agent loop has already
        # populated token-level scores into `rm_scores`.
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]


            
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: Dict[str, list] = defaultdict(list)

        already_print_data_sources: Dict[str, int] = {}

        for i in range(len(data)):
            data_item = data[i]

            # Decode prompt/response
            prompt_ids = data_item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = data_item.batch["attention_mask"][:prompt_len].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = data_item.batch["responses"]
            valid_resp_len = data_item.batch["attention_mask"][prompt_len:].sum()
            valid_resp_ids = response_ids[:valid_resp_len]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)
            # response_str = self._normalize_tool_tags(response_str)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch.get("data_source", "unknown")

            reward_dict = self._compute_reward_components(response_str, ground_truth)

            reward_value = float(reward_dict.get("reward", 0.0))
            log_values = reward_dict.get("log_values", {})

            reward_tensor[i, valid_resp_len - 1] = reward_value

            # Collect extra info
            for k, v in log_values.items():
                # only keep numeric log values to avoid downstream mean() on dicts
                if isinstance(v, (int, float)):
                    reward_extra_info[k].append(v)
            if "error" in reward_dict:
                reward_extra_info["error"].append(reward_dict["error"])

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[reward]", reward_value)
                if log_values:
                    print("[log_values]", log_values)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor
