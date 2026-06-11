import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import regex as re
import requests

from .base import BaseTool, register_tool
from .search import _passages_to_string

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1


@register_tool
class ToolSearchMultiTool(BaseTool):
    """Search tool that accepts <tool_call> JSON with string or list query input."""

    tool_type = "tool_search_multi"

    _session_pool: Dict[str, requests.Session] = {}
    _session_lock = threading.Lock()

    @classmethod
    def _get_shared_session(cls, base_url: str) -> requests.Session:
        with cls._session_lock:
            if base_url not in cls._session_pool:
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=512,
                    pool_maxsize=512,
                    max_retries=0,
                    pool_block=False,
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                cls._session_pool[base_url] = session
            return cls._session_pool[base_url]

    def __init__(
        self,
        num_workers: int = 1,
        search_url: Optional[str] = None,
        topk: int = 3,
        timeout: int = DEFAULT_TIMEOUT,
        log_requests: bool = True,
    ):
        super().__init__(num_workers=num_workers)
        self.search_url = search_url or os.getenv("SEARCH_URL", "http://localhost:8888/retrieve")
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests

        parsed_url = urlparse(self.search_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.session = self._get_shared_session(self.base_url)

    def get_usage_inst(self) -> str:
        return (
            'Use <tool_call>{"name":"search","arguments":{"query":["q1","q2"],"topk":3}}</tool_call> '
            "to run retrieval for one or more queries."
        )

    @staticmethod
    def _normalize_str_or_list(value: Any) -> List[str]:
        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []
        if isinstance(value, list):
            results = []
            for item in value:
                if isinstance(item, str):
                    item = item.strip()
                else:
                    item = str(item).strip()
                if item:
                    results.append(item)
            return results
        return []

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        if not isinstance(action, str):
            return {}, False

        match = re.search(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)
        if not match:
            return {}, False

        try:
            payload = json.loads(match.group(1).strip())
        except Exception:
            return {}, False

        if not isinstance(payload, dict):
            return {}, False

        if payload.get("name") not in {"search", self.tool_type}:
            return {}, False

        arguments = payload.get("arguments", {}) or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) or {}
            except Exception:
                arguments = {"query": arguments}
        if not isinstance(arguments, dict):
            return {}, False

        queries = self._normalize_str_or_list(arguments.get("query") or arguments.get("q"))
        if not queries:
            return {}, False

        return {
            "queries": queries,
            "topk": arguments.get("topk"),
            "timeout": arguments.get("timeout"),
        }, True

    def _call_api(self, queries: List[str], topk: int, timeout: int) -> Tuple[str, bool]:
        payload = {"queries": queries, "topk": topk, "return_scores": True}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        last_error: Optional[str] = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.post(
                    self.search_url, json=payload, headers=headers, timeout=timeout
                )
                if resp.status_code in {500, 502, 503, 504} and attempt < MAX_RETRIES - 1:
                    last_error = f"Server error {resp.status_code}"
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                api_resp: Dict[str, Any] = resp.json()
                if self.log_requests:
                    logger.info("Search API call ok: %s", self.search_url)
                raw_results = api_resp.get("result", [])
                if not raw_results:
                    return "No search results found.", True

                blocks: List[str] = []
                for idx, query in enumerate(queries):
                    retrieval = raw_results[idx] if idx < len(raw_results) else []
                    formatted = _passages_to_string(retrieval) if retrieval else "No search results found."
                    blocks.append(
                        f'<query_block index="{idx + 1}" query="{query}">\n{formatted}\n</query_block>'
                    )
                final_result = "\n\n".join(blocks) if blocks else "No search results found."
                return final_result, True
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    time.sleep(delay)
                    continue
                logger.error("Search API error: %s", e)

        return f"Search error: {last_error}", False

    def conduct_action(self, trajectory_id, action, extra_field):
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            observation = (
                'Invalid search action. Use '
                '<tool_call>{"name":"search","arguments":{"query":["..."]}}</tool_call>.'
            )
            done = False
            valid = False
            parsed_queries: List[str] = []
        else:
            parsed_queries = parsed_action["queries"]
            topk = parsed_action.get("topk")
            if topk is None:
                topk = extra_field.get("topk", self.topk) if isinstance(extra_field, dict) else self.topk
            timeout = parsed_action.get("timeout")
            if timeout is None:
                timeout = extra_field.get("timeout", self.timeout) if isinstance(extra_field, dict) else self.timeout

            try:
                topk = max(1, int(topk))
            except Exception:
                topk = self.topk
            try:
                timeout = max(1, int(timeout))
            except Exception:
                timeout = self.timeout

            result, success = self._call_api(parsed_queries, topk, timeout)
            observation = f"\n<tool_response>\n{result}\n</tool_response>"
            done = False
            valid = success

        self.update_env(trajectory_id, env, parsed_queries, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)

        return observation, done, valid
