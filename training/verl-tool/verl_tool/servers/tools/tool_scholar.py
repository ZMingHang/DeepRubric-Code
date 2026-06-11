import json
import logging
import os
import threading
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import regex as re
import requests

from .base import BaseTool, register_tool
from .scholar import _format_scholar_results

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


@register_tool
class ToolScholarTool(BaseTool):
    """Scholar tool that only accepts <tool_call> JSON actions."""

    tool_type = "tool_scholar"

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
        scholar_url: str | None = None,
        topk: int = 3,
        timeout: int = DEFAULT_TIMEOUT,
        log_requests: bool = True,
    ):
        super().__init__(num_workers=num_workers)
        self.scholar_url = scholar_url or os.getenv("SCHOLAR_URL", "http://localhost:8000/search")
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests

        parsed_url = urlparse(self.scholar_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.session = self._get_shared_session(self.base_url)

    def get_usage_inst(self) -> str:
        return (
            'Use <tool_call>{"name":"scholar","arguments":{"query":"your query","topk":3}}</tool_call> '
            "to search Google Scholar."
        )

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

        if payload.get("name") not in {"scholar", self.tool_type}:
            return {}, False

        arguments = payload.get("arguments", {}) or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) or {}
            except Exception:
                arguments = {"query": arguments}
        if not isinstance(arguments, dict):
            return {}, False

        query = (arguments.get("query") or arguments.get("q") or "").strip()
        if not query:
            return {}, False

        return {
            "query": query,
            "topk": arguments.get("topk", arguments.get("n_docs")),
            "timeout": arguments.get("timeout"),
        }, True

    def _call_api(self, query: str, topk: int, timeout: int) -> Tuple[str, bool]:
        payload = dict(query=[query], n_docs=topk, domains="pes2o_v3")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        try:
            resp = self.session.post(
                self.scholar_url, json=payload, headers=headers, timeout=timeout
            )
            resp.raise_for_status()
            api_resp: Dict[str, Any] = resp.json()
            if self.log_requests:
                logger.info("Scholar API call ok: %s", self.scholar_url)
            return _format_scholar_results(api_resp, query), True
        except Exception as e:
            logger.error("Scholar API error: %s", e)
            return f"Search error: {e}", False

    def conduct_action(self, trajectory_id, action, extra_field):
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            observation = (
                'Invalid scholar action. Use '
                '<tool_call>{"name":"scholar","arguments":{"query":"..."}}</tool_call>.'
            )
            done = False
            valid = False
            parsed_query = ""
        else:
            parsed_query = parsed_action["query"]
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

            result, success = self._call_api(parsed_query, topk, timeout)
            observation = f"\n<tool_response>\n{result}\n</tool_response>"
            done = False
            valid = success

        self.update_env(trajectory_id, env, parsed_query, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)

        return observation, done, valid
