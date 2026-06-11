import json
import logging
import os
import threading
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import regex as re
import requests

from .base import BaseTool, register_tool
from .browse import build_snippets

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 50


@register_tool
class ToolBrowseMultiTool(BaseTool):
    """Browse tool that accepts <tool_call> JSON with string or list url input."""

    tool_type = "tool_browse_multi"

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
        browse_url: str | None = None,
        topk: int = 3,
        timeout: int = DEFAULT_TIMEOUT,
        log_requests: bool = True,
    ):
        super().__init__(num_workers=num_workers)
        self.browse_url = browse_url or os.getenv("BROWSE_URL", "http://localhost:8888/access")
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests

        parsed_url = urlparse(self.browse_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.session = self._get_shared_session(self.base_url)

    def get_usage_inst(self) -> str:
        return (
            'Use <tool_call>{"name":"browse_multi","arguments":{"url":["https://a.com","https://b.com"],"topk":3}}</tool_call> '
            "to fetch one or more pages."
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

        if payload.get("name") not in {"browse_multi", self.tool_type}:
            return {}, False

        arguments = payload.get("arguments", {}) or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) or {}
            except Exception:
                arguments = {"url": arguments}
        if not isinstance(arguments, dict):
            return {}, False

        urls = self._normalize_str_or_list(arguments.get("url"))
        if not urls:
            return {}, False

        return {
            "urls": urls,
            "topk": arguments.get("topk"),
            "timeout": arguments.get("timeout"),
        }, True

    @staticmethod
    def _extract_contents(api_resp: Dict[str, Any], urls: List[str]) -> str:
        result = api_resp.get("result", [])
        blocks: List[str] = []
        for idx, url in enumerate(urls):
            contents = ""
            if idx < len(result):
                item = result[idx]
                if isinstance(item, dict):
                    contents = item.get("contents", "") or ""
            formatted = "\n".join(build_snippets([contents])) if contents else f"Failed to read page {url}."
            blocks.append(
                f'<browse_block index="{idx + 1}" url="{url}">\n{formatted}\n</browse_block>'
            )
        return "\n\n".join(blocks)

    def _call_api(self, urls: List[str], topk: int, timeout: int) -> Tuple[str, bool]:
        payload = {"urls": urls, "topk": topk, "return_scores": True}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        try:
            resp = self.session.post(
                self.browse_url, json=payload, headers=headers, timeout=timeout
            )
            resp.raise_for_status()
            api_resp: Dict[str, Any] = resp.json()
            if self.log_requests:
                logger.info("Browse API call ok: %s", self.browse_url)
            return self._extract_contents(api_resp, urls), True
        except Exception as e:
            logger.error("Browse API error: %s", e)
            return f"Visit error: {e}", False

    def conduct_action(self, trajectory_id, action, extra_field):
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            observation = (
                'Invalid browse action. Use '
                '<tool_call>{"name":"browse_multi","arguments":{"url":["https://example.com"]}}</tool_call>.'
            )
            done = False
            valid = False
            parsed_urls: List[str] = []
        else:
            parsed_urls = parsed_action["urls"]
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

            result, success = self._call_api(parsed_urls, topk, timeout)
            observation = f"\n<tool_response>\n{result}\n</tool_response>"
            done = False
            valid = success

        self.update_env(trajectory_id, env, parsed_urls, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)

        return observation, done, valid
