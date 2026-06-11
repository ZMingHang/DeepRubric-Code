import json
import logging
import os
import threading
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import requests
import regex as re

from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 50


import base64
import hashlib
import html
from typing import List, Dict


# ---------- 1. 稳定哈希 ID ----------
def stable_snippet_id(text: str, prefix: str = "S_", length: int = 7) -> str:
    """
    Deterministic snippet id based on content.
    Same text -> same ID
    """
    normalized = " ".join(text.split())  # 去多余空白，避免格式差异导致不同hash
    h = hashlib.blake2s(normalized.encode("utf-8"), digest_size=16).digest()
    b64 = base64.urlsafe_b64encode(h).decode("ascii").rstrip("=")
    return f"{prefix}{b64[:length]}"


# ---------- 2. 将搜索结果转为 snippet ----------
def build_snippets(search_results: List[Dict]) -> Dict:
    seen = set()
    snippets_xml = []
    id_map = {}

    for r in search_results:
        text = r.strip()
        if not text:
            continue

        sid = stable_snippet_id(text)

        if sid in seen:
            continue
        seen.add(sid)

        id_map[sid] = r

        # escape 防止破坏 XML
        safe_text = html.escape(text[:800])  # 截断避免上下文爆炸

        snippets_xml.append(
            f'<webpage id="{sid}">{safe_text}\n</webpage>'
        )

    return snippets_xml


# def _extract_content(api_resp: Dict[str, Any], url: str) -> str:
#     """Format browse API response to a short string."""
#     result = api_resp.get("result", [])
#     if not result:
#         return f"Failed to read page {url}."
#     contents = result[0].get("contents", "")
#     if not contents:
#         return f"Failed to read page {url}."
#     return contents

def _extract_content(api_resp: Dict[str, Any], url: str) -> str:
    """Format browse API response to a short string."""
    result = api_resp.get("result", [])
    if not result:
        return f"Failed to read page {url}."
    contents = result[0].get("contents", "")
    if not contents:
        return f"Failed to read page {url}."
    return build_snippets([contents])


@register_tool
class BrowseTool(BaseTool):
    """
    Simple browsing tool backed by a retrieval endpoint.
    Action format:
      - <browse>https://example.com</browse>
      - ```browse
        https://example.com
        ```
      - browse: https://example.com
    """

    tool_type = "browse"

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
        return "Use <browse>url</browse> or ```browse\\nurl\\n``` to fetch page content."

    def parse_action(self, action: str) -> Tuple[str, bool]:
        patterns = [
            r"<browse>(.*?)</browse>",
            r"```\s*browse\s*\n(.*?)\n```",
            r"browse:\s*(.*?)(?:\n|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, action, re.DOTALL | re.IGNORECASE)
            if matches:
                url = matches[0].strip()
                if url:
                    return url, True
        return "", False

    def _call_api(self, url: str, topk: int, timeout: int) -> Tuple[str, bool]:
        payload = {"urls": [url], "topk": topk, "return_scores": True}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        try:
            resp = self.session.post(
                self.browse_url, json=payload, headers=headers, timeout=timeout
            )
            resp.raise_for_status()
            api_resp: Dict[str, Any] = resp.json()
            if self.log_requests:
                logger.info("Browse API call ok: %s", self.browse_url)
            return _extract_content(api_resp, url), True
        except Exception as e:
            logger.error("Browse API error: %s", e)
            return f"Visit error: {e}", False

    def conduct_action(self, trajectory_id, action, extra_field):
        parsed_url, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            observation = "Invalid browse action. Use <browse>url</browse>."
            done = False
            valid = False
        else:
            topk = extra_field.get("topk", self.topk) if isinstance(extra_field, dict) else self.topk
            timeout = extra_field.get("timeout", self.timeout) if isinstance(extra_field, dict) else self.timeout
            result, success = self._call_api(parsed_url, topk, timeout)
            observation = f"\n<tool_response>\n{result}\n</tool_response>"
            done = False
            valid = success

        self.update_env(trajectory_id, env, parsed_url, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)

        return observation, done, valid
