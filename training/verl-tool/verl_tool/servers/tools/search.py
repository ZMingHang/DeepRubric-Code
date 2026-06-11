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
        safe_text = html.escape(text)  # 截断避免上下文爆炸

        snippets_xml.append(
            f'<snippet id="{sid}">{safe_text}\n</snippet>'
        )

    return snippets_xml


logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1




# def _passages_to_string(retrieval_result: List[Dict[str, Any]]) -> str:
#     """Convert retrieval results to a readable string."""
#     formatted: List[str] = []
#     for idx, doc_item in enumerate(retrieval_result):
#         contents = doc_item.get("document", {}).get("contents", "")
#         # contents = contents.split("# Sources.")[0]
#         formatted.append(f"Doc {idx}: {contents}")
#     return "\n\n".join(formatted).strip()


def _passages_to_string(retrieval_result: List[Dict[str, Any]]) -> str:
    """Convert retrieval results to a readable string."""
    formatted: List[str] = []
    for idx, doc_item in enumerate(retrieval_result):
        contents = doc_item.get("document", {}).get("contents", "")
        url = doc_item.get("document", {}).get("url", "")
        contents += (f"\n# Sources. \n{url}")
        formatted.append(f"{contents}")
    
    return "\n\n".join(build_snippets(formatted)).strip()


@register_tool
class SearchTool(BaseTool):
    """
    Retrieval search tool.
    Action format:
      - <search>your query</search>
      - ```search
        your query
        ```
      - search: your query
    """

    tool_type = "search"

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
        return "Use <search>query</search> or ```search\\nquery\\n``` to run retrieval."

    def parse_action(self, action: str) -> Tuple[str, bool]:
        patterns = [
            r"<search>(.*?)</search>",
            r"```\s*search\s*\n(.*?)\n```",
            r"search:\s*(.*?)(?:\n|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, action, re.DOTALL | re.IGNORECASE)
            if matches:
                query = matches[0].strip()
                if query:
                    return query, True
        return "", False

    def _call_api(self, query: str, topk: int, timeout: int) -> Tuple[str, bool]:
        payload = {"queries": [query], "topk": topk, "return_scores": True}
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

                pretty_results = []
                for retrieval in raw_results:
                    formatted = _passages_to_string(retrieval)
                    if formatted:
                        pretty_results.append(formatted)
                final_result = "\n---\n".join(pretty_results) if pretty_results else "No search results found."
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
        parsed_query, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            observation = "Invalid search action. Use <search>query</search>."
            done = False
            valid = False
        else:
            topk = extra_field.get("topk", self.topk) if isinstance(extra_field, dict) else self.topk
            timeout = extra_field.get("timeout", self.timeout) if isinstance(extra_field, dict) else self.timeout
            result, success = self._call_api(parsed_query, topk, timeout)
            observation = f"\n<tool_response>\n{result}\n</tool_response>"
            done = False
            valid = success

        self.update_env(trajectory_id, env, parsed_query, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)

        return observation, done, valid
