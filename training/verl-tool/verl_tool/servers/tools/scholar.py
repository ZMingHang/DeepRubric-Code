import json
import logging
import os
import threading
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import regex as re
import requests

from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
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


# def _format_scholar_results(api_resp: Dict[str, Any], query: str) -> str:
#     """Convert scholar API response to a printable string."""
#     # snippets = []
#     # print(api_resp)
#     try:
#         snippets = api_resp.get("results", [[]])['passages'][0]
#     except:
#         snippets = []

#     if not snippets:
#         return f"No results found for '{query}'."

#     text = "\n\n".join(snippets)
#     return f"A Google scholar for '{query}' found {len(snippets)} results:\n\n{text}"

def _format_scholar_results(api_resp: Dict[str, Any], query: str) -> str:
    """Convert scholar API response to a printable string."""
    # snippets = []
    # print(api_resp)
    try:
        snippets = api_resp.get("results", [[]])['passages'][0]
    except:
        snippets = []

    if not snippets:
        return f"No results found for '{query}'."

    text = "\n\n".join(build_snippets(snippets))
    return f"{text}"

@register_tool
class ScholarTool(BaseTool):
    """
    Google Scholar retrieval tool.
    Action format:
      - <scholar>your query</scholar>
      - ```scholar
        your query
        ```
      - scholar: your query
    """

    tool_type = "scholar"

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
        return "Use <scholar>query</scholar> or ```scholar\\nquery\\n``` to search Google Scholar."

    def parse_action(self, action: str) -> Tuple[str, bool]:
        patterns = [
            r"<scholar>(.*?)</scholar>",
            r"```\s*scholar\s*\n(.*?)\n```",
            r"scholar:\s*(.*?)(?:\n|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, action, re.DOTALL | re.IGNORECASE)
            if matches:
                query = matches[0].strip()
                if query:
                    return query, True
        return "", False

    def _call_api(self, query: str, topk: int, timeout: int) -> Tuple[str, bool]:
        # payload = {"queries": [query], "topk": topk, "return_scores": True}
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
        parsed_query, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            observation = "Invalid scholar action. Use <scholar>query</scholar>."
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
