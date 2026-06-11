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


def _format_scholar_results(api_resp: Dict[str, Any], query: str) -> str:
    """Convert scholar API response to a printable string."""
    snippets = []
    for item in api_resp.get("result", [[]])[0]:
        doc = item.get("document", {})
        contents = doc.get("contents", "")
        snippets.append(contents.split("# Sources.")[0])

    if not snippets:
        return f"No results found for '{query}'."

    text = "\n\n".join(snippets)
    return f"A Google scholar for '{query}' found {len(snippets)} results:\n\n{text}"


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

    tool_type = "scholar_lite"

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
        self.scholar_url = scholar_url or os.getenv("SCHOLAR_LITE_URL", "http://localhost:8001/retrieve")
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
        payload = {"queries": [query], "topk": topk, "return_scores": True}
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
            observation = f"<result>{result}</result>"
            done = False
            valid = success

        self.update_env(trajectory_id, env, parsed_query, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)

        return observation, done, valid
