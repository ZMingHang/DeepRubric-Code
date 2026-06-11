"""
Recursive deep-research question generator.

This reimplements the existing data-generation pipeline as a recursive
search tree:
1) Start from a root query (wiki entity name).
2) Recursively expand child queries until depth/novelty/LLM-stop.
3) Aggregate all extracted statements as evidence.
4) Generate a long-form research question + rubrics from the evidence.

Defaults are conservative (depth=3, branching<=3, nodes<=50) and largely
reuse the prompt templates from the original implementation.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from qa_synthesis_agent_openai_v9 import (
    ConstructQAPrompts,
    OpenAIAPIClient,
)
import asyncio, aiohttp


class RetrieveSearchClient:
    """Minimal async client for the /retrieve endpoint."""
    def __init__(self, base_url: str = "http://localhost:8000/search", timeout: int = 20):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def retrieve(self, queries, topk=5, return_scores=True):
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            payload = dict(query=queries, n_docs=topk, domains="pes2o_v3")
            async with session.post(self.base_url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("results", []).get('passages',[])
    async def doc_retrieve(self, queries, topk=3, return_scores=True):
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            payload = dict(query=queries, topk=topk, return_scores=return_scores)
            async with session.post(self.base_url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                content = ""
                for index, r in enumerate(data.get("result", [])[0]):
                    content += f"Doc {index}:{r['document']['contents'].split('# Sources.')[0]} \n\n"
                
                return content

BASE_QA_TREE_PROMPT = '''You are an autonomous agent for proposing **deep-research questions** from a search tree.

Input is a compact JSON representation of a search tree with:
- `nodes`: id, parent, depth, query, children
- `leaves`: leaf id, path (root->leaf queries), and evidence statements for that leaf

Your task is to define a **complete, well-formed research problem** that ultimately requires a **long-form, structured answer**.
You are expected to make the question self-contained and answerable in scope (but you are NOT expected to solve it).

Important: The tree may contain **noise** or even **harmful/irrelevant information** that is not related to the root topic.
You must identify and avoid such nodes/claims when forming the question and rubrics. If a leaf is misleading or harmful but must be addressed, reflect this with **negative-weight rubrics** that require debunking or exclusion.

You must:
1) Select a **small set of leaf nodes** that best cover distinct facets of the root topic.
2) Perform a **bottom-up merge**: combine selected leaf queries into intermediate queries, then merge those into the final research question.
3) Create rubrics **grounded only in statements from the selected leaves**.
4) Provide a **visualizable reasoning trace**: which leaves were selected, why, and how they were merged step-by-step.

### Bottom-up merge requirements
- Start from selected leaves (last-layer nodes). Group them into **multiple** intermediate queries.
- Then merge intermediate queries **iteratively** into higher-level queries until you reach one final question.
- The merge can take **many steps**; This is a reasoning process; include it in `merge_trace` with **explicit inputs, outputs, and short rationale**.  Each merge step should explain why those inputs were grouped and what new query they form.

### Rubric Types
Each rubric MUST belong to exactly one of the following categories:
- **Logical**: structure/organization/reasoning checks for a long-form answer.
- **Factual**: specific facts/claims that must be addressed, each **supported by evidence from selected leaves only**.

### Weighting
- Assign each rubric an importance score between **0 and 1**
- Higher weight means the rubric is more critical to answering the question correctly
- Negative weight means the rubric captures **misleading/irrelevant/harmful** content that should be explicitly rejected, debunked, or excluded from the answer
- Weights do not need to sum to 1

### Rubric quality (applies to every rubric)
- Make each rubric precise, atomic, and binary-checkable; avoid vague verbs.
- Anchor factual rubrics to concrete supporting statements from the selected leaf evidence; do not invent new facts.
- Cover evidence breadth & attribution, reasoning soundness, completeness of expected findings, organization/clarity, and handling of ambiguity or counterpoints.
- Keep rubrics concise and avoid duplicate checks.
- Each rubric item must be evaluated independently and must not depend on the satisfaction or failure of any other rubric item.
- Every rubric MUST include `source_leaf_ids` and all ids must be a subset of `selected_leaf_ids`.


### Question Construction
- The question should be motivated by the selected rubrics and evidence.
- It should expose meaningful gaps, unresolved relationships, or implicit constraints.
- It must NOT be fully answerable using the provided material alone.
- Do NOT explicitly list rubrics or solution steps in the question.
- Prefer questions that are brief, precise, and research-heavy.
- Do NOT invent facts beyond the material.
- You MUST generate a minimum of 5 rubrics in the output.
- The rubrics should be numbered sequentially (R1, R2, R3, R4, R5...)

# Search Tree JSON
{content}

Return JSON ONLY like:
```json
{{
  "question": "a research-worthy deep-research question",
  "selected_leaf_ids": ["N12", "N19"],
  "selection_trace": [
    {{"leaf_id": "N12", "reason": "covers mechanism/causal pathway"}},
    {{"leaf_id": "N19", "reason": "covers policy/regulatory constraints"}}
  ],
  "merge_trace": [
    {{"level": "leaf", "selected": ["N12", "N19"], "rationale": "diverse facets"}},
    {{"level": "merge_1", "inputs": ["N12", "N19"], "output_query": "intermediate query", "rationale": "shared causal chain"}},
    {{"level": "final", "inputs": ["Q1", "Q2"], "output_query": "final question", "rationale": "unifies mechanisms + constraints"}}
  ],
  "rubrics": [
    {{
      "id": "R1",
      "type": "logical",
      "description": "logical or structural rubric of the answer",
      "weight": 0.8,
      "source_leaf_ids": ["N12,N11"]
    }},
    {{
      "id": "R2",
      "type": "factual",
      "description": "fact-based rubric that must be addressed",
      "weight": 0.6,
      "source_leaf_ids": ["N19"],
      "evidence": [
        "supporting statement from selected leaf evidence",
        "another supporting statement"
      ]
    }}
  ]
}}
```
'''


# ---------- Utility ----------
def normalize_url(url: str) -> str:
    if "index.php/" in url:
        url = url.replace("index.php/", "index.php?title=")
    if "/wiki/" in url:
        url = url.replace("/wiki/", "/w/index.php?title=")
    if "_" in url:
        url = url.replace("_", "%20")
    return url


# ---------- Data classes ----------
@dataclass
class SearchNode:
    query: str
    depth: int
    statements: List[str] = field(default_factory=list)
    children: List["SearchNode"] = field(default_factory=list)
    stop: bool = False
    stop_reason: Optional[str] = None


@dataclass
class TreeResult:
    root: SearchNode
    statements: List[str]


# ---------- Agent ----------
class RecursiveQAAgent:
    def __init__(
        self,
        pages: Dict[str, Any],
        search_client: RetrieveSearchClient,
        max_depth: int = 3,
        max_children: int = 3,
        root_children: int = 6,
        mid_children: int = 4,
        topk_per_query: int = 5,
        max_nodes: int = 50,
        semaphore: Optional[asyncio.Semaphore] = None,  # kept for API compatibility, not used
    ):
        self.pages = pages
        self.search_client = search_client
        self.max_depth = max_depth
        self.max_children = max_children
        self.root_children = root_children
        self.mid_children = mid_children
        self.topk_per_query = topk_per_query
        self.max_nodes = max_nodes
        self.node_count = 0
  

    def max_children_for_depth(self, depth: int) -> int:
        if depth == 0:
            return self.root_children
        if depth == 1:
            return self.mid_children
        return self.max_children

    # ---- LLM helpers ----
    async def call_llm(self, prompt: str, client: OpenAIAPIClient) -> str:
        print(prompt)
        sampling_kwargs = dict(temperature=0.4, top_p=0.95, max_new_tokens=4096)
        out = await client.async_generate(prompt, sampling_kwargs)
        return out["text"]

    # ---- Retrieval + extraction ----
    async def retrieve_snippets(self, queries: List[str], topk: Optional[int] = None) -> List[str]:
        if self.search_client is None:
            return []
        topk = topk or self.topk_per_query
        try:
            results = await self.search_client.retrieve(queries, topk=topk, return_scores=True)
        except Exception as e:
            print(f"[WARN] retrieve failed: {e}")
            return []
        flat = []
        for entry in results:
            flat.extend(entry if isinstance(entry, list) else [entry])
        snippets = []
        for doc in flat:
            # document = doc.get("document", doc)
            # content = (
            #     document.get("contents")
            #     or document.get("text")
            #     or document.get("content")
            #     or document.get("passage")
            #     or ""
            # )
            content = doc
            if content:
                snippets.append(content[:5000])
        return snippets

    async def extract_statements(self, queries: List[str], note: str, snippets: List[str], client) -> List[str]:
        """LLM-based evidence extraction with light post-filtering/dedup."""
        if not snippets:
            return []
        prompt = ConstructQAPrompts.extract_relevant_for_note.format(
            queries="\n".join(queries),
            note=note,
            summary="\n".join(snippets[:10]),  # cap prompt size
        )
        text = await self.call_llm(prompt, client)
        try:
            parsed = json.loads(text.split("```json")[-1].split("```")[0].strip())
            if isinstance(parsed.get("statements"), list):
                stmts = [s.strip() for s in parsed["statements"] if isinstance(s, str) and s.strip()]
                # simple dedup preserving order
                seen = set()
                dedup = []
                for s in stmts:
                    if s not in seen:
                        dedup.append(s)
                        seen.add(s)
                return dedup
        except Exception as e:
            print(f"[WARN] extract_statements parse failed: {e}")
        return []

    async def propose_child_queries(
        self,
        parent_query: str,
        snippets: List[str],
        depth: int,
        statements: List[str],
        client: OpenAIAPIClient,
    ) -> List[str]:
        # Prompt emphasizes同层=并行角度(宽度)，下一层再深挖(深度)，且要求全面覆盖 research 维度
        prompt = f"""You are expanding a research search tree.
Parent query (to be deepened later): "{parent_query}"
Current depth: {depth}, max depth: {self.max_depth}. Remaining depth will be used to DRILL each child later.

Current evidence (snippets + extracted statements):
{chr(10).join(snippets)}
Statements (sample, deduped top 8):
{chr(10).join(statements[:8])}

Goal: maximize coverage AND depth potential for a future deep-research question. Propose up to {self.max_children} sibling queries that:
- Each covers a DIFFERENT parallel facet of the parent (breadth): stakeholders, timeline, mechanisms/causality, outcomes/impacts, risks/failures, alternatives/counterfactuals, quantitative data/metrics, geography, policy/regulation, controversy/debate.
- Ensure at least one query targets comparison/alternatives, one targets risks/failures/downsides, and one targets quantitative evidence or data sources (add them if missing).
- Keep at the SAME abstraction level as the parent (do not prematurely narrow); deeper drilling will happen in the next recursion.
- Avoid overlap; prioritize facets not yet evidenced by the snippets.
- Keep queries concise (<12 tokens), factual, and non-redundant.

Return JSON:
{{
  "queries": ["q1", "q2", "q3"]
}}"""
        text = await self.call_llm(prompt, client)
        try:
            parsed = json.loads(text.split("```json")[-1].split("```")[0].strip())
            # print(parsed)
            if isinstance(parsed.get("queries"), list):
                return [q for q in parsed["queries"] if isinstance(q, str) and q.strip()][: self.max_children]
        except Exception as e:
            print(f"[WARN] propose_child_queries parse failed: {e}")
        return []

    async def propose_child_queries_v2(
        self,
        parent_query: str,
        snippets: List[str],
        depth: int,
        statements: List[str],
        client: OpenAIAPIClient,
    ) -> List[str]:
        """Depth-aware child query generator (replaces propose_child_queries)."""
        limit = self.max_children_for_depth(depth)
        prompt = f"""You are expanding a research search tree.
Parent query (to be deepened later): "{parent_query}"
Current depth: {depth}, max depth: {self.max_depth}. Remaining depth will be used to DRILL each child later.

Current evidence (snippets + extracted statements):
{chr(10).join(snippets)}
Statements (sample, deduped top 8):
{chr(10).join(statements[:8])}

Goal: maximize coverage AND depth potential for a future deep-research question. Propose up to {limit} sibling queries that:
- Each covers a DIFFERENT parallel facet of the parent (breadth): stakeholders, timeline, mechanisms/causality, outcomes/impacts, risks/failures, alternatives/counterfactuals, quantitative data/metrics, geography, policy/regulation, controversy/debate.
- Ensure at least one query targets comparison/alternatives, one targets risks/failures/downsides, and one targets quantitative evidence or data sources (add them if missing).
- Keep at the SAME abstraction level as the parent (do not prematurely narrow); deeper drilling will happen in the next recursion.
- Avoid overlap; prioritize facets not yet evidenced by the snippets.
- Keep queries concise (<12 tokens), factual, and non-redundant.

Return JSON:
{{
  "queries": ["q1", "q2", "q3"]
}}"""
        text = await self.call_llm(prompt, client)
        try:
            parsed = json.loads(text.split("```json")[-1].split("```")[0].strip())
            print(parsed)
            if isinstance(parsed.get("queries"), list):
                return [q for q in parsed["queries"] if isinstance(q, str) and q.strip()][: limit]
        except Exception as e:
            print(f"[WARN] propose_child_queries_v2 parse failed: {e}")
        return []

    async def should_stop(self, depth: int, child_queries: List[str], statements: List[str], client) -> (bool, str):
        if depth >= self.max_depth:
            return True, "depth>=max_depth"
        if self.node_count >= self.max_nodes:
            return True, "node_count>=max_nodes"
        if not child_queries:
            return True, "no_child_queries"
        # Lightweight LLM gate to avoid useless expansion; reminds width/depth semantics and completeness
        gate_prompt = f"""Decide whether to branch further in the research tree.
Depth now: {depth} (max {self.max_depth})
Role of siblings: parallel breadth facets; deeper levels: drill each facet.
Statements so far (sample): {statements[:10]}
Proposed sibling queries: {child_queries}

Ask yourself:
- Do the proposed siblings open meaningfully different facets (stakeholders, mechanisms, timeline, evidence/data, risks, alternatives, geography, regulation, controversy)?
- Are mandatory facets covered: at least one comparison/alternative, one risks/failures, one quantitative/data-source angle?
- Is there still obvious missing breadth or depth that could change the final research answer?

Return JSON:
{{
  "stop": true/false,
  "reason": "short rationale"
}}
Stop only if additional branching is unlikely to add novel, decision-relevant evidence."""
        text = await self.call_llm(gate_prompt, client)
        try:
            parsed = json.loads(text.split("```json")[-1].split("```")[0].strip())
            return bool(parsed.get("stop", False)), parsed.get("reason") or "llm_gate"
        except Exception:
            return False, "gate_parse_fail"

    async def expand_node(self, node: SearchNode, qa_client: OpenAIAPIClient, extract_client: OpenAIAPIClient, note: str = "") -> SearchNode:
        self.node_count += 1
        snippets = await self.retrieve_snippets([node.query])
        node.statements = await self.extract_statements([node.query], f"focus on '{node.query}'", snippets, extract_client)

        child_queries = await self.propose_child_queries_v2(
            node.query, snippets, node.depth, node.statements, extract_client
        )
        stop, reason = await self.should_stop(node.depth, child_queries, node.statements, qa_client)
        node.stop = stop
        node.stop_reason = reason

        if stop:
            return node

        uniq_child = []
        seen = set()
        for q in child_queries:
            if q not in seen:
                uniq_child.append(q)
                seen.add(q)

        max_c = self.max_children_for_depth(node.depth)
        for q in uniq_child[: max_c]:
            child = SearchNode(query=q, depth=node.depth + 1)
            child = await self.expand_node(child, qa_client, extract_client, note)
            node.children.append(child)

        return node

    def collect_statements(self, node: SearchNode) -> List[str]:
        acc = list(node.statements)
        for c in node.children:
            acc.extend(self.collect_statements(c))
        # de-duplicate while preserving order
        seen = set()
        dedup = []
        for s in acc:
            if s not in seen:
                dedup.append(s)
                seen.add(s)
        return dedup

    async def extract_root_content(self, root_url: str, client: OpenAIAPIClient) -> tuple[List[str], Optional[str]]:
        """Use page cache + LLM summary/info extraction to seed root statements."""
        content = find_page(self.pages, root_url)["contents"]
        # content = page.get("contents") or page.get("page") or ""
        if not content:
            return [], None
        # Limit to avoid huge prompts
        content = content[:15000]
        summary_prompt = ConstructQAPrompts.summarize_webpage.format(content=content)
        info_prompt = ConstructQAPrompts.extract_information_points.format(content=content)
        summary_text = await self.call_llm(summary_prompt, client)
        info_text = await self.call_llm(info_prompt, client)

        stmts = []
        title = None
        if "<title>" in summary_text and "</title>" in summary_text:
            title = summary_text.split("<title>")[-1].split("</title>")[0].strip()
        # parse <title>/<summary> similar to extract_webpage
        if "<summary>" in summary_text and "</summary>" in summary_text:
            summary = summary_text.split("<summary>")[-1].split("</summary>")[0].strip()
            if summary:
                stmts.append(summary)
        # information points JSON list
        if "```json" in info_text and "```" in info_text.split("```json")[-1]:
            try:
                info_json = info_text.split("```json")[-1].split("```")[0].strip()
                infos = json.loads(info_json)
                if isinstance(infos, list):
                    stmts.extend([s for s in infos if isinstance(s, str)])
            except Exception:
                pass

        # fallback: add first 2 paragraphs if stmts still empty
        if not stmts:
            paragraphs = content.split("\n\n")[:2]
            stmts.extend(paragraphs)
        return stmts, title

    async def build_tree(self, root_query: str, qa_client: OpenAIAPIClient, extract_client: OpenAIAPIClient, root_url: Optional[str] = None) -> TreeResult:
        self.node_count = 0
        root = SearchNode(query=root_query, depth=0)
        # seed root statements from cached page if available
        if root_url:
            try:
                seeded, title = await self.extract_root_content(root_url, extract_client)
                root.statements.extend(seeded)
                if title:
                    root.query = title
            except Exception as e:
                print(f"[WARN] seed root content failed: {e}")
        root = await self.expand_node(root, qa_client, extract_client, note=root.query)
        statements = self.collect_statements(root)
        return TreeResult(root=root, statements=statements)

    def render_tree(self, node: SearchNode, indent: int = 0, lines: Optional[List[str]] = None, max_lines: int = 200, include_reason: bool = False) -> str:
        """Pretty-print search tree; optionally include stop reasons."""
        if lines is None:
            lines = []
        if len(lines) >= max_lines:
            return "\n".join(lines[:max_lines]) + "\n... (truncated)"
        prefix = "  " * indent
        sample_stmt = node.statements[0][:80] + "..." if node.statements else ""
        reason = f" | stop_reason={node.stop_reason}" if (include_reason and node.stop) else ""
        lines.append(f"{prefix}- query: {node.query} | depth={node.depth} | stmts={len(node.statements)}{reason} | sample: {sample_stmt}")
        for child in node.children:
            self.render_tree(child, indent + 1, lines, max_lines, include_reason)
            if len(lines) >= max_lines:
                break
        return "\n".join(lines[:max_lines])

    def _trim_statements(self, statements: List[str], max_items: int = 4, max_len: int = 240) -> List[str]:
        trimmed = []
        seen = set()
        for s in statements:
            if not s:
                continue
            s = s.strip().replace("\n", " ")
            if not s or s in seen:
                continue
            if len(s) > max_len:
                s = s[:max_len].rstrip() + "..."
            trimmed.append(s)
            seen.add(s)
            if len(trimmed) >= max_items:
                break
        return trimmed

    def build_tree_payload(self, root: SearchNode) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        id_to_node: Dict[str, SearchNode] = {}

        def walk(node: SearchNode, parent_id: Optional[str] = None) -> str:
            node_id = f"N{len(nodes) + 1}"
            id_to_node[node_id] = node
            entry = {
                "id": node_id,
                "parent": parent_id,
                "depth": node.depth,
                "query": node.query,
                "children": [],
            }
            nodes.append(entry)
            for child in node.children:
                child_id = walk(child, node_id)
                entry["children"].append(child_id)
            return node_id

        walk(root, None)
        id_to_entry = {n["id"]: n for n in nodes}

        def path_for(node_id: str) -> List[str]:
            path = []
            cur = id_to_entry.get(node_id)
            while cur is not None:
                path.append(cur["query"])
                cur = id_to_entry.get(cur["parent"])
            return list(reversed(path))

        leaves = []
        for n in nodes:
            if not n["children"]:
                leaf_node = id_to_node[n["id"]]
                leaves.append(
                    {
                        "id": n["id"],
                        "path": path_for(n["id"]),
                        "statements": self._trim_statements(leaf_node.statements),
                    }
                )

        return {
            "root_query": root.query,
            "root_id": "N1",
            "nodes": nodes,
            "leaves": leaves,
        }

    # ---- End-to-end generation ----
    async def generate_single(self, root_query: str, qa_client: OpenAIAPIClient, extract_client: Optional[OpenAIAPIClient], save_path: str, root_url: Optional[str] = None) -> Dict[str, Any]:
        extract_client = extract_client or qa_client
        tree = await self.build_tree(root_query, qa_client, extract_client, root_url=root_url)
        root_query = tree.root.query
        tree_repr = self.render_tree(tree.root, include_reason=False)
        tree_repr_reason = self.render_tree(tree.root, include_reason=True)
        tree_payload = self.build_tree_payload(tree.root)
        enriched_json = json.dumps(
            {
                "search_tree_text": tree_repr_reason,
                "tree_payload": tree_payload,
                "root_query": root_query,
            },
            ensure_ascii=False,
        )
        enriched_content = json.dumps(tree_payload, ensure_ascii=False)
        base_qa = await self.call_llm(BASE_QA_TREE_PROMPT.format(content=enriched_content), qa_client)
        try:
            qa = json.loads(base_qa.split("```json")[-1].split("```")[0].strip())
        except Exception:
            qa = {"question": root_query, "rubrics": []}

        uid = str(uuid.uuid4())
        out = dict(
            uid=uid,
            root_query=root_query,
            question=qa.get("question"),
            rubrics=qa.get("rubrics", []),
            statements=tree.statements,
            tree_depth=self.max_depth,
            selected_leaf_ids=qa.get("selected_leaf_ids", []),
            selection_trace=qa.get("selection_trace", []),
            merge_trace=qa.get("merge_trace", []),
            tree_repr=tree_repr,
            tree_repr_with_reason=tree_repr_reason,
            tree_json=enriched_json,
        )
        with open(f"{save_path}/{uid}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False))
        return out

def normalize_url(url):    
    if "index.php/" in url:
        url = url.replace("index.php/", "index.php?title=")
    if "/wiki/" in url:
        url = url.replace("/wiki/", "/w/index.php?title=")
    if "_" in url:
        url = url.replace("_", "%20")
    return url
def find_page(pages, url):
    if isinstance(pages, dict):
        url = normalize_url(url)
        return pages.get(url, None)
    if isinstance(pages, list):
        if not pages:
            return None
        # pages may be a list of text passages; return one line's text.
        return random.choice(pages)
    return None


    
# ---- Convenience runner ----
async def run_recursive_generation(
    pages_path: str,
    links_path: str,
    save_path: str,
    n_samples: int = 1,
    max_depth: int = 3,
    max_children: int = 3,
    retriever_url: str = "http://localhost:8000/search",
    qa_model: str = "deepseek/deepseek-chat",
    extract_model: str = "Qwen/Qwen3.5-35B-A3B-Instruct",
):
    import os
    import json as pyjson
    import tqdm

    os.makedirs(save_path, exist_ok=True)

    def load_pages(p_path: str):
        pages = []
        with open(p_path, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                if not line.strip():
                    continue
                pages.append(pyjson.loads(line))
        if not pages:
            return []
        sample = pages[0]
        if isinstance(sample, dict) and "url" in sample:
            return {page["url"]: page for page in pages if isinstance(page, dict) and "url" in page}
        if isinstance(sample, dict) and "text" in sample:
            return [page.get("text", "") for page in pages if isinstance(page, dict) and page.get("text")]
        return []

    pages_data = load_pages(pages_path)

    all_links = None
    all_urls = None
    if isinstance(pages_data, dict):
        all_links = pyjson.load(open(links_path, "r", encoding="utf-8"))
        all_urls = list(all_links.keys())

    search_client = RetrieveSearchClient(base_url=retriever_url)
    qa_client = OpenAIAPIClient(model_name=qa_model)
    extract_client = OpenAIAPIClient(model_name=extract_model)


    task_sem = asyncio.Semaphore(32)

    async def worker(idx: int):
        async with task_sem:
            root_url = None
            if isinstance(pages_data, list):
                root_query = random.choice(pages_data)
            else:
                root_url = random.choice(all_urls)
                root_query = all_links[root_url]["title"] if isinstance(all_links[root_url], dict) and "title" in all_links[root_url] else root_url
            # instantiate agent per task to avoid shared mutable state across tasks
            agent = RecursiveQAAgent(
                pages=pages_data,
                search_client=search_client,
                max_depth=max_depth,
                max_children=max_children,
            )
            return await agent.generate_single(root_query, qa_client, extract_client, save_path, root_url=root_url)

    async with qa_client, extract_client:
        tasks = [asyncio.create_task(worker(i)) for i in range(n_samples)]
        results = []
        for fut in asyncio.as_completed(tasks):
            try:
                results.append(await fut)
            except Exception as e:
                print(f"[WARN] sample task failed: {e}")
        return results



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", required=False, default="/path/to/OpenScholar-DataStore-V3/passages/raw_passages-0-of-16.jsonl")
    parser.add_argument("--links", required=False, default="/path/to/ASearcher-Local-Knowledge/wikilinks.json")
    parser.add_argument("--save", required=False, default="outputs/openscholar_evidence_trees")
    parser.add_argument("--retriever-url", default="http://localhost:8000/search")
    parser.add_argument("--qa-model", default="deepseek/deepseek-chat")
    parser.add_argument("--extract-model", default="Qwen/Qwen3.5-35B-A3B-Instruct")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--max_children", type=int, default=3)
    args = parser.parse_args()

    asyncio.run(
        run_recursive_generation(
            pages_path=args.pages,
            links_path=args.links,
            save_path=args.save,
            n_samples=args.n,
            max_depth=args.max_depth,
            max_children=args.max_children,
            retriever_url=args.retriever_url,
            qa_model=args.qa_model,
            extract_model=args.extract_model,
        )
    )
