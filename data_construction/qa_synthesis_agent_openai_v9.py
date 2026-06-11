import re
import time
from datetime import datetime
import random
import uuid
import json
import copy
import asyncio, aiohttp
import requests
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from typing import Dict, List, Any, Optional
import openai
import os
import asyncio
from prompt import SYSTEM_PROMPT

class RetrieveSearchClient:
    """Minimal async client for the /retrieve endpoint."""
    def __init__(self, base_url: str = "http://localhost:8888/retrieve", timeout: int = 20):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def retrieve(self, queries, topk=5, return_scores=True):
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            payload = dict(queries=queries, topk=topk, return_scores=return_scores)
            async with session.post(self.base_url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("result", [])
    async def doc_retrieve(self, queries, topk=3, return_scores=True):
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            payload = dict(queries=queries, topk=topk, return_scores=return_scores)
            async with session.post(self.base_url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                content = ""
                for index, r in enumerate(data.get("result", [])[0]):
                    content += f"Doc {index}:{r['document']['contents'].split('# Sources.')[0]} \n\n"
                
                return content

class ConstructQAPrompts:
    # base & link qa construct
    base_qa = '''You are an autonomous agent for proposing **deep-research seed questions**.

Your task is to define a research problem that is *worth expanding* and that ultimately requires a **long-form, structured answer**, such as an analytical report or research-style write-up.
You are NOT expected to fully specify or solve the problem.

Given the provided material, you must:
1) formulate ONE research-worthy question,
2) explicitly enumerate the rubrics a thorough answer would need to satisfy,
3) assign an importance weight to each rubric,
4) and ground factual rubrics in concrete evidence from the material.

The available evidence may be partial or incomplete. This is expected.

### rubric Types
Each rubric MUST belong to exactly one of the following categories:

- **Logical**: rubrics about the structure, organization, or reasoning of a long-form answer
  (e.g., section dependencies, comparative logic, causal structure, coherence across sections, readability).

- **Factual**: rubrics about specific facts, claims, or descriptions that must be addressed in a long-form answer
  (these MUST be supported by evidence from the material).

### Weighting
- Assign each rubric an importance score between **0 and 1**
- Higher weight means the rubric is more critical to answering the question correctly
- Weights do not need to sum to 1

### Rubric quality (applies to every rubric)
- Make each rubric precise, atomic, and binary-checkable; avoid vague verbs.
- Anchor factual rubrics to concrete supporting statements from the material; do not invent new facts.
- Cover evidence breadth & attribution, reasoning soundness, completeness of expected findings, organization/clarity, and handling of ambiguity or counterpoints.
- Keep rubrics concise and avoid duplicate checks.
- Each rubric item must be evaluated independently and must not depend on the satisfaction or failure of any other rubric item.

### Question Construction
- The question should be motivated by the selected rubrics and evidence
- It should expose meaningful gaps, unresolved relationships, or implicit constraints
- It must NOT be fully answerable using the provided material alone
- Do NOT explicitly list rubrics or solution steps in the question
- Prefer questions that are brief, precise, and research-heavy, rather than descriptive or essay-like. Avoid trivial, purely descriptive, or artificially verbose questions
- Do NOT invent facts beyond the material
- You MUST generate a minimum of 5 rubrics in the output
- The rubrics should be numbered sequentially (R1, R2, R3, R4, R5...)

# Material
{content}

Return JSON ONLY like:
```json
{{
  "question": "a research-worthy deep-research question",
  "rubrics": [
    {{
      "id": "R1",
      "type": "logical",
      "description": "logical or structural rubric of the answer",
      "weight": 0.8
    }},
    {{
      "id": "R2",
      "type": "factual",
      "description": "fact-based rubric that must be addressed",
      "weight": 0.6,
      "evidence": [
        "supporting statement from the material",
        "another supporting statement"
      ]
    }}
    ......
  ]
}}
```
'''
  


    extract_relevant_for_note = """You are an evidence extractor. Given search queries, an action note (goal), and a webpage summary, return 1-3 concise factual statements for each query from the summary that are relevant to the queries and help achieve the action note. 

Queries:
```txt
{queries}
```

Action note:
```txt
{note}
```

Webpage summary:
```txt
{summary}
```

Return JSON in the following format:
```json
{{
    "statements": [ "...", "..." ]
}}
```"""


    

    # action
    action = '''You are an autonomous action controller in a deep-research question construction pipeline.

Your task is to iteratively improve the difficulty and research quality of a question whose correct answer must take the form of a **long-form, structured report** by selecting EXACTLY ONE action at each step.

GLOBAL CONSTRAINT (MUST HOLD AT ALL TIMES):
- The question must preserve ONE single research target.
- You MUST generate a minimum of 5 rubrics in the output
- Difficulty must arise from deeper or broader investigation of the SAME target,
  NOT from concatenating multiple unrelated questions or introducing parallel goals.

RUBRIC EXPECTATIONS (ALWAYS APPLY):
- Keep rubrics as precise, atomic, binary-checkable rubric items; avoid vague verbs.
- Tie factual rubrics to existing supporting statements; do not invent facts.
- Cover evidence & attribution, reasoning soundness, coverage/scope, clarity/structure, and handling of ambiguity or counterpoints.
- Keep rubric checks concise, non-duplicative, and grounded.
- Each rubric item must be evaluated independently and must not depend on the satisfaction or failure of any other rubric item.

Difficulty should increase progressively through:
- discovering missing or implicit research obligations,
- grounding existing obligations with sufficient evidence,
- or improving how the question is framed as a research problem.

You MUST choose the most appropriate next action based on the current state.
If an action would violate the single research target constraint, it is INVALID.

---

## Available Actions (choose ONE)


- **EXPAND**  
  Choose this action when:
  - the current rubrics appear incomplete, shallow, or under-specified,
  - the question could benefit from broader context, alternative perspectives,
    deeper mechanisms, or unresolved aspects,
  - and additional exploration is needed to discover *new or missing*
    research obligations.

- **RETRIEVE**  
  Choose this action when:
  - the research obligations are already identified,
  - but existing evidence is insufficient, thin, or homogeneous,
  - and more factual grounding is needed to support the SAME rubrics.

- **REWRITE**  
  Choose this action when:
  - the question wording is too explicit, checklist-like, or trivially solvable,
  - the question does not read like a genuine research or investigation prompt,
  - or recent rubric updates should be implicitly reflected in the wording.

- **EXIT**  
   - the question has ONE clear and coherent research target,
   - all research obligations are sufficiently identified and explicitly defined through a comprehensive set of rubrics,
   - each rubric must be specific, atomic, and actionable, providing clear evaluation dimensions and evidentiary requirements for assessing the answer,
   - answering the question requires multi-step reasoning, critical thinking, and the synthesis and verification of evidence from diverse sources,
   - and further actions would risk introducing artificial difficulty, goal drift, or disrupting the research integrity and evaluative balance of the current question.

---
## Current State

Current question:
```txt
{question}

Actions attempted so far (in order):
{action_history}

Recent qa history (action and resulting question after each step):
{qa_history}

You can choose one action from the following types:
{actions}

'''
    


    REWRITE='''REWRITE: rewrite the question wording to improve research style or reduce trivial solvability, while preserving the SAME single research target.

You MAY rewrite the question to:
- improve abstraction, clarity, or research-style framing,
- reduce explicit clues, shortcuts, or checklist-like phrasing,
- make the question sound like a genuine research or investigation prompt.

You are NOT required to explicitly reflect every rubric in the question.
However, when appropriate, you SHOULD allow changes in the rubrics
(e.g., newly added or refined research obligations)
to be implicitly suggested through the scope, framing, or level of abstraction
of the rewritten question.

Constraints:
- You MUST preserve the underlying research target.
- You MUST NOT enumerate rubrics or solution steps.
- You MUST NOT introduce new research goals or parallel inquiries.

The rewritten question should remain minimal and natural,
while implicitly accommodating the current set of research rubrics.

Return JSON:
```json
{
  "action": "REWRITE",
  "question": "rewritten question with the same research target and implicit alignment to current rubrics",
  "note": "what was rewritten and whether any rubric changes were implicitly reflected"
}
```'''
    RETRIEVE='''RETRIEVE: propose search queries to gather additional supporting statements for EXISTING research rubrics.

Purpose:
- ground factual rubrics,
- expand evidence coverage or heterogeneity,
- support deeper reasoning for the SAME research target.

Constraints:
- You MUST NOT retrieve evidence for a new or unrelated question.
- Each query must map to an existing rubric.

Return JSON:
```json
{
  "action": "RETRIEVE",
  "queries": ["query 1", "query 2"],
  "topk": 3,
  "note": "which existing rubrics these queries support"
}
```'''
 
    EXIT='''EXIT: stop editing the question when it is already a good deep-research question.
1. the question requires extensive research to find the correct answer
2. Direct answer accuracy has dropped to a low, stable level
3. The question does NOT leak the exact answer span, document title, or a unique easily searchable string.

If you choose EXIT, the output should be in json format:
```json
{
    "action": "EXIT",
    "note": a short description of why you choose to exit
}
```'''
    EXPAND = '''EXPAND: identify promising directions to expand the difficulty of the question
through retrieval-guided exploration, while preserving a SINGLE coherent research target.

The goal of this action is NOT to rewrite the question or rubrics directly,
but to discover additional information that reveals missing, implicit,
or under-explored research obligations.

You should:
- propose search queries that explore broader context, alternative perspectives,
  deeper mechanisms, or unresolved aspects related to the SAME research target,
- use retrieval to surface signals for potential rubric expansion.

Constraints:
- You MUST preserve exactly ONE research target.
- Queries must relate to the existing question and rubrics,
  not introduce a new or parallel research goal.
- Do NOT rewrite the question.
- Do NOT add or modify rubrics in this step.

Return JSON:
```json
{
  "action": "EXPAND",
  "queries": [
    "search query 1",
    "search query 2"
  ],
  "note": "what expansion directions these queries aim to explore and how they may reveal additional research obligations"
}

```'''




    # utils
    summarize_webpage = '''Given a webpage, summarize the content of this webpage.
    
Webpage content:
```markdown
{content}
```

The output should enclose the title and summary in <title> </title> and <summary> </summary> tags respectively.

'''
    # extract_information_points = '''Given a webpage, extract all information points in this webpage. Your task is to extract ONLY the key factual information points from a Wikipedia article.
    extract_information_points = '''Given a webpage, your task is to extract ONLY the key factual information points in this webpage.

1. ONLY keep information that is:
   - Objective
   - Verifiable
   - Encyclopedic
2. DISCARD:
   - Navigation text
   - References section
   - See also
   - External links
   - Footnotes, citations, templates
   - Lists of trivial details
   - Editorial descriptions or vague statements
3. If the page is noisy, disorganized, or low-quality:
   - Extract ONLY the core identity facts (who/what/where/when)
   - Do NOT attempt to summarize every paragraph
4. Each information point must be a SINGLE, ATOMIC factual statement.
5. Do NOT rephrase with vague language. Be precise.  


Webpage content:
```markdown
{content}
```

The output should be a list of strings in json format:
```json
[
    information point 1,
    information point 2,
    ...
]
```

'''

    # rubric generation for long-form deep research
    generate_rubric = """You are a rubric writer for DEEP-RESEARCH long-form tasks. Given the prompt, expected findings outline, and current supporting statements, produce 5-8 evaluation dimensions with 2-4 precise, atomic rubric items each. Focus on multi-source evidence, reasoning soundness, coverage, clarity, citation/attribution, and handling of ambiguity/alternatives. Keep items short and checkable.

Prompt:
```
{question}
```

Expected findings (long-form answer sketch):
```
{answer}
```

Supporting statements:
```txt
{statements}
```

Return JSON:
```json
{{
  "rubric": [
    {{ "dimension": "Coverage & Scope", "items": ["...", "..."] }},
    {{ "dimension": "Evidence & Attribution", "items": ["...", "..."] }},
    {{ "dimension": "Reasoning", "items": ["...", "..."] }},
    {{ "dimension": "Clarity & Structure", "items": ["...", "..."] }},
    {{ "dimension": "Ambiguity Handling", "items": ["...", "..."] }}
  ]
}}
```"""

 
    propose_queries = """You are a search strategist. Goal: support deep-research QAs. Given a question and already retrieved document snippets, propose up to {num_queries} follow-up search queries that explore different angles (entities, events, dates, organizations, counterfactuals) to force multi-hop evidence. Avoid duplicates and keep each query under 12 tokens.

Question:
{question}

Current snippets:
```txt
{snippets}
```

Return JSON in the following format:
```json
{{
    "queries": ["query 1", "query 2", "query 3"]
}}
```"""

    filter_evidence = """You are an evidence selector. Given a question and a list of candidate statements, return the minimum subset required to support the answer. Keep statements factual and avoid redundancy.

Question:
{question}

Candidate statements:
```txt
{statements}
```

Return JSON in the following format:
```json
{{
    "selected_statements": [ "...", "..." ]
}}
```"""

    EXPAND_FOLLOWUP="""You are an expansion synthesizer following an EXPAND action in a deep-research question construction pipeline.

Your input consists of:
- the current question,
- the current set of research rubrics,
- newly retrieved information from the EXPAND queries.

Your task is to analyze the retrieved information and determine
how it reveals missing, implicit, or under-developed research obligations.
and then produce a REVISED and COMPLETE set of rubrics.

You should:
1) Identify which NEW research obligations are suggested by the retrieved information,
   or which existing rubrics should be refined or clarified.
2) Update the rubrics accordingly by adding or refining research obligations.
3) Decide whether the question wording should be lightly rewritten
   to implicitly accommodate the updated rubrics.
   (Rewriting is OPTIONAL and must preserve the same research target.)
4) Output the FULL, UPDATED set of rubrics after revision
   (including both previous and newly added/modified rubrics).

rubric format:
- id: short identifier (e.g., R1, R2)
- type: "logical" or "factual"
- description: concise research obligation

Constraints:
- You MUST preserve exactly ONE research target.
- All rubrics must jointly constrain the SAME question.
- Do NOT enumerate rubrics or solution steps in the question.
- Do NOT introduce parallel or independent research goals.
- Any question rewrite must be implicit and minimal.

---

### Inputs

Current question:
```txt
{qa}

Current rubrics:
```json
{rubrics}

Newly retrieved information:
```txt
{new_statements}
```

Action Note
```txt
{note}


Return JSON:
```json
{{
  "updated_rubrics": [
    {{
      "id": "R_number",
      "type": "logical | factual",
      "description": "new or refined research obligation suggested by the retrieved information",
      "weight": "the weight of the question",
      "evidence": [], "evidence list if the type is factual"
    }}
  ],
  "updated_question": "OPTIONAL: rewritten question that preserves the same research target and implicitly reflects rubric updates ",
  "note": "brief explanation of how the retrieved information motivated the rubric updates and whether question wording was adjusted"
}}
```
"""

    fuse_multi_source = """You are an evidence refiner in a deep-research QA construction pipeline.

Your task is to improve the supporting evidence for an EXISTING question
and its research rubrics, based on newly retrieved material.

You MUST preserve:
- the question text,
- each rubric's id, type, and weight.

You MAY:
- add or update the `evidence` field for FACTUAL rubrics,
- slightly refine the description of a rubric for clarity
  (without changing its meaning).

You MUST NOT:
- add or remove rubrics,
- change rubric ids, types, or weights,
- introduce new research obligations,
- modify the question.



Current question:
```txt
{qa}
```

Current rubrics (may be empty):
```json
{rubrics}
```

New supporting evidence (retrieved this round):
```txt
{new_statements}
```

The note to use new supporting statements:
```txt
{note}
```

For each rubric:
If type is “factual”:
• select relevant supporting statements from the retrieved material,
• append them to the evidence list (create it if missing),
• do NOT add evidence that is weakly related or redundant.
If type is “logical”:
• leave the rubric unchanged (no evidence needed).

Include ONLY evidence that directly supports the factual rubrics.
Do NOT invent facts or evidence beyond the retrieved material.

Return JSON:
```json
{{
  "question": "unchanged question",
  "rubrics": [
    {{
      "id": "R1",
      "type": "logical",
      "description": "unchanged or slightly clarified description",
      "weight": 0.8
    }},
    {{
      "id": "R2",
      "type": "factual",
      "description": "unchanged or slightly clarified description",
      "weight": 0.6,
      "evidence": [
        "supporting statement selected from retrieved material",
        "another supporting statement selected from retrieved material"
      ]
    }}
  ]
}}
```"""


class AgentMemory:
    '''store the memory, including:
    1. the current question & answer 
    2. the basic statements
    3. the relevent entities and the relevant links
    4. the edit history
    '''

    def __init__(self):
        self.qa = dict(question=None, answer=None)
        self.statements = []  # supporting statements
        self.rubrics = []  # model-stated rubrics to answer
        self.distractor_statements = []  # misleading but factual snippets
        self.relevant = []
        self.rubric = []  # list of dimensions with items
        self.edit_history = []
        self.qa_history = []
        self.action_history = []
        self.statement_history = []
    
    def repr(self):
        relevant = '\n'.join([f'- [{e.name}]({e.url})' for e in self.relevant])
        statements = '\n'.join(self.statements)
        distracts = '\n'.join(self.distractor_statements)
        reqs = '\n'.join(self.rubrics)
        ret = f"""
Question: {self.qa['question']}
rubrics:
```txt
{reqs}
```

Relevant Statements:
```txt
{statements}
```
"""
        return ret

    def statements_repr(self, additional=None):
        return '\n'.join(self.statements + (additional or []))

    def statements_support_repr(self, additional=None):
        return '\n'.join(self.statements + (additional or []))

    def statements_distract_repr(self, additional=None):
        return '\n'.join(self.distractor_statements + (additional or []))

    def dict(self):
        return dict(
            qa=self.qa,
            rubrics=self.rubrics,
            relevant=[e.dict() for e in self.relevant],
            statements=self.statements,
            distractor_statements=self.distractor_statements,
            rubric=self.rubric,
            statement_history = self.statement_history,
            edit_history=self.edit_history,
            action_history=self.action_history,
            qa_history=self.qa_history,
        )

class WebPage:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.summary = None
        self.information_points = None
        self.relevant_links = None
    
    def repr(self):
        ret = f"Title: {self.name}\n URL: {self.url}\n\n# Summary\n\n{self.summary}\n\n# Information Points\n\n{self.information_points}"
        return ret
    
    def summary_repr(self):
        ret = f"Title: {self.name} # Summary\n\n{self.summary}\n"
        return ret
    
    def information_points_repr(self):
        return self.information_points
    
    def dict(self):
        return dict(
            name=self.name,
            url=self.url,
            summary=self.summary,
            information_points=self.information_points,
            relevant_links=self.relevant_links,
        )

def normalize_url(url):    
    if "index.php/" in url:
        url = url.replace("index.php/", "index.php?title=")
    if "/wiki/" in url:
        url = url.replace("/wiki/", "/w/index.php?title=")
    if "_" in url:
        url = url.replace("_", "%20")
    return url

def find_page(pages, url):
    url = normalize_url(url)
    return pages.get(url, None)

def exists_page(pages, url):
    url = normalize_url(url)
    return url in pages

class ConstructQAAgent:
    def __init__(self, all_links, pages, search_client, max_turns=20):
        self.all_links = all_links
        self.pages = pages
        self.all_urls = list(all_links.keys())
        self.search_client = search_client
        self.max_turns = max_turns
        # self.tokenizer = AutoTokenizer.from_pretrained("Qwen__Qwen2.5-32B")

    async def retrieve_and_parse(self, queries, client, topk=4):
        """Multi-query retrieval and parsing into WebPage objects."""
        if self.search_client is None:
            return []
        try:
            results = await self.search_client.retrieve(queries, topk=topk, return_scores=True)
        except Exception as e:
            print(f"[WARNING] retrieve failed for {queries}: {e}")
            return []
        # results may be list per query; flatten
        flat = []
        for entry in results:
            if isinstance(entry, list):
                flat.extend(entry)
            else:
                flat.append(entry)
        webpages = []
        for idx, doc in enumerate(flat):
            document = doc.get("document", doc)
            content = document.get("contents") or document.get("text") or document.get("content") or document.get("passage") or ""
            if not content:
                continue
            url = document.get("url") or document.get("source") or document.get("doc_id") or f"search://{uuid.uuid4()}"
            url = normalize_url(str(url))
            name = document.get("wikipedia_title") or document.get("title") or url
            page = WebPage(name, url)
            
            page.name = name
            page.summary = content[:5000]
            page.information_points = []
            page.relevant_links = []
            webpages.append(page.summary_repr())
            

        return webpages


    async def propose_followup_queries(self, question, snippets, client, num_queries=3):
        print("propose_followup_queries__________________________________________ " ,question,snippets,num_queries)
        prompt = ConstructQAPrompts.propose_queries.format(question=question, snippets="\n".join(snippets), num_queries=num_queries)
        print("propose_followup_queries done__________________________________________ " )
        try:

            text = await self.call_llm(prompt, client)
            print("call done ________________",text)
            parsed = json.loads(text.split("```json")[-1].split("```")[0].strip())
            print("parsed        ",parsed)
            if isinstance(parsed, dict) and isinstance(parsed.get("queries"), list):
                return [q for q in parsed["queries"] if isinstance(q, str) and q.strip()]
        except Exception as e:
            print(f"[WARNING] propose queries failed: {e}")
        return []

    async def filter_evidence_min(self, question, statements, client):
        if not statements:
            return statements
        prompt = ConstructQAPrompts.filter_evidence.format(question=question, statements="\n".join(statements))
        try:
            text = await self.call_llm(prompt, client)
            selected = json.loads(text.split("```json")[-1].split("```")[0].strip())
            if isinstance(selected["selected_statements"], list):
                return [s for s in selected["selected_statements"] if isinstance(s, str)]
        except Exception as e:
            print(f"[WARNING] filter evidence failed: {e}")
        return statements

    async def expand_with_search(self, question, memory, client, hops=1, topk=3, bucket="support",note=""):
        """Run multi-hop search and return new statements without mutating memory."""
        if self.search_client is None:
            return [], []
        queries = [question]
        seed_queries = getattr(memory, "seed_queries", None)
        if seed_queries:
            queries = seed_queries
        seen_urls = {normalize_url(e.url) for e in memory.relevant}
        new_support = []
        new_distract = []
        for hop in range(hops):
            webpages = await self.retrieve_and_parse(queries, client, topk=topk)
            snippets = []
            prompt = ConstructQAPrompts.extract_relevant_for_note.format(
                    queries="\n".join(queries),
                    note=note,
                    summary="\n".join(webpages),
                )
            text = await self.call_llm(prompt, client)
            parsed = json.loads(text.split("```json")[-1].split("```")[0].strip())
            if isinstance(parsed['statements'], list):
                if bucket == "distract":
                    new_distract.extend(parsed['statements'])
                else:
                    new_support.extend(parsed['statements'])
            else:
                s = str(parsed)
                if bucket == "distract":
                    new_distract.extend(s)
                else:
                    new_support.extend(s)
            if not snippets:
                break
            queries = await self.propose_followup_queries(question, webpages, client)
            if not queries:
                break
        
        new_distract = new_distract[:50]
        new_support  = new_support[:50]
        return new_support, new_distract

    async def generate_rubric(self, qa, statements, client):
        """Generate rubric dimensions/items for long-form deep research tasks."""
        prompt = ConstructQAPrompts.generate_rubric.format(
            question=qa.get("question", ""),
            answer=qa.get("answer", ""),
            statements="\n".join(statements or []),
        )
        try:
            text = await self.call_llm(prompt, client)
            parsed = json.loads(text.split("```json")[-1].split("```")[0].strip())
            if isinstance(parsed, dict) and isinstance(parsed.get("rubric"), list):
                # keep only dimension/items keys
                cleaned = []
                for dim in parsed["rubric"]:
                    if isinstance(dim, dict) and "dimension" in dim and "items" in dim:
                        items = [i for i in dim["items"] if isinstance(i, str)]
                        cleaned.append(dict(dimension=dim["dimension"], items=items))
                return cleaned
        except Exception as e:
            print(f"[WARNING] generate_rubric failed: {e}")
        return []

    async def call_llm(self, prompt, client):
        # prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) 
        # print("PROMPT:", [prompt[:1000]])
        # max_new_tokens = 32000 - self.tokenizer([prompt], return_length=True)["length"][0]
        max_new_tokens = 4096 # 为 OpenAI 设置一个合理的默认值

        sampling_kwargs = dict(
            temperature=0.8,
            # sseed=args.seed,
            top_p=0.95,
            top_k=1000,
            max_new_tokens=max_new_tokens,
            n=1,
            # stop_token_ids=[151645, 151643]
        )
        try:
            # 现在调用的是 OpenAIAPIClient 的 async_generate
            output = await client.async_generate(prompt, sampling_kwargs)
        except Exception as e: # 使用更通用的异常捕获
            print(f"Exception when calling llm: {e}")
            raise e
        # print(prompt)
        # print(output["text"])
        return output["text"]
        
    async def extract_webpage(self, url, client) -> WebPage:
        print(f"[INFO] extract information from webpage: {url}", flush=True)
        name = url
        webpage = WebPage(name, url)
        content = find_page(self.pages, url)["contents"] # await self.search_client.access_async([url])
        assert content is not None
        if isinstance(content, dict) and "page" in content:
            content = content["page"]
        title_summary = await self.call_llm(ConstructQAPrompts.summarize_webpage.format(content=content[:15000]), client)
        information_points = await self.call_llm(ConstructQAPrompts.extract_information_points.format(content=content[:15000]), client)

        title, summary, info_pts = None, None, None

        if "<title>"in title_summary and "</title>" in title_summary:
            title = title_summary.split("<title>")[-1].split("</title>")[0].strip()
            name = title
        if "<summary>" in title_summary and "</summary>" in title_summary:
            summary = title_summary.split("<summary>")[-1].split("</summary>")[0].strip()
        if "```json" in information_points and "```" in information_points.split("```json")[-1]:
            info_pts = information_points.split("```json")[-1].split("```")[0].strip()
        
        webpage.name = name
        webpage.summary = summary
        webpage.information_points = info_pts
        webpage.relevant_links = self.all_links[normalize_url(url)]["links"] + self.all_links[normalize_url(url)]["in_links"]

        return webpage

    async def construct_base_qa(self, material, client):
        prompt = ConstructQAPrompts.base_qa.format(content=material)
        text = await self.call_llm(prompt, client)
        base_qa = json.loads(text.split("```json")[-1].split("```")[0].strip())
        return base_qa
    
    async def choose_action(self, state, client, ready_to_exit=False, action_history=None, qa_history=None, edit_history=None):
        # print(f"choose action @ state\n{state}", flush=True)
        recent_actions = action_history or []
        actions = [ConstructQAPrompts.REWRITE,  ConstructQAPrompts.RETRIEVE, ConstructQAPrompts.EXPAND]

        random.shuffle(actions)
        ready_to_exit = True
        if ready_to_exit :
            actions.append(ConstructQAPrompts.EXIT)
        prompt = ConstructQAPrompts.action.format(
            question=state,
            actions=actions,
            action_history=" -> ".join(action_history or []),
            qa_history=" | ".join([json.dumps(qh) for qh in qa_history])
            # last_acc=last_acc or "N/A"
        )
        text = await self.call_llm(prompt, client)
        print("Choose Action:",text)
        action = json.loads(text.split("```json")[-1].split("```")[0].strip())
        assert "action" in action
        if action["action"] == "SELECT":
            assert "target" in action and "note" in action
        elif action["action"] == "FUZZ":
            assert "question" in action
            assert "note" in action
        elif action["action"] == "REFRAME":
            assert "question" in action
            assert "note" in action
        elif action["action"] == "REWRITE":
            assert "question" in action
            assert "note" in action
        elif action["action"] == "RETRIEVE":
            assert "queries" in action
        elif action["action"] == "EXPAND":
            assert "queries" in action
        elif action["action"] == "DISTRACT":
            assert "note" in action
            assert "source_statement" in action
            assert "queries" in action
        elif action["action"] == "EXIT":
            assert "note" in action
        elif action["action"] == "BRAINSTORM":
            assert "entities" in action
            assert all([isinstance(e, dict) and 'name' in e and 'wiki_url' in e and 'statement' in e for e in action['entities']])
        assert action["action"] in ["SELECT", "FUZZ", "REWRITE","EXPAND","RETRIEVE", "FUSE", "DISTRACT", "EXIT", "BRAINSTORM", "REFRAME", "CONDENSE"]
        return action

        
    async def generate(self, semaphore, reasoning_client, instruct_client, gpt_client, save_path):
        async with semaphore:
            # sample a root entity
            root_url = random.choice(self.all_urls)
            memory = AgentMemory()
            uid = str(uuid.uuid4())
            root_entity: WebPage = await self.extract_webpage(root_url, instruct_client)
            memory.relevant.append(root_entity)
            memory.uid = uid

            print("start generting qa")

            # Step1: ask model for queries to gather info
            seed_queries = [root_entity.name]
            query_prompt = f"""You are a search strategist. Root entity: "{root_entity.name}". Propose 3-5 short search queries to gather background and key facts for a deep research question. Return JSON:
```json
{{"queries": ["q1", "q2", "q3"]}}
```
"""
            try:
                q_text = await self.call_llm(query_prompt, gpt_client)
                if "```json" in q_text:
                    qs = json.loads(q_text.split("```json")[-1].split("```")[0].strip()).get("queries", [])
                else:
                    # fallback: parse lines
                    qs = [line.strip("- ").strip() for line in q_text.splitlines() if line.strip()][:5]
                if qs:
                    seed_queries = qs
            except Exception as e:
                print(f"[WARNING] initial query generation failed: {e}")

            # Step2: retrieve with those queries
            extra_snippets = []
            try:
                webpages = await self.retrieve_and_parse(seed_queries, instruct_client, topk=3)
                extra_snippets = webpages[:5]
            except Exception as e:
                print(f"[WARNING] initial multi-query retrieve failed: {e}")
            combined_material = root_entity.repr() + "\n\n" + "\n\n".join(extra_snippets)

            # Step3: generate base QA with rubrics and evidence
            try:
                base_qa = await self.construct_base_qa(combined_material, gpt_client)
            except:
                print("[ERROR] generate base qa failed")
                return None
            if base_qa is None or not (isinstance(base_qa, dict) and "question" in base_qa and "rubrics" in base_qa):
                print("[ERROR] generate base qa failed")
                return None
            
            memory.qa["question"] = base_qa["question"]
            memory.rubrics = base_qa.get("rubrics") or []
            # extract evidence from rubrics into statements
            for req in memory.rubrics:
                evs = req.get("evidence") if isinstance(req, dict) else None
                if isinstance(evs, list):
                    for ev in evs:
                        if isinstance(ev, str) and ev not in memory.statements:
                            memory.statements.append(ev)
            memory.qa_history.append(base_qa)
            memory.edit_history.append(f"Create a base question from entity: {root_entity.name}\nQuestion: {base_qa['question']}")
            # try:
            #     memory.rubric = await self.generate_rubric(memory.qa, memory.statements, instruct_client)
            # except Exception as e:
            #     print(f"[WARNING] initial rubric generation failed: {e}")

            # print(memory.edit_history[-1]) 

            
            ready_to_exit = False

            action_stats = defaultdict(int)
        
            for turn in range(self.max_turns):

                print(f"Turn: {turn}")
                state = memory.qa['question']
                if turn == 0:
                    continue
                    action = dict(action="none")
                else:
                    try:
                        action = await self.choose_action(
                            state,
                            gpt_client,
                            ready_to_exit=ready_to_exit,
                            # action_history=memory.action_history,
                            action_history=memory.action_history,
                            edit_history=memory.edit_history,
                            qa_history=memory.qa_history,
                        )
                    except:
                        print(f"[WARNING] generate action failed at turn {turn}")
                        continue
                # print("History:\n\n{}".format("\n".join(memory.edit_history)))
                
                q_new = None
                memory_new = copy.deepcopy(memory)
                memory_new.edit_history.append(f"Action: {action['action']}. Note: {action.get('note', None)}")
                print(action)
                action_stats[action["action"]] += 1
                
                memory_new.action_history.append(action["action"])
                if action["action"] == "FUZZ":
                    q_new = action["question"]
                    memory_new.edit_history.append(f"FUZZ operation modifies the question to: {q_new}")
                elif action["action"] == "EXPAND":
                    queries = action.get("queries", [])
                    topk = action.get("topk", 3) or 3
                    memory_new.seed_queries = queries
                    try:
                        support_new, distract_new = await self.expand_with_search(memory.qa["question"], memory_new, instruct_client, hops=1, topk=topk, note=action.get("note", ""))
                        # memory_new.edit_history.append(f"RETRIEVE adds docs via queries={queries}, topk={topk}")
                        # Immediately rewrite question with multi-source fusion using new evidence
                        prompt = ConstructQAPrompts.EXPAND_FOLLOWUP.format(
                            qa=json.dumps(memory.qa),
                            rubrics=json.dumps(memory.rubrics, ensure_ascii=False),
                            new_statements="\n".join(support_new),
                            # statements=memory.statements_support_repr(additional=[]),
                            # new_statements="\n".join(support_new),
                            note =  action.get("note",""),
                        )
                        text = await self.call_llm(prompt, gpt_client)
                        fused = json.loads(text.split("```json")[-1].split("```")[0].strip())
                        
                        q_new = fused.get("updated_question",memory.qa["question"])
                        
                        if isinstance(fused.get("updated_rubrics"), list):
                            memory_new.rubrics = fused["updated_rubrics"]
                        # if isinstance(fused.get("statements"), list):
                        #     for ev in fused["statements"]:
                        #         if ev not in memory_new.statements:
                        #             memory_new.statements.append(ev)
                        memory_new.edit_history.append(f"EXPAND question with new evidence: {q_new}")
                        
                    except Exception as e:
                        print(f"[WARNING] EXPAND failed: {e}")
                        continue

                    
                elif action["action"] == "REWRITE":
                    q_new = action["question"]
                    memory_new.edit_history.append(f"REWRITE operation modifies the question to: {q_new}")
                elif action["action"] == "RETRIEVE":
                    queries = action.get("queries", [])
                    topk = action.get("topk", 3) or 3
                    if self.search_client is None or len(queries) == 0:
                        print("[WARNING] RETRIEVE skipped (no search client or empty queries)")
                        continue
                    memory_new.seed_queries = queries
                    try:
                        support_new, distract_new = await self.expand_with_search(memory.qa["question"], memory_new, instruct_client, hops=1, topk=topk, note=action.get("note", ""))
                        # memory_new.edit_history.append(f"RETRIEVE adds docs via queries={queries}, topk={topk}")
                        # Immediately rewrite question with multi-source fusion using new evidence
                        prompt = ConstructQAPrompts.fuse_multi_source.format(
                            qa=json.dumps(memory.qa),
                            rubrics=json.dumps(memory.rubrics, ensure_ascii=False),
                            # statements=memory.statements_support_repr(additional=[]),
                            new_statements="\n".join(support_new),
                            note =  action.get("note"),
                        )
                        text = await self.call_llm(prompt, gpt_client)
                        fused = json.loads(text.split("```json")[-1].split("```")[0].strip())
                        q_new = fused["question"]
                        
                        if isinstance(fused.get("rubrics"), list):
                            memory_new.rubrics = fused["rubrics"]
                        if isinstance(fused.get("statements"), list):
                            for ev in fused["statements"]:
                                if ev not in memory_new.statements:
                                    memory_new.statements.append(ev)
                        memory_new.edit_history.append(f"RETRIEVE with new evidence: {json.dumps(memory_new.rubrics, ensure_ascii=False)}")
                        for s in support_new:
                            if s not in memory_new.statements:
                                memory_new.statements.append(s)
                    except Exception as e:
                        print(f"[WARNING] RETRIEVE failed: {e}")
                        continue
               
                elif action["action"] == "EXIT":
                    print("[INFO] question generation is done, exiting")
                    break
                elif action["action"] == "none":
                    assert turn == 0
                print(f"New Question at turn {turn}:\n{q_new}")
                memory_new.qa["question"] = q_new



                # if len(memory_new.statements) > 10:
                #     try:
                #         print(f"The number of statements is {len(memory_new.statements)}")
                #         memory_new.statement_history.append(memory_new.statements)
                #         if len(memory_new.statements) > 30:
                #             memory_new.statements = await self.filter_evidence_min(q_new, memory_new.statements, instruct_client)
                #         else:
                #             memory_new.statements = await self.filter_evidence_min(q_new, memory_new.statements, gpt_client)
                #         memory_new.statement_history.append(memory_new.statements)
                #         print(f"The number of statements after filter is {len(memory_new.statements)}")
                #     except Exception as e:
                #         print(f"[WARNING] filter_evidence_min failed: {e}")

                # refresh rubric for the updated prompt

                # try:
                #     memory_new.rubric = await self.generate_rubric(memory_new.qa, memory_new.statements, instruct_client)
                # except Exception as e:
                #     print(f"[WARNING] rubric refresh failed: {e}")

#                 # check validity
#                 valid = False
#                 try:
#                     # valid = await self.check_qa_valid(memory_new.repr(), reasoning_client)
#                     valid = await self.check_qa_valid(memory_new.repr(), instruct_client)
#                 except:
#                     pass
#                 if not valid:
#                     print(f"[WARNING] the constructed qa is invalid at turn {turn}.")
#                     continue
#                 # critique-based evaluation (replaces direct/tool acc)
#                 try:
#                     prompt = f"""{ConstructQAPrompts.CRITIQUE}

# Current prompt:
# {memory_new.qa["question"]}


# Rubric (json):
# {json.dumps(memory_new.rubric, ensure_ascii=False)}

# Supporting statements:
# {memory_new.statements_support_repr()}
# """
#                     text = await self.call_llm(prompt, gpt_client)
#                     crit = json.loads(text.split("```json")[-1].split("```")[0].strip())
#                     crit_issues = crit.get("issues") or []
#                     crit_question = crit.get("question") or memory_new.qa["question"]
#                     crit_note = crit.get("note")
#                     if crit_question and crit_question != memory_new.qa["question"]:
#                         memory_new.qa["question"] = crit_question
#                         memory_new.edit_history.append(f"[CRITIQUE rewrite] {crit_question}")
#                     if crit_issues:
#                         memory_new.edit_history.append(f"[CRITIQUE issues] {crit_issues}")
#                     if crit_note:
#                         memory_new.edit_history.append(f"[CRITIQUE note] {crit_note}")
#                     memory_new.qa_history.append(dict(
#                         question=memory_new.qa["question"],
#                         answer=memory_new.qa["answer"],
#                         critique_issues=crit_issues,
#                         critique_note=crit_note,
#                     ))
#                 except Exception as e:
#                     print(f"[WARNING] critique evaluation failed: {e}")
#                     memory_new.qa_history.append(dict(
#                         question=memory_new.qa["question"],
#                         answer=memory_new.qa["answer"],
#                         critique_issues=[],
#                         critique_note="[critique failed]"
#                     ))

                memory_new.qa_history.append(dict(
                    question =memory_new.qa["question"],
                    rubrics=json.dumps(memory_new.rubrics, ensure_ascii=False),))

                memory = memory_new
                
                print(f"Action stats: {action_stats}", flush=True)
            with open(f"{save_path}/{uid}.jsonl", "w") as f:
                _=f.write(json.dumps(memory.dict()))
            return memory
            
async def main(agent, reasoning_client, instruct_client, gpt_client, save_path):
    semaphore = asyncio.Semaphore(128)
    await instruct_client.__aenter__()
    tasks = [agent.generate(semaphore, reasoning_client, instruct_client, gpt_client, save_path) for i in range(1024*8)]
    results = await asyncio.gather(*tasks)
    await instruct_client.__aexit__()

class OpenAIAPIClient:
    """
    OpenAI-compatible async client used by the data-construction pipeline.

    Configure the endpoint with environment variables instead of hard-coding
    private services:

    - OPENAI_BASE_URL, OPENAI_API_KEY
    - or DEEPRUBRIC_LLM_BASE_URL, DEEPRUBRIC_LLM_API_KEY
    """
    def __init__(self, model_name="gpt-4o", base_url=None, api_key=None):
        self.model = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.client = None
        self.qwen_endpoints = []
        self.qwq_endpoints = []
        self._qwen_client_pool = {}

    async def __aenter__(self):
        base_url = (
            self.base_url
            or os.getenv("DEEPRUBRIC_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "http://localhost:8008/v1"
        )
        api_key = (
            self.api_key
            or os.getenv("DEEPRUBRIC_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY"
        )
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出方法"""
        if self.client:
            # AsyncOpenAI 客户端通常不需要显式关闭会话
            await self.client.close()
        # close any pooled qwen clients
        for c in getattr(self, "_qwen_client_pool", {}).values():
            try:
                await c.close()
            except Exception:
                pass


    async def async_generate(self, prompt: str, sampling_kwargs: Dict) -> Dict[str, Any]:
        """
        使用 OpenAI API 生成文本。

        Args:
            prompt: 发送给模型的提示文本。
            sampling_kwargs: 包含采样参数的字典。

        Returns:
            一个字典，格式为 {"text": "生成的文本"} 或 {"text": ["文本1", "文本2", ...]}
        """
        if not self.client:
            raise RuntimeError("Client has not been initialized. Use 'async with' statement.")

        # 映射 sampling_kwargs 到 OpenAI API 参数
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": sampling_kwargs.get("temperature", 0.8),
            "top_p": sampling_kwargs.get("top_p", 0.95),
            "n": sampling_kwargs.get("n", 1),
            # 注意：OpenAI 使用 max_tokens 而不是 max_new_tokens
            "max_tokens": sampling_kwargs.get("max_new_tokens", 8096),
            # "reasoning_mode": "off",  # 禁用自动推理
            # "reasoning":{"effort":"minimal"},
            # 注意：OpenAI 的 stop 参数是字符串列表，不是 token IDs
            # 如果需要，可以在这里添加 stop: sampling_kwargs.get("stop", None)
        }
        # print(params)

        try:
            st = time.time()

            client = self.client

            response = await client.chat.completions.create(**params)

            
            # response = await self.client.chat.completions.create(**params)

            # response = await self.client.chat.completions.create(**params)
            latency = time.time() - st
            print(f"[INFO] {self.model} API call took {latency:.2f} seconds.")
            # print(response)
            usage = getattr(response, "usage", None)
            usage_dict = None
            if usage is not None:
                if hasattr(usage, "model_dump"):
                    usage_dict = usage.model_dump()
                elif isinstance(usage, dict):
                    usage_dict = usage
                else:
                    usage_dict = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
            if usage_dict:
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "model": self.model,
                    "prompt_tokens": usage_dict.get("prompt_tokens"),
                    "completion_tokens": usage_dict.get("completion_tokens"),
                    "total_tokens": usage_dict.get("total_tokens"),
                    "n": params["n"],
                    "prompt": prompt[:1000],
                }
                # print(
                #     f"[TOKENS] model={log_entry['model']} prompt={log_entry['prompt_tokens']} "
                #     f"completion={log_entry['completion_tokens']} total={log_entry['total_tokens']}"
                # )
                try:
                    with open("token_usage.log", "a") as log_f:
                        log_f.write(json.dumps(log_entry) + "\n")
                except Exception as log_err:
                    print(f"[WARNING] failed to write token usage log: {log_err}")

            if params["n"] == 1:
                output_text = response.choices[0].message.content
                return {"text": output_text, "usage": usage_dict}
            else:
                output_texts = [choice.message.content for choice in response.choices]
                return {"text": output_texts, "usage": usage_dict}

        except Exception as e:
            print(f"[ERROR] An error occurred with the OpenAI API: {e}")
            # 为了保持兼容性，在出错时可以返回一个空结果
            if params["n"] == 1:
                return {"text": ""}
            else:
                return {"text": [""] * params["n"]}


if __name__ == "__main__":
    raise SystemExit(
        "This module provides shared prompts and clients. "
        "Use recursive_qa_agent_v4.py or recursive_qa_agent_v44.py as entrypoints."
    )
