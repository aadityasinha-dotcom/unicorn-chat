"""Tools the model can call. Schemas are strict JSON schema; the executor maps
tool names onto ``CompanyRepository`` methods and returns compact JSON strings.

``ask_clarifying_question`` is special: it terminates the turn and its argument
is returned to the user verbatim. Modelling clarification as a tool (rather than
free text) makes it observable - we can count it, score it, and test it.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from app.data.repository import CompanyRepository

CLARIFY_TOOL = "ask_clarifying_question"
MAX_RESULTS = 40


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _str_array(desc: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": desc}


TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "get_company",
        "description": (
            "Fetch the full record for one company by name (exact, substring or fuzzy match). "
            "Returns the description, sector, city, stage, founding year, investors, revenue, valuation, "
            "funding, head-count and unicorn-entry details. If several companies match, all candidates are returned."
        ),
        "strict": True,
        "input_schema": _schema({"name": {"type": "string", "description": "Company name as the user wrote it"}}, ["name"]),
    },
    {
        "name": "filter_companies",
        "description": (
            "Structured filter over the catalogue. All arguments are optional and combined with AND; "
            "lists are OR-ed. Sector and location values should be copied from the catalogue in the system prompt "
            "(matching is case-insensitive substring). Use `keywords` to match words in the company description."
        ),
        "strict": True,
        "input_schema": _schema(
            {
                "sectors": {**_str_array("Primary sectors, e.g. [\"Payments\", \"Alternative Lending\"]"), "default": []},
                "locations": {**_str_array("Cities, e.g. [\"Bengaluru\"]"), "default": []},
                "countries": {**_str_array("Countries: India, United States, Singapore"), "default": []},
                "stages": {**_str_array("Funding stages, e.g. [\"Public\", \"Series D\"]"), "default": []},
                "founded_after": {"type": ["integer", "null"], "description": "Inclusive lower bound on founding year"},
                "founded_before": {"type": ["integer", "null"], "description": "Inclusive upper bound on founding year"},
                "investor": {"type": ["string", "null"], "description": "Investor name substring, e.g. \"Sequoia\""},
                "keywords": {**_str_array("Words that must appear in name/sector/description (any of)"), "default": []},
            },
            ["sectors", "locations", "countries", "stages", "founded_after", "founded_before", "investor", "keywords"],
        ),
    },
    {
        "name": "search_companies",
        "description": (
            "Lexical relevance search (BM25) over company names, sectors and descriptions. Use for vague or "
            "capability-based asks such as 'companies doing supply-chain software for SMEs'. Returns ranked matches "
            "with a short description so you can reason about fit."
        ),
        "strict": True,
        "input_schema": _schema(
            {
                "query": {"type": "string", "description": "Natural-language or keyword query"},
                "limit": {"type": "integer", "description": "Max results (1-20)", "minimum": 1, "maximum": 20},
            },
            ["query", "limit"],
        ),
    },
    {
        "name": CLARIFY_TOOL,
        "description": (
            "Ask the user ONE short clarifying question and end your turn. Use when the request is ambiguous or "
            "underspecified in a way that would change the answer (e.g. 'best company to partner with' without a "
            "domain, a name matching several companies, or 'these' with no prior list in the conversation). "
            "Offer 2-4 concrete options in the question when possible."
        ),
        "strict": True,
        "input_schema": _schema(
            {
                "question": {"type": "string", "description": "The clarifying question to show the user"},
                "reason": {"type": "string", "description": "Internal note: what is ambiguous (not shown to user)"},
            },
            ["question", "reason"],
        ),
    },
]


class ToolExecutor:
    def __init__(self, repo: CompanyRepository):
        self._repo = repo
        self._handlers: dict[str, Callable[[dict], Any]] = {
            "get_company": self._get_company,
            "filter_companies": self._filter_companies,
            "search_companies": self._search_companies,
        }

    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        """Return (json_text, is_error)."""
        handler = self._handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"unknown tool {name}"}), True
        try:
            return json.dumps(handler(args), ensure_ascii=False), False
        except Exception as exc:  # tool bugs must not kill the turn
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True

    # ---- handlers ----------------------------------------------------------
    def _get_company(self, args: dict) -> dict:
        matches = self._repo.find_by_name(str(args.get("name", "")))
        if not matches:
            return {"found": 0, "hint": "No company with a similar name in the dataset. Check the catalogue list."}
        if len(matches) == 1:
            return {"found": 1, "company": matches[0].to_dict()}
        return {"found": len(matches), "candidates": [m.to_dict(compact=True) for m in matches]}

    def _filter_companies(self, args: dict) -> dict:
        rows = self._repo.filter(
            sectors=args.get("sectors") or None,
            locations=args.get("locations") or None,
            countries=args.get("countries") or None,
            stages=args.get("stages") or None,
            founded_after=args.get("founded_after"),
            founded_before=args.get("founded_before"),
            investor=args.get("investor") or None,
            keywords=args.get("keywords") or None,
        )
        items = [
            {**c.to_dict(compact=True), "summary": c.background.split(". ")[0][:160]} for c in rows[:MAX_RESULTS]
        ]
        return {"total": len(rows), "returned": len(items), "companies": items}

    def _search_companies(self, args: dict) -> dict:
        limit = max(1, min(int(args.get("limit", 10)), 20))
        hits = self._repo.search(str(args.get("query", "")), limit=limit)
        return {
            "total": len(hits),
            "companies": [
                {**h.company.to_dict(compact=True), "score": h.score, "description": h.company.background[:400]}
                for h in hits
            ],
        }
