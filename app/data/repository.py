"""In-memory query layer over the company dataset.

Two retrieval modes are exposed to the LLM as tools:
  * structured filtering (exact-ish match on sector / city / stage / year / investor)
  * lexical BM25 search over name + sector + description for "fuzzy" asks

The dataset is 118 rows, so everything is computed in-process with no external
index. BM25 is implemented inline to avoid a dependency for a 100 KB corpus.
"""
from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from dataclasses import dataclass

from app.data.loader import Company

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "it", "its", "is",
    "are", "that", "this", "as", "by", "at", "from", "which", "also", "company", "provider",
    "platform", "offers", "offering", "provides", "services", "service", "india", "indian",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class SearchHit:
    company: Company
    score: float


class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.doc_len = [len(d) for d in docs]
        self.avg_len = sum(self.doc_len) / max(len(docs), 1)
        self.tf = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: list[str], idx: int) -> float:
        tf, dl = self.tf[idx], self.doc_len[idx]
        s = 0.0
        for t in query:
            if t not in tf:
                continue
            f = tf[t]
            s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg_len))
        return s


class CompanyRepository:
    def __init__(self, companies: list[Company]):
        self._companies = list(companies)
        self._by_lower_name = {c.name.lower(): c for c in self._companies}
        self._bm25 = _BM25([tokenize(c.search_text) for c in self._companies])

    # ---- catalogue -------------------------------------------------------
    @property
    def companies(self) -> list[Company]:
        return list(self._companies)

    def __len__(self) -> int:
        return len(self._companies)

    def distinct(self, field: str) -> list[str]:
        values = {getattr(c, field) for c in self._companies if getattr(c, field)}
        return sorted(values)

    # ---- exact / fuzzy lookup --------------------------------------------
    def get(self, name: str) -> Company | None:
        return self._by_lower_name.get(name.strip().lower())

    def find_by_name(self, name: str, limit: int = 5) -> list[Company]:
        """Exact match first, then substring, then fuzzy (difflib) candidates."""
        q = name.strip().lower()
        if not q:
            return []
        exact = self._by_lower_name.get(q)
        if exact:
            return [exact]
        compact_q = re.sub(r"[^a-z0-9]", "", q)
        substr = [
            c for c in self._companies
            if q in c.name.lower() or compact_q and compact_q in re.sub(r"[^a-z0-9]", "", c.name.lower())
        ]
        if substr:
            return substr[:limit]
        close = difflib.get_close_matches(q, list(self._by_lower_name), n=limit, cutoff=0.7)
        return [self._by_lower_name[c] for c in close]

    # ---- structured filter -----------------------------------------------
    def filter(
        self,
        sectors: list[str] | None = None,
        locations: list[str] | None = None,
        countries: list[str] | None = None,
        stages: list[str] | None = None,
        founded_after: int | None = None,
        founded_before: int | None = None,
        investor: str | None = None,
        keywords: list[str] | None = None,
    ) -> list[Company]:
        def norm(xs: list[str] | None) -> set[str]:
            return {x.strip().lower() for x in xs or [] if x and x.strip()}

        sectors_n, locs_n, countries_n, stages_n = norm(sectors), norm(locations), norm(countries), norm(stages)
        investor_n = (investor or "").strip().lower()
        kw_n = [k.strip().lower() for k in keywords or [] if k and k.strip()]

        out = []
        for c in self._companies:
            if sectors_n and not any(s == c.sector.lower() or s in c.sector.lower() for s in sectors_n):
                continue
            if locs_n and not any(l == c.location.lower() or l in c.location.lower() for l in locs_n):
                continue
            if countries_n and c.country.lower() not in countries_n:
                continue
            if stages_n and c.stage.lower() not in stages_n:
                continue
            if founded_after is not None and (c.founded_year is None or c.founded_year < founded_after):
                continue
            if founded_before is not None and (c.founded_year is None or c.founded_year > founded_before):
                continue
            if investor_n and not any(investor_n in inv.lower() for inv in (*c.top_investors, c.round_led_by)):
                continue
            if kw_n:
                hay = c.search_text.lower()
                if not any(k in hay for k in kw_n):
                    continue
            out.append(c)
        return out

    # ---- lexical search --------------------------------------------------
    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        q = tokenize(query)
        if not q:
            return []
        scored = [(self._bm25.score(q, i), c) for i, c in enumerate(self._companies)]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda sc: (-sc[0], sc[1].rank))
        return [SearchHit(company=c, score=round(s, 3)) for s, c in scored[:limit]]
