"""Load and clean the Kaggle 'List of 118 Unicorn Startups in India (May 2025)' CSV.

The raw file (exported from Tracxn) is cp1252-encoded, uses '?' where the rupee
sign was lost, and has trailing non-breaking spaces in several columns. All of
that is normalised here so the rest of the app sees clean, typed records.
"""
from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

_WS = re.compile(r"\s+")


def _clean(value: str | None) -> str:
    if value is None:
        return ""
    value = value.replace("\xa0", " ").replace("�", "")
    value = _WS.sub(" ", value).strip()
    # Tracxn exports lose the rupee glyph; a leading '?' before a number is ₹.
    value = re.sub(r"\?(?=\s?[\d,])", "₹", value)
    # Dangling '?' at the end of prose is an encoding artefact, not a question.
    value = re.sub(r"\s*\?$", "", value) if not value.endswith("??") else value
    # "led by , ," -> "led by undisclosed investors"
    value = re.sub(r"led by\s*(,\s*)*\.", "led by undisclosed investors.", value)
    value = re.sub(r"led by\s*(,\s*)+", "led by undisclosed investors ", value)
    return value


def _to_int(value: str) -> int | None:
    digits = re.sub(r"[^\d]", "", value.split(" as on")[0]) if value else ""
    return int(digits) if digits else None


@dataclass(frozen=True)
class Company:
    rank: int
    name: str
    sector: str
    location: str
    country: str
    stage: str
    founded_year: int | None
    background: str
    unicorn_entry: str
    time_to_unicorn: str
    round_led_by: str
    top_investors: tuple[str, ...]
    annual_revenue: str
    valuation: str
    total_funding: str
    employee_count: str
    latest_funding_round: str
    source_url: str
    employees: int | None = field(default=None, compare=False)

    def to_dict(self, compact: bool = False) -> dict:
        if compact:
            return {
                "name": self.name,
                "sector": self.sector,
                "location": self.location,
                "country": self.country,
                "stage": self.stage,
                "founded_year": self.founded_year,
            }
        d = asdict(self)
        d["top_investors"] = list(self.top_investors)
        d.pop("employees", None)
        return d

    @property
    def search_text(self) -> str:
        return f"{self.name}. {self.sector}. {self.location}. {self.background}"


def load_companies(path: str | Path) -> list[Company]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. See README for download steps.")
    companies: list[Company] = []
    with path.open(encoding="cp1252", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            row = {k.strip(): _clean(v) for k, v in row.items() if k}
            investors = tuple(i for i in (row.get("top_investors_1"), row.get("top_investors_2")) if i)
            companies.append(
                Company(
                    rank=int(row["Rank"]),
                    name=row["Company"],
                    sector=row["primary_sector"],
                    location=row["location"],
                    country=row["country"],
                    stage=row["stage"],
                    founded_year=_to_int(row.get("founded_year", "")),
                    background=row["company_background"],
                    unicorn_entry=row["company_info"],
                    time_to_unicorn=row["time_to_unicorn"],
                    round_led_by=row.get("round_led_by", ""),
                    top_investors=investors,
                    annual_revenue=row.get("annual_revenue", ""),
                    valuation=row.get("valuation", ""),
                    total_funding=row.get("total_funding_till_date", ""),
                    employee_count=row.get("employee_count", ""),
                    latest_funding_round=row.get("latest_funding_round", ""),
                    source_url=row.get("company_link", ""),
                    employees=_to_int(row.get("employee_count", "")),
                )
            )
    companies.sort(key=lambda c: c.rank)
    return companies
