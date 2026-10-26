from app.chatbot.prompts import build_system_prompt
from app.chatbot.tools import CLARIFY_TOOL, TOOL_DEFINITIONS, ToolExecutor
import json


def test_dataset_loads_and_is_clean(repo):
    assert len(repo) == 118
    juspay = repo.get("Juspay")
    assert juspay.location == "Bengaluru"  # trailing nbsp removed
    assert juspay.annual_revenue.startswith("₹348Cr")  # '?' restored to rupee sign
    assert "led by undisclosed investors" in juspay.unicorn_entry
    assert not repo.get("Rapido").background.endswith("?")


def test_name_lookup_exact_substring_fuzzy(repo):
    assert [c.name for c in repo.find_by_name("razorpay")] == ["Razorpay"]
    assert [c.name for c in repo.find_by_name("razor")] == ["Razorpay"]
    assert [c.name for c in repo.find_by_name("byjus")] == ["BYJU'S"]
    assert {c.name for c in repo.find_by_name("Paytm")} == {"Paytm"}
    assert repo.find_by_name("Microsoft") == []


def test_structured_filter(repo):
    fintech_blr = repo.filter(sectors=["Payments", "Alternative Lending"], locations=["Bengaluru"])
    assert {c.name for c in fintech_blr} >= {"Juspay", "Money View", "Razorpay"}
    assert all(c.location == "Bengaluru" for c in fintech_blr)
    assert all(c.founded_year >= 2018 for c in repo.filter(founded_after=2018))
    assert {c.name for c in repo.filter(investor="Kedaara")} >= {"Juspay"}
    assert repo.filter(stages=["Public"], countries=["India"])


def test_bm25_search_ranks_relevant_first(repo):
    hits = repo.search("bike taxi and auto rickshaw rides")
    assert hits[0].company.name == "Rapido"
    assert repo.search("") == []


def test_tool_executor_outputs(repo):
    ex = ToolExecutor(repo)
    out, err = ex.execute("get_company", {"name": "Zerodha"})
    assert not err and json.loads(out)["company"]["sector"]
    out, err = ex.execute("get_company", {"name": "Nope Inc"})
    assert not err and json.loads(out)["found"] == 0
    out, err = ex.execute("filter_companies", {"sectors": ["Payments"], "locations": [], "countries": [], "stages": [], "founded_after": None, "founded_before": None, "investor": None, "keywords": []})
    assert not err and json.loads(out)["total"] > 3
    out, err = ex.execute("search_companies", {"query": "electric scooters", "limit": 3})
    assert not err and json.loads(out)["companies"][0]["name"] in {"Ather Energy", "Ola Electric"}
    out, err = ex.execute("does_not_exist", {})
    assert err


def test_tool_schemas_are_strict():
    names = {t["name"] for t in TOOL_DEFINITIONS}
    assert CLARIFY_TOOL in names
    for t in TOOL_DEFINITIONS:
        assert t["strict"] is True
        assert t["input_schema"]["additionalProperties"] is False
        assert set(t["input_schema"]["required"]) == set(t["input_schema"]["properties"])


def test_system_prompt_contains_catalogue(repo):
    sp = build_system_prompt(repo)
    assert "Razorpay" in sp and "Distinct sectors" in sp
    assert len(sp) < 20000  # stays cheap enough to cache
