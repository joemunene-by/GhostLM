"""Tests for the domain-aware generalist corpus rebalancer.

These cover the lever that de-specializes GhostLM from a cybersec-only
model: mapping each record source to a coarse training domain, then
capping each domain's token contribution so cybersec stops owning the
corpus while general web / code / math / knowledge carry real share.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.collect import domain_of, rebalance_by_domain
from scripts.rebuild_corpus import (
    CORPUS_PROFILES,
    DEFAULT_EXCLUDE_GLOBS,
    parse_domain_budget,
    select_corpus_sources,
)


# ---------- domain_of ----------

def test_domain_of_explicit_map():
    assert domain_of("nvd") == "cybersec"
    assert domain_of("exploitdb") == "cybersec"
    assert domain_of("security_code") == "cybersec"
    assert domain_of("fineweb_edu") == "general_web"
    assert domain_of("code_corpus") == "code"
    assert domain_of("math_reasoning") == "math"
    assert domain_of("wikipedia") == "knowledge"


def test_domain_of_heuristic_fallback():
    # Unknown sources classify by substring so new collectors need no edit.
    assert domain_of("nvd_2026_topup") == "cybersec"
    assert domain_of("owasp_api_top10") == "cybersec"
    assert domain_of("github_python_extra") == "code"
    assert domain_of("gsm8k_math") == "math"
    assert domain_of("simple_wiki_dump") == "knowledge"
    assert domain_of("c4_web_sample") == "general_web"
    assert domain_of("oasst_chat") == "instruction"


def test_domain_of_unknown_is_other():
    assert domain_of("totally_unrelated") == "other"
    assert domain_of("") == "other"


# ---------- rebalance_by_domain ----------

def _recs(source, n, chars=400):
    return [{"source": source, "text": "x" * chars, "id": f"{source}-{i}"} for i in range(n)]


def test_rebalance_caps_only_budgeted_domain():
    # 100 cyber recs (~10k tok) capped to 4k tok keeps ~40; general untouched.
    recs = _recs("nvd", 100) + _recs("fineweb_edu", 50)
    out = rebalance_by_domain(recs, {"cybersec": 4000})
    cyber = [r for r in out if r["source"] == "nvd"]
    gen = [r for r in out if r["source"] == "fineweb_edu"]
    assert 35 <= len(cyber) <= 45
    assert len(gen) == 50  # uncapped domain passes through whole


def test_rebalance_under_budget_is_noop():
    recs = _recs("nvd", 10)  # ~1k tokens, budget 10k
    out = rebalance_by_domain(recs, {"cybersec": 10000})
    assert len(out) == 10


def test_rebalance_zero_budget_drops_domain():
    recs = _recs("nvd", 10) + _recs("fineweb_edu", 5)
    out = rebalance_by_domain(recs, {"cybersec": 0})
    assert all(r["source"] == "fineweb_edu" for r in out)
    assert len(out) == 5


def test_rebalance_is_deterministic():
    recs = _recs("nvd", 100)
    a = [r["id"] for r in rebalance_by_domain(recs, {"cybersec": 4000})]
    b = [r["id"] for r in rebalance_by_domain(recs, {"cybersec": 4000})]
    assert a == b


def test_rebalance_empty_budget_passes_through():
    recs = _recs("nvd", 5)
    assert rebalance_by_domain(recs, {}) == recs


def test_generalist_profile_caps_cybersec_below_general():
    # The defining property of the generalist pivot: cybersec is no longer
    # allowed the largest budget; general web is.
    gen = CORPUS_PROFILES["generalist"]
    assert gen["cybersec"] < gen["general_web"]
    assert "code" in gen and "math" in gen and "knowledge" in gen


def test_cybersec_profile_has_no_domain_caps():
    assert CORPUS_PROFILES["cybersec"] == {}


# ---------- parse_domain_budget ----------

def test_parse_domain_budget_suffixes():
    out = parse_domain_budget(["cybersec=120m", "code=80M", "math=5b", "knowledge=500k"])
    assert out["cybersec"] == 120_000_000
    assert out["code"] == 80_000_000
    assert out["math"] == 5_000_000_000
    assert out["knowledge"] == 500_000


def test_parse_domain_budget_plain_int_and_empty():
    assert parse_domain_budget(["cybersec=1000"]) == {"cybersec": 1000}
    assert parse_domain_budget(None) == {}
    assert parse_domain_budget([]) == {}


# ---------- select_corpus_sources eval exclusion ----------

def _touch(d, name):
    p = d / name
    p.write_text('{"id":"x","source":"s","text":"y"}\n', encoding="utf-8")
    return p


def test_select_excludes_eval_and_bench_files(tmp_path):
    # Training sources kept; eval/bench rulers dropped.
    _touch(tmp_path, "cve_full.jsonl")
    _touch(tmp_path, "fineweb_edu.jsonl")
    _touch(tmp_path, "code_security_patterns.jsonl")   # synthetic TRAIN bank: keep
    _touch(tmp_path, "code_security_eval.jsonl")        # eval: drop
    _touch(tmp_path, "ctf_eval_bench.jsonl")            # bench: drop
    _touch(tmp_path, "fact_recall_bench_v2.jsonl")      # bench: drop
    _touch(tmp_path, "secqa.jsonl")                     # ruler: drop
    _touch(tmp_path, "general_mcq_bench.jsonl")         # ruler: drop
    sources, _ = select_corpus_sources(tmp_path)
    names = {Path(s).name for s in sources}
    assert "fineweb_edu.jsonl" in names
    assert "code_security_patterns.jsonl" in names
    assert "cve_full.jsonl" in names
    for dropped in ("code_security_eval.jsonl", "ctf_eval_bench.jsonl",
                    "fact_recall_bench_v2.jsonl", "secqa.jsonl",
                    "general_mcq_bench.jsonl"):
        assert dropped not in names, f"{dropped} should be excluded from training"


def test_select_exclusion_can_be_disabled(tmp_path):
    _touch(tmp_path, "fineweb_edu.jsonl")
    _touch(tmp_path, "secqa.jsonl")
    sources, _ = select_corpus_sources(tmp_path, exclude_globs=())
    names = {Path(s).name for s in sources}
    assert "secqa.jsonl" in names  # opt-out restores the old all-globs behaviour


def test_default_exclude_globs_cover_known_rulers():
    assert "secqa.jsonl" in DEFAULT_EXCLUDE_GLOBS
    assert "*_eval.jsonl" in DEFAULT_EXCLUDE_GLOBS
