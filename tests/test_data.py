"""GhostLM data-pipeline unit tests — windowing, source selection, dedup."""

import datetime
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.collect import (
    build_nvd_year_windows,
    collect_capec,
    collect_ctf_repos,
    collect_ctftime_writeups,
    collect_cve_full,
    collect_mitre_attack,
    deduplicate_records,
    load_jsonl,
    parse_ctftime_event_tasks,
    parse_ctftime_task_writeups,
    parse_ctftime_writeup,
)
from scripts.rebuild_corpus import select_corpus_sources


# ---------- build_nvd_year_windows ----------

def test_year_windows_single_year_has_four_chunks():
    """A full year should split into four 119-day windows (Jan-Apr, May-Aug, Sep-Dec, plus a tail)."""
    windows = build_nvd_year_windows(2023, 2023)
    # 365 days / 119-day chunks → 4 windows (the last is shorter)
    assert len(windows) == 4
    # All windows belong to the queried year
    assert all(year == 2023 for _, _, year in windows)


def test_year_windows_no_window_crosses_year_boundary():
    """Each window must be contained within a single calendar year — NVD's date filter
    can otherwise return CVEs published in a different year than the window's end."""
    windows = build_nvd_year_windows(2020, 2024)
    for pub_start, pub_end, year in windows:
        assert pub_start.startswith(f"{year}-")
        assert pub_end.startswith(f"{year}-")


def test_year_windows_max_window_is_119_days():
    """No window can exceed 119 days — NVD's API caps at 120 and returns 404 above that."""
    windows = build_nvd_year_windows(2020, 2024)
    for pub_start, pub_end, _ in windows:
        start = datetime.datetime.fromisoformat(pub_start.replace("T", " "))
        end = datetime.datetime.fromisoformat(pub_end.replace("T", " "))
        assert (end - start).days <= 119


def test_year_windows_cover_full_year_no_gaps():
    """The union of windows in a year must cover Jan 1 through Dec 31 with no gaps."""
    windows = [(s, e) for s, e, y in build_nvd_year_windows(2024, 2024) if y == 2024]
    # First window starts on Jan 1
    assert windows[0][0].startswith("2024-01-01")
    # Last window ends on Dec 31
    assert windows[-1][1].startswith("2024-12-31")
    # Adjacent windows are contiguous (next starts the day after previous ends)
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        prev_day = datetime.date.fromisoformat(prev_end[:10])
        next_day = datetime.date.fromisoformat(next_start[:10])
        assert next_day == prev_day + datetime.timedelta(days=1)


def test_year_windows_inclusive_end_year():
    """end_year is inclusive — querying (2020, 2022) must produce windows for 2022."""
    windows = build_nvd_year_windows(2020, 2022)
    years_seen = {year for _, _, year in windows}
    assert years_seen == {2020, 2021, 2022}


# ---------- select_corpus_sources ----------

def test_source_selection_cve_full_preferred(tmp_path):
    """When both cve.jsonl and cve_full.jsonl exist, cve_full wins by default."""
    (tmp_path / "cve.jsonl").write_text("")
    (tmp_path / "cve_full.jsonl").write_text("")
    (tmp_path / "ctf.jsonl").write_text("")

    sources, cve_choice = select_corpus_sources(tmp_path, prefer_full_nvd=True)
    sources_names = {Path(s).name for s in sources}
    assert "cve_full.jsonl" in sources_names
    assert "cve.jsonl" not in sources_names
    assert "ctf.jsonl" in sources_names
    assert cve_choice == tmp_path / "cve_full.jsonl"


def test_source_selection_legacy_when_full_absent(tmp_path):
    """If cve_full.jsonl is missing, cve.jsonl is selected even with prefer_full_nvd=True."""
    (tmp_path / "cve.jsonl").write_text("")
    (tmp_path / "papers.jsonl").write_text("")

    sources, cve_choice = select_corpus_sources(tmp_path, prefer_full_nvd=True)
    sources_names = {Path(s).name for s in sources}
    assert "cve.jsonl" in sources_names
    assert cve_choice == tmp_path / "cve.jsonl"


def test_source_selection_force_legacy(tmp_path):
    """prefer_full_nvd=False keeps the v0.3.0 baseline corpus reproducible."""
    (tmp_path / "cve.jsonl").write_text("")
    (tmp_path / "cve_full.jsonl").write_text("")

    sources, cve_choice = select_corpus_sources(tmp_path, prefer_full_nvd=False)
    sources_names = {Path(s).name for s in sources}
    assert "cve.jsonl" in sources_names
    assert "cve_full.jsonl" not in sources_names
    assert cve_choice == tmp_path / "cve.jsonl"


def test_source_selection_neither_cve_present(tmp_path):
    """If no CVE file exists, cve_choice is None and only other sources are returned."""
    (tmp_path / "ctf.jsonl").write_text("")
    (tmp_path / "papers.jsonl").write_text("")

    sources, cve_choice = select_corpus_sources(tmp_path)
    assert cve_choice is None
    assert len(sources) == 2


# ---------- collect_cve_full pagination + resume ----------

def _fake_nvd_response(vulns, total_results):
    """Build a MagicMock that mimics requests.get() → JSON with vulnerabilities + totalResults."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "vulnerabilities": vulns,
        "totalResults": total_results,
    }
    return resp


def _fake_cve(cve_id, description="A real-looking vulnerability description for unit tests."):
    """Build a fake NVD vulnerability record matching the API v2.0 shape."""
    return {
        "cve": {
            "id": cve_id,
            "descriptions": [{"lang": "en", "value": description}],
        }
    }


def test_collect_cve_full_paginates_dense_window(tmp_path):
    """A dense window with totalResults > page_size must trigger multiple startIndex requests."""
    out_path = tmp_path / "cve_full.jsonl"

    # Two pages of fake CVEs for the first window, then empty windows for the rest.
    page1 = [_fake_cve(f"CVE-2024-{i:05d}") for i in range(3)]
    page2 = [_fake_cve(f"CVE-2024-{i:05d}") for i in range(3, 5)]

    responses = [
        _fake_nvd_response(page1, total_results=5),  # first window, page 1
        _fake_nvd_response(page2, total_results=5),  # first window, page 2
    ]
    # Subsequent windows return empty
    responses += [_fake_nvd_response([], total_results=0) for _ in range(20)]

    with patch("data.collect.requests.get", side_effect=responses), \
         patch("data.collect.time.sleep"):
        collect_cve_full(
            output_path=str(out_path),
            start_year=2024,
            end_year=2024,
            page_size=3,
            flush_every=1000,
        )

    assert out_path.exists()
    records = load_jsonl(str(out_path))
    ids = {r["id"] for r in records}
    # All 5 CVEs from the dense window must land in the output, proving pagination
    assert ids == {f"CVE-2024-{i:05d}" for i in range(5)}


def test_collect_cve_full_resume_loads_existing(tmp_path):
    """When the output file already has records, a re-run must dedupe by id and not re-fetch them."""
    out_path = tmp_path / "cve_full.jsonl"
    # Pre-seed the output with one existing record
    existing = {"id": "CVE-2024-00001", "text": "previously collected", "source": "nvd"}
    out_path.write_text(json.dumps(existing) + "\n")

    # API now returns the same id (must be skipped) plus a new one (must be kept)
    api_records = [
        _fake_cve("CVE-2024-00001"),  # already in existing, must be deduped
        _fake_cve("CVE-2024-00002"),  # new
    ]
    responses = [_fake_nvd_response(api_records, total_results=2)]
    responses += [_fake_nvd_response([], total_results=0) for _ in range(20)]

    with patch("data.collect.requests.get", side_effect=responses), \
         patch("data.collect.time.sleep"):
        collect_cve_full(
            output_path=str(out_path),
            start_year=2024,
            end_year=2024,
            page_size=2000,
            flush_every=1000,
        )

    records = load_jsonl(str(out_path))
    ids_to_text = {r["id"]: r.get("text") for r in records}
    # Existing record's text is preserved (not overwritten by the API's stub text)
    assert ids_to_text["CVE-2024-00001"] == "previously collected"
    # The new record was appended
    assert "CVE-2024-00002" in ids_to_text


def test_collect_cve_full_drops_short_descriptions(tmp_path):
    """Records with descriptions shorter than 50 chars must be filtered out."""
    out_path = tmp_path / "cve_full.jsonl"

    api_records = [
        _fake_cve("CVE-2024-0001", description="too short"),
        _fake_cve("CVE-2024-0002", description="A genuinely long-enough vulnerability description for the test."),
    ]
    responses = [_fake_nvd_response(api_records, total_results=2)]
    responses += [_fake_nvd_response([], total_results=0) for _ in range(20)]

    with patch("data.collect.requests.get", side_effect=responses), \
         patch("data.collect.time.sleep"):
        collect_cve_full(
            output_path=str(out_path),
            start_year=2024,
            end_year=2024,
            page_size=2000,
            flush_every=1000,
        )

    ids = {r["id"] for r in load_jsonl(str(out_path))}
    assert ids == {"CVE-2024-0002"}


# ---------- deduplicate_records ----------

def test_dedup_collapses_byte_identical_texts():
    """Two records with the same text must be deduped by content hash."""
    records = [
        {"id": "a", "text": "Same thing"},
        {"id": "b", "text": "Same thing"},
        {"id": "c", "text": "Different thing"},
    ]
    unique = deduplicate_records(records)
    texts = [r["text"] for r in unique]
    assert len(unique) == 2
    assert "Same thing" in texts
    assert "Different thing" in texts


def test_dedup_normalizes_whitespace():
    """Records that differ only in whitespace must be treated as duplicates."""
    records = [
        {"id": "a", "text": "Hello world"},
        {"id": "b", "text": "  Hello  world  "},
    ]
    unique = deduplicate_records(records)
    assert len(unique) == 1


# ---------- collect_ctf_repos ----------

def _stub_clone(repo_dir, files, license_filename="LICENSE"):
    """Lay out a fake clone: ``files`` is {relative_path: content}."""
    repo_dir = Path(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    if license_filename:
        (repo_dir / license_filename).write_text("Permissive license placeholder.")
    for rel, content in files.items():
        path = repo_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _patched_clone(files_per_url, license_filename="LICENSE"):
    """Build a side_effect that lays out files based on the clone target URL."""
    def fake_run(cmd, *args, **kwargs):
        # cmd is ["git", "clone", "--depth", "1", url, dest]
        url = cmd[-2]
        dest = cmd[-1]
        files = files_per_url.get(url, {})
        _stub_clone(dest, files, license_filename=license_filename)
        return MagicMock(returncode=0, stdout="", stderr="")
    return fake_run


def test_ctf_repos_extracts_markdown_with_metadata(tmp_path):
    """Markdown files inside a cloned repo are emitted with repo + license tags."""
    out = tmp_path / "ctf_repos.jsonl"
    repos = [
        {"url": "https://github.com/team-a/writeups", "license": "MIT"},
    ]
    files = {
        "2024/web/sql-injection.md": "# SQL Injection writeup\n\n" + ("Detailed payload analysis. " * 30),
        "2024/pwn/buffer-overflow.md": "# Buffer overflow writeup\n\n" + ("ROP chain explanation. " * 30),
        "README.md": "Top-level readme.",  # short — should be filtered out by min_chars
    }
    with patch("data.collect.subprocess.run",
               side_effect=_patched_clone({repos[0]["url"]: files})):
        collect_ctf_repos(repos, output_path=str(out), min_chars=200, max_chars=12000)

    records = load_jsonl(str(out))
    paths = {r["path"] for r in records}
    # README is too short (~16 chars) → dropped; both writeups kept
    assert paths == {"2024/web/sql-injection.md", "2024/pwn/buffer-overflow.md"}
    assert all(r["repo"] == "https://github.com/team-a/writeups" for r in records)
    assert all(r["license"] == "MIT" for r in records)
    assert all(r["license_file_present"] is True for r in records)
    assert all(r["source"] == "ctf_repos" for r in records)


def test_ctf_repos_truncates_oversized_files(tmp_path):
    """Files longer than max_chars are truncated, not dropped."""
    out = tmp_path / "ctf_repos.jsonl"
    repos = [{"url": "https://github.com/team-b/writeups", "license": "CC-BY-4.0"}]
    files = {"huge.md": "A" * 50000}  # way past max_chars
    with patch("data.collect.subprocess.run",
               side_effect=_patched_clone({repos[0]["url"]: files})):
        collect_ctf_repos(repos, output_path=str(out), min_chars=10, max_chars=12000)

    records = load_jsonl(str(out))
    assert len(records) == 1
    assert len(records[0]["text"]) == 12000


def test_ctf_repos_subdir_scopes_walk(tmp_path):
    """When a subdir is set, only files inside it are collected."""
    out = tmp_path / "ctf_repos.jsonl"
    repos = [{
        "url": "https://github.com/team-c/writeups",
        "license": "MIT",
        "subdir": "2024",
    }]
    files = {
        "2024/web/inside.md": "Inside the subdir. " * 30,
        "2023/pwn/outside.md": "Outside the subdir. " * 30,
    }
    with patch("data.collect.subprocess.run",
               side_effect=_patched_clone({repos[0]["url"]: files})):
        collect_ctf_repos(repos, output_path=str(out), min_chars=100, max_chars=12000)

    records = load_jsonl(str(out))
    paths = {r["path"] for r in records}
    assert "2024/web/inside.md" in paths
    assert "2023/pwn/outside.md" not in paths


def test_ctf_repos_flags_missing_license_file(tmp_path):
    """Records carry license_file_present=False when no LICENSE is in the repo."""
    out = tmp_path / "ctf_repos.jsonl"
    repos = [{"url": "https://github.com/team-d/writeups", "license": "MIT"}]
    files = {"writeup.md": "Some real-looking writeup content. " * 20}
    with patch("data.collect.subprocess.run",
               side_effect=_patched_clone({repos[0]["url"]: files}, license_filename=None)):
        collect_ctf_repos(repos, output_path=str(out), min_chars=100, max_chars=12000)

    records = load_jsonl(str(out))
    assert len(records) == 1
    assert records[0]["license_file_present"] is False


def test_ctf_repos_skips_failed_clone(tmp_path):
    """A failed clone for one repo doesn't block collection from the others."""
    import subprocess as sp
    out = tmp_path / "ctf_repos.jsonl"
    repos = [
        {"url": "https://example.invalid/broken", "license": "MIT"},
        {"url": "https://github.com/team-e/writeups", "license": "MIT"},
    ]
    good_files = {"writeup.md": "A real writeup. " * 30}

    def fake_run(cmd, *args, **kwargs):
        url = cmd[-2]
        if url == repos[0]["url"]:
            raise sp.CalledProcessError(1, cmd, output="", stderr="fatal: repository not found\n")
        # second repo: lay out files
        _stub_clone(cmd[-1], good_files)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("data.collect.subprocess.run", side_effect=fake_run):
        collect_ctf_repos(repos, output_path=str(out), min_chars=100, max_chars=12000)

    records = load_jsonl(str(out))
    assert len(records) == 1
    assert records[0]["repo"] == "https://github.com/team-e/writeups"


def test_ctf_repos_empty_input_is_noop(tmp_path):
    """No repos = no output file, no crash."""
    out = tmp_path / "ctf_repos.jsonl"
    collect_ctf_repos([], output_path=str(out))
    assert not out.exists()


# ---------- collect_mitre_attack ----------

def _stix_attack_bundle(techniques):
    """Build a fake MITRE ATT&CK STIX 2.1 bundle from a list of (id, name, description, tactics)."""
    objects = []
    for ext_id, name, desc, tactics in techniques:
        objects.append({
            "type": "attack-pattern",
            "name": name,
            "description": desc,
            "external_references": [{"source_name": "mitre-attack", "external_id": ext_id}],
            "kill_chain_phases": [
                {"kill_chain_name": "mitre-attack", "phase_name": t} for t in tactics
            ],
        })
    return {"objects": objects}


def test_mitre_attack_extracts_techniques(tmp_path):
    """Real STIX-shaped JSON yields one record per attack-pattern with id, tactics, description."""
    out = tmp_path / "mitre_attack.jsonl"
    bundle = _stix_attack_bundle([
        ("T1059", "Command and Scripting Interpreter",
         "Adversaries may abuse command and script interpreters to execute commands. " * 5,
         ["execution"]),
        ("T1078", "Valid Accounts",
         "Adversaries may obtain and abuse credentials of existing accounts. " * 5,
         ["defense-evasion", "persistence", "privilege-escalation", "initial-access"]),
    ])
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = bundle

    with patch("data.collect.requests.get", return_value=resp):
        collect_mitre_attack(output_path=str(out), max_records=100)

    records = load_jsonl(str(out))
    assert {r["id"] for r in records} == {"T1059", "T1078"}
    assert all(r["source"] == "mitre_attack" for r in records)
    # Tactic phases land in the rendered text
    t1078 = next(r for r in records if r["id"] == "T1078")
    assert "persistence" in t1078["text"]
    # Technique IDs aren't double-prefixed
    assert "MITRE ATT&CK Technique T1078" in t1078["text"]


def test_mitre_attack_skips_revoked_and_deprecated(tmp_path):
    """Records flagged revoked or x_mitre_deprecated must be dropped."""
    out = tmp_path / "mitre_attack.jsonl"
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "name": "Active",
                "description": "A real, non-revoked technique. " * 10,
                "external_references": [{"source_name": "mitre-attack", "external_id": "T0001"}],
                "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "execution"}],
            },
            {
                "type": "attack-pattern",
                "name": "Revoked technique",
                "description": "Should be dropped because revoked. " * 10,
                "revoked": True,
                "external_references": [{"source_name": "mitre-attack", "external_id": "T9998"}],
            },
            {
                "type": "attack-pattern",
                "name": "Deprecated technique",
                "description": "Should be dropped because deprecated. " * 10,
                "x_mitre_deprecated": True,
                "external_references": [{"source_name": "mitre-attack", "external_id": "T9999"}],
            },
            # non-attack-pattern object — must be ignored entirely
            {"type": "course-of-action", "name": "Mitigation", "description": "irrelevant"},
        ]
    }
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = bundle

    with patch("data.collect.requests.get", return_value=resp):
        collect_mitre_attack(output_path=str(out), max_records=100)

    records = load_jsonl(str(out))
    assert {r["id"] for r in records} == {"T0001"}


# ---------- collect_capec ----------

def _stix_capec_bundle(patterns):
    """Build a fake CAPEC STIX bundle from a list of (id, name, description)."""
    return {
        "objects": [
            {
                "type": "attack-pattern",
                "name": name,
                "description": desc,
                "external_references": [{"source_name": "capec", "external_id": cap_id}],
            }
            for cap_id, name, desc in patterns
        ]
    }


def test_capec_label_not_double_prefixed(tmp_path):
    """Regression: CAPEC text must not start with 'CAPEC CAPEC-N' (the id is already 'CAPEC-N')."""
    out = tmp_path / "capec.jsonl"
    bundle = _stix_capec_bundle([
        ("CAPEC-1", "Accessing Functionality Not Properly Constrained by ACLs",
         "Access control lists describe access rights. " * 10),
    ])
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = bundle

    with patch("data.collect.requests.get", return_value=resp):
        collect_capec(output_path=str(out), max_records=100)

    records = load_jsonl(str(out))
    assert len(records) == 1
    text = records[0]["text"]
    assert text.startswith("CAPEC-1:")
    assert "CAPEC CAPEC-1" not in text


def test_capec_extracts_patterns_with_external_ids(tmp_path):
    """Each attack-pattern with a CAPEC external_id becomes one record keyed by id."""
    out = tmp_path / "capec.jsonl"
    bundle = _stix_capec_bundle([
        ("CAPEC-66", "SQL Injection", "An attacker exploits SQL queries by injecting input. " * 10),
        ("CAPEC-100", "Overflow Buffers", "Buffer overflow attacks target unchecked bounds. " * 10),
    ])
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = bundle

    with patch("data.collect.requests.get", return_value=resp):
        collect_capec(output_path=str(out), max_records=100)

    records = load_jsonl(str(out))
    assert {r["id"] for r in records} == {"CAPEC-66", "CAPEC-100"}
    assert all(r["source"] == "capec" for r in records)


# ---------- CTFtime parsers + collector ----------

def _ctftime_writeup_html(body: str = "Step one: ssh into the box.\n```\n$ ssh user@host\n```\nFlag: FLAG{x}",
                          team: str = "TestTeam",
                          rating: str = "4.5",
                          event_id: int = 1405,
                          event_name: str = "FwordCTF 2021",
                          task_id: int = 17065,
                          task_name: str = "devprivops",
                          original_url: str = "https://example.com/orig"):
    """Render a minimal CTFtime writeup page mirroring the live HTML structure."""
    body_html = body.replace("\n", "<br />")
    orig_anchor = (
        f'<a href="{original_url}" target="_new" rel="nofollow noopener">Original writeup</a>'
        if original_url else ""
    )
    return f"""<html><body>
<ul class="breadcrumb">
  <li><a href="/">Home</a> <span class="divider">/</span></li>
  <li><a href="/event/list/">CTF events</a> <span class="divider">/</span></li>
  <li><a href="/event/{event_id}">{event_name}</a> <span class="divider">/</span></li>
  <li><a href="/event/{event_id}/tasks/">Tasks</a></li> <span class="divider">/</span></li>
  <li><a href="/task/{task_id}">{task_name}</a> <span class="divider">/</span></li>
  <li class="active">Writeup</li>
</ul>
<div class="page-header"><h2>{task_name}</h2>
by <a href="/team/159663">{team}</a></div>
<div class="row">
  <div class="span7">
    <p>Rating: <span id="user_rating" class="category-value">{rating}</span></p>
  </div>
  <div class="span4"></div>
</div>
<div class="well" id="id_description">
<p>{body_html}</p>
</div>
<div class="page-header"><h3>Comments</h3></div>
<p>Note: {orig_anchor}.</p>
</body></html>"""


def test_ctftime_parse_event_tasks_returns_unique_ids():
    """Tasks page links should be deduplicated and returned sorted as ints."""
    html = """
    <a href="/task/100">Foo</a>
    <a href="/task/200">Bar</a>
    <a href="/task/100">Foo (duplicate link)</a>
    <a href="/event/2230">unrelated event link</a>
    """
    assert parse_ctftime_event_tasks(html) == [100, 200]


def test_ctftime_parse_task_writeups_returns_unique_ids():
    """Task page links should yield deduped writeup IDs."""
    html = """
    <a href="/writeup/30000">Writeup A</a>
    <a href="/writeup/38909">Writeup B</a>
    <a href="/writeup/30000">repeat</a>
    """
    assert parse_ctftime_task_writeups(html) == [30000, 38909]


def test_ctftime_parse_writeup_extracts_metadata_and_body():
    """A full writeup page should yield body + breadcrumb metadata + original link."""
    html = _ctftime_writeup_html()
    rec = parse_ctftime_writeup(html)
    assert rec is not None
    assert rec["task_name"] == "devprivops"
    assert rec["team"] == "TestTeam"
    assert rec["rating"] == "4.5"
    assert rec["event_id"] == 1405
    assert rec["event_name"] == "FwordCTF 2021"
    assert rec["task_id"] == 17065
    assert rec["original_url"] == "https://example.com/orig"
    assert "ssh into the box" in rec["body"]
    assert "$ ssh user@host" in rec["body"]
    # HTML tags must not survive the body extraction
    assert "<br" not in rec["body"]
    assert "<p>" not in rec["body"]


def test_ctftime_parse_writeup_returns_none_when_no_inline_body():
    """Pages without an id_description container (external-only redirects) are skipped."""
    html = """<html><body>
    <div class="page-header"><h2>External</h2></div>
    <p>This writeup is hosted off-site: <a href="https://example.com/blog">go here</a>.</p>
    </body></html>"""
    assert parse_ctftime_writeup(html) is None


def test_ctftime_parse_writeup_unescapes_entities():
    """HTML entities in the body must round-trip back to their character form."""
    html = _ctftime_writeup_html(body="if a &lt; b &amp;&amp; c &gt; d: pass")
    rec = parse_ctftime_writeup(html)
    assert rec is not None
    assert "if a < b && c > d" in rec["body"]


def test_ctftime_parse_writeup_unescapes_title_entities():
    """Title HTML entities (e.g. & rendered as &amp;) must be decoded — real
    CTFtime task names contain ampersands and other entities."""
    html = _ctftime_writeup_html(task_name="Peaky &amp; the Brain")
    rec = parse_ctftime_writeup(html)
    assert rec is not None
    assert rec["task_name"] == "Peaky & the Brain"


def test_ctftime_parse_writeup_extracts_team_with_user_link_first():
    """When the page-header has 'by <a href=/user/...> / <a href=/team/...>',
    the team must still be extracted — the user link can't shadow it."""
    html = """<html><body>
<ul class="breadcrumb">
  <li><a href="/event/1405">FwordCTF 2021</a></li>
  <li><a href="/task/17065">devprivops</a></li>
</ul>
<div class="page-header">
<h2>devprivops</h2>
by <a href="/user/103712">someuser_</a> / <a href="/team/132008">RootMeUpBeforeYouGoGo</a>
</div>
<div class="well" id="id_description">
<p>body content here that is long enough to keep</p>
</div>
</body></html>"""
    rec = parse_ctftime_writeup(html)
    assert rec is not None
    assert rec["team"] == "RootMeUpBeforeYouGoGo"


def test_ctftime_parse_writeup_handles_empty_rating():
    """Unrated writeups render <span id=user_rating ...></span> with no inner
    text — the parser must accept that and return rating='' rather than crash
    or leave the field unset."""
    html = """<html><body>
<ul class="breadcrumb">
  <li><a href="/event/1405">FwordCTF 2021</a></li>
  <li><a href="/task/17065">t</a></li>
</ul>
<div class="page-header"><h2>t</h2>
by <a href="/team/1">x</a></div>
<p>Rating: <span id="user_rating" class="category-value"></span></p>
<div class="well" id="id_description"><p>some inline body content</p></div>
</body></html>"""
    rec = parse_ctftime_writeup(html)
    assert rec is not None
    assert rec["rating"] == ""


def test_ctftime_collector_skips_already_collected_writeups(tmp_path):
    """Resume mode: a writeup_id present in output_path should not trigger a re-fetch."""
    out = tmp_path / "ctftime.jsonl"
    # Pre-seed output with one already-collected writeup
    pre = [{
        "id": "ctftime-30000",
        "text": "previously collected body",
        "source": "ctftime",
        "ctftime_url": "https://ctftime.org/writeup/30000",
        "writeup_id": 30000,
    }]
    with open(out, "w", encoding="utf-8") as f:
        for r in pre:
            f.write(json.dumps(r) + "\n")

    # Mock fetch order: tasks-page → task-page (yielding writeups 30000 + 30001)
    # The collector should only fetch /writeup/30001 (30000 is already on disk).
    tasks_html = '<a href="/task/17065">devprivops</a>'
    task_html = '<a href="/writeup/30000">old</a> <a href="/writeup/30001">new</a>'
    new_writeup_html = _ctftime_writeup_html(
        body="brand new exploit narrative " * 20,
        task_id=17065,
    )

    fetched_paths = []

    def fake_get(url, headers=None, timeout=None):
        fetched_paths.append(url)
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if url.endswith("/event/1405/tasks/"):
            resp.text = tasks_html
        elif url.endswith("/task/17065"):
            resp.text = task_html
        elif url.endswith("/writeup/30001"):
            resp.text = new_writeup_html
        else:
            # If the collector tries to fetch /writeup/30000 it would hit
            # this branch — we want the test to fail in that case.
            resp.text = ""
            raise AssertionError(f"unexpected fetch: {url}")
        return resp

    with patch("data.collect.requests.get", side_effect=fake_get), \
         patch("data.collect.time.sleep"):
        collect_ctftime_writeups(
            event_ids=[1405],
            output_path=str(out),
            request_delay=0,
        )

    # Both records (existing + new) should now be on disk
    records = load_jsonl(str(out))
    ids = {r["writeup_id"] for r in records}
    assert ids == {30000, 30001}
    # The pre-seeded body must still be there
    old = next(r for r in records if r["writeup_id"] == 30000)
    assert old["text"] == "previously collected body"
    # The new record carries the standard CTFtime metadata
    new = next(r for r in records if r["writeup_id"] == 30001)
    assert new["source"] == "ctftime"
    assert new["license"] == "ctftime-user-submitted"
    assert new["event_id"] == 1405
    assert "brand new exploit narrative" in new["text"]
    # /writeup/30000 must not have been fetched
    assert not any(p.endswith("/writeup/30000") for p in fetched_paths)
