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
    _is_metasploit_module,
    build_nvd_year_windows,
    collect_capec,
    collect_ctf_repos,
    collect_ctftime_writeups,
    collect_cve_full,
    collect_exploitdb,
    collect_mitre_attack,
    deduplicate_records,
    load_jsonl,
    merge_datasets,
    parse_ctftime_event_tasks,
    parse_ctftime_task_writeups,
    parse_ctftime_writeup,
    subsample_cve_records,
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


# ---------- subsample_cve_records ----------

def _make_cve(cid, char_count):
    """Build a CVE record with a given text length and unique content.
    Including the CVE ID in the text guarantees each record has a distinct
    md5, mirroring real-world CVE descriptions which are never identical."""
    pad = "A" * max(0, char_count - len(cid) - 1)
    return {"id": cid, "text": f"{cid} {pad}", "source": "nvd"}


def test_subsample_caps_cve_chars_at_target():
    """A CVE pool that overshoots the budget should be cut to the prefix that
    just covers it. Char count after the cap is roughly token_budget * 4 (the
    last record can push us over by its size, but never under)."""
    cves = [_make_cve(f"CVE-2024-{i:04d}", 1000) for i in range(100)]
    others = [{"id": "P-1", "text": "paper", "source": "papers"}]
    out = subsample_cve_records(cves + others, max_cve_tokens=5000)

    out_cve = [r for r in out if r["source"] == "nvd"]
    out_other = [r for r in out if r["source"] == "papers"]

    # Budget is 5000 tokens = 20000 chars. Each record is 1000 chars,
    # so we expect exactly 20 records.
    assert len(out_cve) == 20
    # Non-CVE records are passed through untouched
    assert out_other == others


def test_subsample_no_op_when_already_under_budget():
    """If total CVE chars are already within budget, the helper returns the
    input unchanged — no records dropped or reordered."""
    cves = [_make_cve(f"CVE-2024-{i:04d}", 1000) for i in range(10)]
    others = [{"id": "P-1", "text": "paper", "source": "papers"}]
    inp = cves + others
    # 10 * 1000 = 10000 chars = ~2500 tokens; budget is 100000 tokens
    out = subsample_cve_records(inp, max_cve_tokens=100000)
    assert out is inp  # same object — full passthrough


def test_subsample_no_op_when_no_cve_records():
    """If the corpus has no NVD records at all, the helper is a no-op even
    when a budget is set."""
    others = [
        {"id": "P-1", "text": "paper", "source": "papers"},
        {"id": "M-1", "text": "mitre", "source": "mitre_attack"},
    ]
    out = subsample_cve_records(others, max_cve_tokens=100)
    assert out is others


def test_subsample_is_deterministic():
    """Two runs over the same input must keep the same CVE prefix.
    Determinism is what makes train/val splits reproducible after a rebuild."""
    cves = [_make_cve(f"CVE-2024-{i:04d}", 1000) for i in range(50)]
    others = [{"id": "P-1", "text": "paper", "source": "papers"}]

    a = subsample_cve_records(cves + others, max_cve_tokens=2500)
    b = subsample_cve_records(cves + others, max_cve_tokens=2500)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_subsample_independent_of_input_order():
    """Reordering the input list should not change which CVE records are kept
    (the sort is by content hash, not by index). Otherwise subsampling would
    silently depend on whichever file got loaded first."""
    cves = [_make_cve(f"CVE-2024-{i:04d}", 1000) for i in range(50)]
    others = [{"id": "P-1", "text": "paper", "source": "papers"}]

    a = subsample_cve_records(cves + others, max_cve_tokens=2500)
    b = subsample_cve_records(others + list(reversed(cves)), max_cve_tokens=2500)
    assert sorted(r["id"] for r in a if r["source"] == "nvd") == \
           sorted(r["id"] for r in b if r["source"] == "nvd")


def test_merge_datasets_applies_cve_subsample(tmp_path):
    """End-to-end: merge_datasets with max_cve_tokens should reflect the cap
    in the resulting train+val output, while non-CVE sources land intact."""
    cve_path = tmp_path / "cve_full.jsonl"
    other_path = tmp_path / "papers.jsonl"

    # 200 CVE records of 1000 chars each = 200K chars = 50K tokens.
    # Budget of 5000 tokens = 20K chars = 20 records.
    with open(cve_path, "w", encoding="utf-8") as f:
        for i in range(200):
            f.write(json.dumps({"id": f"CVE-2024-{i:04d}",
                                "text": "A" * 1000 + f" {i}",
                                "source": "nvd"}) + "\n")
    with open(other_path, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"id": f"PAPER-{i}",
                                "text": "research abstract " * 20 + f"{i}",
                                "source": "papers"}) + "\n")

    out_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    merge_datasets(
        input_paths=[str(cve_path), str(other_path)],
        output_path=str(out_path),
        val_split=0.05,
        max_cve_tokens=5000,
    )

    # Count across both splits — the deterministic-hash split routes a
    # small fraction to val so we can't just count train.
    all_kept = load_jsonl(str(out_path)) + load_jsonl(str(val_path))
    cve_kept = [r for r in all_kept if r["source"] == "nvd"]
    other_kept = [r for r in all_kept if r["source"] == "papers"]

    # ~20 CVE records survive the cap; non-CVE pass through fully
    assert 18 <= len(cve_kept) <= 22  # boundary record may push us slightly over
    assert len(other_kept) == 5


# ---------- collect_exploitdb ----------

def _stub_exploitdb_mirror(mirror_path: Path, csv_rows, files):
    """Lay out a fake Exploit-DB mirror at ``mirror_path``.

    Pre-creates a ``.git`` directory so the mirror probe in
    ``_ensure_exploitdb_mirror`` treats this as an existing clone (the
    ``git pull`` it then tries is mocked away by the caller). Writes
    ``files_exploits.csv`` from ``csv_rows`` and the per-row exploit
    files from ``files``.

    Args:
        mirror_path: Directory to populate.
        csv_rows: List of dicts to serialize into files_exploits.csv.
        files: ``{relative_path: content}`` mapping for the exploit files.
    """
    mirror_path.mkdir(parents=True, exist_ok=True)
    (mirror_path / ".git").mkdir(exist_ok=True)
    if csv_rows:
        import csv as _csv
        fieldnames = list(csv_rows[0].keys())
        with (mirror_path / "files_exploits.csv").open("w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)
    for rel, content in files.items():
        path = mirror_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_is_metasploit_module_detects_path_and_content():
    """Both path-based and header-based Metasploit detection signals fire."""
    # Path signal: file under metasploit/ is flagged regardless of content
    assert _is_metasploit_module("exploits/multiple/metasploit/foo.rb",
                                 "irrelevant body")
    # Content signal: Msf:: boilerplate near the top is flagged regardless of path
    msf_content = (
        "##\n"
        "# This module requires Metasploit\n"
        "##\n"
        "class MetasploitModule < Msf::Exploit::Remote\n"
        "  Rank = ExcellentRanking\n"
        "end\n"
    )
    assert _is_metasploit_module("exploits/linux/remote/whatever.rb", msf_content)
    # Negative case: an ordinary Python PoC is not flagged
    assert not _is_metasploit_module(
        "exploits/php/webapps/12345.py",
        "#!/usr/bin/env python3\n# CVE-2024-1234 PoC\nimport requests\n",
    )


def test_exploitdb_extracts_metadata_and_writes_record(tmp_path):
    """A single PoC under a fake mirror produces a record with structured metadata."""
    mirror = tmp_path / "mirror"
    out = tmp_path / "exploitdb.jsonl"

    csv_rows = [{
        "id": "50001",
        "file": "exploits/linux/local/50001.py",
        "description": "Linux Kernel 5.x privilege escalation via PoC",
        "date_published": "2024-03-12",
        "author": "research-team",
        "type": "local",
        "platform": "linux",
        "port": "",
        "screenshot_url": "",
        "application_url": "",
        "source_url": "",
        "codes": "CVE-2024-99999",
        "tags": "",
        "aliases": "",
        "verified": "1",
    }]
    poc = (
        "#!/usr/bin/env python3\n"
        "# CVE-2024-99999 — Linux kernel UAF privilege escalation PoC.\n"
        "# Tested on Ubuntu 22.04 with kernel 5.15.0-91-generic.\n"
        "import ctypes, os\n\n"
        + ("payload = b'A' * 4096\n" * 30)
    )
    files = {"exploits/linux/local/50001.py": poc}
    _stub_exploitdb_mirror(mirror, csv_rows, files)

    with patch("data.collect.subprocess.run",
               return_value=MagicMock(returncode=0, stdout="", stderr="")):
        collect_exploitdb(
            output_path=str(out),
            mirror_path=str(mirror),
            max_records=10,
        )

    records = load_jsonl(str(out))
    assert len(records) == 1
    r = records[0]
    assert r["id"] == "edb-50001"
    assert r["source"] == "exploitdb"
    assert r["platform"] == "linux"
    assert r["type"] == "local"
    assert r["codes"] == "CVE-2024-99999"
    assert r["language"] == "py"
    assert r["date"] == "2024-03-12"
    assert r["license"] == "GPL-2.0"
    # The header lines (Exploit-DB #, Platform, CVE, Date, Author) must
    # land at the start of the cleaned text so downstream readers can
    # see what the record is without reaching into structured fields.
    assert r["text"].startswith("Exploit-DB #50001:")
    assert "Platform: linux / local" in r["text"][:500]
    assert "CVE: CVE-2024-99999" in r["text"][:500]


def test_exploitdb_skips_metasploit_modules_by_default(tmp_path):
    """Metasploit framework modules are filtered unless ``skip_metasploit=False``."""
    mirror = tmp_path / "mirror"
    out = tmp_path / "exploitdb.jsonl"

    csv_rows = [
        {
            "id": "60001",
            "file": "exploits/linux/remote/msf_module.rb",
            "description": "Some Metasploit module",
            "date_published": "2023-01-01", "author": "msf",
            "type": "remote", "platform": "linux", "codes": "",
        },
        {
            "id": "60002",
            "file": "exploits/php/webapps/clean_poc.py",
            "description": "Clean python PoC",
            "date_published": "2023-02-01", "author": "researcher",
            "type": "webapps", "platform": "php", "codes": "CVE-2023-1111",
        },
    ]
    msf_body = (
        "##\n"
        "class MetasploitModule < Msf::Exploit::Remote\n"
        "  include Msf::Exploit::Remote::HttpClient\n"
        "  def exploit\n    print_status('go')\n  end\n"
        "end\n"
    ) + ("# filler\n" * 30)
    clean_body = ("# Clean PoC\n" + ("import requests\nrequests.get('/')\n" * 30))

    files = {
        "exploits/linux/remote/msf_module.rb": msf_body,
        "exploits/php/webapps/clean_poc.py": clean_body,
    }
    _stub_exploitdb_mirror(mirror, csv_rows, files)

    with patch("data.collect.subprocess.run",
               return_value=MagicMock(returncode=0, stdout="", stderr="")):
        collect_exploitdb(
            output_path=str(out), mirror_path=str(mirror), max_records=10,
        )

    ids = {r["id"] for r in load_jsonl(str(out))}
    assert ids == {"edb-60002"}, "Metasploit module should be filtered out"

    # With keep_metasploit, both records survive
    out2 = tmp_path / "exploitdb_keep.jsonl"
    with patch("data.collect.subprocess.run",
               return_value=MagicMock(returncode=0, stdout="", stderr="")):
        collect_exploitdb(
            output_path=str(out2), mirror_path=str(mirror), max_records=10,
            skip_metasploit=False,
        )
    ids2 = {r["id"] for r in load_jsonl(str(out2))}
    assert ids2 == {"edb-60001", "edb-60002"}


def test_exploitdb_resume_preserves_existing_and_appends(tmp_path):
    """A re-run keeps already-saved records and only appends new ones."""
    mirror = tmp_path / "mirror"
    out = tmp_path / "exploitdb.jsonl"

    csv_rows = [
        {"id": "70001", "file": "exploits/a.py", "description": "first",
         "type": "webapps", "platform": "linux", "codes": "",
         "date_published": "", "author": ""},
        {"id": "70002", "file": "exploits/b.py", "description": "second",
         "type": "webapps", "platform": "linux", "codes": "",
         "date_published": "", "author": ""},
    ]
    body = ("import requests\n" * 40)
    files = {"exploits/a.py": body, "exploits/b.py": body}
    _stub_exploitdb_mirror(mirror, csv_rows, files)

    # First run: only allow one record through
    with patch("data.collect.subprocess.run",
               return_value=MagicMock(returncode=0, stdout="", stderr="")):
        collect_exploitdb(
            output_path=str(out), mirror_path=str(mirror), max_records=1,
        )
    first_pass = load_jsonl(str(out))
    assert len(first_pass) == 1
    first_id = first_pass[0]["id"]

    # Second run: cap raised to 2, the prior record must persist and the
    # new one appended without duplicating the first.
    with patch("data.collect.subprocess.run",
               return_value=MagicMock(returncode=0, stdout="", stderr="")):
        collect_exploitdb(
            output_path=str(out), mirror_path=str(mirror), max_records=2,
        )
    second_pass = load_jsonl(str(out))
    ids = [r["id"] for r in second_pass]
    assert len(ids) == 2
    assert ids.count(first_id) == 1, "Prior record must not be duplicated on resume"
    assert {"edb-70001", "edb-70002"} == set(ids)


def test_exploitdb_truncates_long_records_preserves_header(tmp_path):
    """Records longer than max_chars are truncated; the metadata header survives."""
    mirror = tmp_path / "mirror"
    out = tmp_path / "exploitdb.jsonl"

    csv_rows = [{
        "id": "80001", "file": "exploits/big.py",
        "description": "Very long PoC", "type": "local", "platform": "linux",
        "codes": "CVE-2024-0001", "date_published": "", "author": "",
    }]
    files = {"exploits/big.py": ("A" * 50_000)}  # way past the 12K default
    _stub_exploitdb_mirror(mirror, csv_rows, files)

    with patch("data.collect.subprocess.run",
               return_value=MagicMock(returncode=0, stdout="", stderr="")):
        collect_exploitdb(
            output_path=str(out), mirror_path=str(mirror), max_records=10,
            max_chars=12000,
        )

    records = load_jsonl(str(out))
    assert len(records) == 1
    assert len(records[0]["text"]) == 12000
    # Header survives the truncation because it is at the start.
    assert records[0]["text"].startswith("Exploit-DB #80001:")
    assert "CVE: CVE-2024-0001" in records[0]["text"][:300]


def test_exploitdb_drops_short_records(tmp_path):
    """Records cleaned to fewer than min_chars are skipped."""
    mirror = tmp_path / "mirror"
    out = tmp_path / "exploitdb.jsonl"

    csv_rows = [
        {"id": "90001", "file": "exploits/tiny.py", "description": "x",
         "type": "webapps", "platform": "linux", "codes": "",
         "date_published": "", "author": ""},
        {"id": "90002", "file": "exploits/ok.py", "description": "ok",
         "type": "webapps", "platform": "linux", "codes": "",
         "date_published": "", "author": ""},
    ]
    files = {
        "exploits/tiny.py": "x",  # well below 100 char floor
        "exploits/ok.py": ("import requests\n" * 40),
    }
    _stub_exploitdb_mirror(mirror, csv_rows, files)

    with patch("data.collect.subprocess.run",
               return_value=MagicMock(returncode=0, stdout="", stderr="")):
        collect_exploitdb(
            output_path=str(out), mirror_path=str(mirror), max_records=10,
            min_chars=100,
        )

    ids = {r["id"] for r in load_jsonl(str(out))}
    assert ids == {"edb-90002"}


def test_exploitdb_missing_csv_is_a_no_op(tmp_path):
    """Mirror without a files_exploits.csv yields no output, no crash."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / ".git").mkdir()
    out = tmp_path / "exploitdb.jsonl"

    with patch("data.collect.subprocess.run",
               return_value=MagicMock(returncode=0, stdout="", stderr="")):
        collect_exploitdb(
            output_path=str(out), mirror_path=str(mirror), max_records=10,
        )

    assert not out.exists()
