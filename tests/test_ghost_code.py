"""Tests for Ghost Code: tools, permission modes, the agent loop and the mascot."""

import json
import re

import pytest

pytest.importorskip("httpx")
pytest.importorskip("rich")

from ghostlm.code import mascot, tools  # noqa: E402
from ghostlm.code.agent import Agent, bash_rule  # noqa: E402
from ghostlm.code.backend import Cancelled, Reply, ToolCall  # noqa: E402


class FakeBackend:
    """Plays back scripted replies and records what it was sent."""

    base_url, model = "http://test/v1", "fake"

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None, on_text=lambda s: None, on_thinking=lambda s: None, cancel=None):
        self.seen.append(json.loads(json.dumps(messages)))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        on_text(reply.text)
        return reply


class FakeUI:
    def __init__(self, answer=0):
        self.answer, self.asked, self.calls, self.results, self.notices = answer, [], [], [], []

    def on_text(self, chunk): ...
    def on_thinking(self, chunk): ...
    def start_reply(self): ...
    def end_reply(self, reply): ...

    def tool_call(self, label):
        self.calls.append(label)

    def tool_result(self, result):
        self.results.append(result)

    def ask(self, title, detail, options):
        self.asked.append((title, detail))
        return self.answer if self.answer >= 0 else len(options) - 1

    def notice(self, text):
        self.notices.append(text)


def call(name, **args):
    return Reply(tool_calls=[ToolCall(f"c{name}", name, json.dumps(args))])


def agent_with(tmp_path, *replies, answer=0, mode="default"):
    ui = FakeUI(answer)
    return Agent(FakeBackend(*replies), ui, tmp_path, mode), ui


def test_read_file_numbers_lines_and_honours_offset(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    r = tools.run("read_file", {"path": "a.txt", "offset": 2, "limit": 1}, tmp_path)
    assert r.output.strip() == "2\ttwo" and r.summary == "Read 1 line"


def test_edit_file_requires_a_unique_match(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\nx = 1\n")
    r = tools.run("edit_file", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}, tmp_path)
    assert not r.ok and "2 places" in r.output and f.read_text() == "x = 1\nx = 1\n"
    r = tools.run("edit_file", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2",
                                "replace_all": True}, tmp_path)
    assert r.ok and f.read_text() == "x = 2\nx = 2\n" and r.summary.endswith("2 additions and 2 removals")


def test_write_file_reports_a_diff(tmp_path):
    r = tools.run("write_file", {"path": "new/b.txt", "content": "hi\n"}, tmp_path)
    assert (tmp_path / "new/b.txt").read_text() == "hi\n" and r.summary.startswith("Created b.txt")
    assert "+hi" in r.diff


def test_bash_reports_failure_and_exit_code(tmp_path):
    r = tools.run("bash", {"command": "echo out; exit 3"}, tmp_path)
    assert not r.ok and "out" in r.output and "[exit code 3]" in r.output


def test_grep_and_glob_skip_vendored_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("needle = 1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules/b.py").write_text("needle = 2\n")
    assert tools.run("grep", {"pattern": "needle"}, tmp_path).output == "src/a.py:1: needle = 1"
    assert tools.run("glob", {"pattern": "**/*.py"}, tmp_path).output == "src/a.py"


def test_unknown_tool_and_missing_argument_are_reported_not_raised(tmp_path):
    assert not tools.run("teleport", {}, tmp_path).ok
    assert "Missing required argument: path" in tools.run("read_file", {}, tmp_path).output


def test_turn_runs_tools_then_returns_the_final_answer(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    agent, ui = agent_with(tmp_path, call("read_file", path="a.txt"), Reply(text="It says hello."))
    assert agent.run_turn("what is in a.txt?") == "It says hello."
    assert ui.calls == ["Read(a.txt)"] and not ui.asked
    tool_msg = agent.backend.seen[1][-1]
    assert tool_msg["role"] == "tool" and "hello" in tool_msg["content"]


def test_default_mode_asks_before_edits_and_respects_refusal(tmp_path):
    agent, ui = agent_with(tmp_path, call("write_file", path="a.txt", content="x"), Reply(text="ok"), answer=2)
    agent.run_turn("write it")
    assert ui.asked and not (tmp_path / "a.txt").exists()
    assert "declined" in agent.messages[-2]["content"]


def test_allowing_edits_for_the_session_stops_the_prompts(tmp_path):
    agent, ui = agent_with(tmp_path, call("write_file", path="a.txt", content="x"),
                           call("write_file", path="b.txt", content="y"), Reply(text="ok"), answer=1)
    agent.run_turn("write both")
    assert len(ui.asked) == 1 and (tmp_path / "b.txt").read_text() == "y"


def test_accept_edits_mode_still_asks_outside_the_working_directory(tmp_path):
    inside, outside = tmp_path / "proj", tmp_path / "elsewhere.txt"
    inside.mkdir()
    agent, ui = agent_with(inside, call("write_file", path="a.txt", content="x"),
                           call("write_file", path=str(outside), content="y"), Reply(text="ok"),
                           answer=2, mode="accept edits")
    agent.run_turn("write both")
    assert (inside / "a.txt").exists() and not outside.exists() and len(ui.asked) == 1


def test_plan_mode_blocks_changes_but_allows_reading(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    agent, ui = agent_with(tmp_path, call("read_file", path="a.txt"), call("bash", command="rm a.txt"),
                           Reply(text="plan"), mode="plan")
    agent.run_turn("clean up")
    assert (tmp_path / "a.txt").exists() and not ui.asked
    assert "Plan mode is on" in agent.messages[-2]["content"]


def test_bash_allow_rule_matches_the_first_two_words(tmp_path):
    assert bash_rule("git status --short") == "git status"
    agent, ui = agent_with(tmp_path, call("bash", command="git status"),
                           call("bash", command="git status --short"), call("bash", command="git log -1"),
                           Reply(text="ok"), answer=1)
    agent.run_turn("go")
    assert [d for _, d in ui.asked] == ["git status", "git log -1"]


def test_bad_tool_arguments_are_fed_back_to_the_model(tmp_path):
    bad = Reply(tool_calls=[ToolCall("c1", "read_file", "{not json")])
    agent, ui = agent_with(tmp_path, bad, Reply(text="sorry"))
    agent.run_turn("go")
    assert "Could not parse" in agent.messages[-2]["content"]


def test_cancel_leaves_a_valid_history(tmp_path):
    agent, ui = agent_with(tmp_path, Cancelled())
    agent.run_turn("go")
    assert agent.messages[-1]["role"] == "assistant" and "Interrupted" in ui.notices


def test_repair_answers_orphaned_tool_calls(tmp_path):
    agent, _ = agent_with(tmp_path)
    agent.messages.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]})
    agent.repair()
    assert agent.messages[-1] == {"role": "tool", "tool_call_id": "c1", "content": "[interrupted before this ran]"}


def test_project_instructions_reach_the_system_prompt(tmp_path):
    (tmp_path / "GHOST.md").write_text("Always use tabs.")
    agent, _ = agent_with(tmp_path)
    assert "Always use tabs." in agent.messages[0]["content"]


def test_mascot_frames_keep_their_shape_and_animate():
    frames = [mascot.frame(t) for t in range(60)]
    assert all(len(f) == 6 and all(len(line) == 11 for line in f) for f in frames)
    assert mascot.frame(0) != mascot.frame(3), "the hem drifts"
    assert mascot.frame(14) != mascot.frame(13), "the eyes blink"
    assert mascot.grid(glance=1)[4] != mascot.grid()[4], "the eyes glance sideways"
    everything = {(c, r) for c in range(mascot.COLS) for r in range(mascot.ROWS)}
    assert mascot.intro_frames()[-1] == mascot.render(mascot.grid(reveal=everything))


def test_cli_picks_a_coder_model_first():
    from ghostlm.code.cli import pick_model
    assert pick_model(["llama3", "qwen2.5-coder:14b", "qwen3:8b"]) == "qwen2.5-coder:14b"
    assert pick_model(["mistral"]) == "mistral" and pick_model([]) is None


def _sse_server(chunks, delay=0.0):
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): ...

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(delay)
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_backend_assembles_streamed_text_and_split_tool_calls():
    from ghostlm.code.backend import Backend
    server = _sse_server([
        {"choices": [{"delta": {"reasoning": "hmm"}}]},
        {"choices": [{"delta": {"content": "Look"}}]},
        {"choices": [{"delta": {"content": "ing."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": '{"pa'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]}}]},
        {"choices": [], "usage": {"total_tokens": 42}},
    ])
    text, thoughts = [], []
    reply = Backend(f"http://127.0.0.1:{server.server_port}/v1", "m").chat(
        [{"role": "user", "content": "hi"}], tools.SPECS, text.append, thoughts.append)
    server.shutdown()
    assert reply.text == "Looking." and text == ["Look", "ing."] and thoughts == ["hmm"]
    assert [(c.id, c.name, json.loads(c.arguments)) for c in reply.tool_calls] == [("c1", "read_file", {"path": "a.py"})]
    assert reply.usage == {"total_tokens": 42}


def test_backend_cancel_returns_promptly_mid_stream():
    import threading
    import time

    from ghostlm.code.backend import Backend
    server = _sse_server([{"choices": [{"delta": {"content": "x"}}]}] * 50, delay=0.2)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.time()
    with pytest.raises(Cancelled):
        Backend(f"http://127.0.0.1:{server.server_port}/v1", "m").chat([], None, cancel=cancel)
    server.shutdown()
    assert time.time() - started < 2


def test_backend_reports_an_unreachable_server_clearly():
    from ghostlm.code.backend import Backend, BackendError
    with pytest.raises(BackendError, match="Is the model server running"):
        Backend("http://127.0.0.1:9/v1", "m").chat([], None)


def test_security_checks_cover_macos_and_linux_and_only_read():
    writers = {"rm", "sudo", "kill", "chmod", "chown", "mv", "cp", "tee", "launchctl load", "launchctl unload",
               "defaults write", "ufw enable", "ufw disable", "systemctl start", "systemctl stop"}
    for area, per_os in tools.CHECKS.items():
        assert {"Darwin", "Linux"} <= set(per_os), area
        for command in (c for commands in per_os.values() for c in commands):
            assert not re.search(r"[^2]>(?!=)|>>", command.replace("2>/dev/null", "")), f"{area} redirects output"
            for segment in re.split(r"\|\||&&|[|;(]", command):
                words = segment.split()
                assert not {" ".join(words[:1]), " ".join(words[:2])} & writers, f"{area}: {segment.strip()}"


def test_security_check_returns_evidence_per_area_without_asking(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "CHECKS", {"demo": {"Darwin": ["echo fine"], "Linux": ["echo fine"]}})
    agent, ui = agent_with(tmp_path, call("security_check", area="all"), Reply(text="done"))
    agent.run_turn("audit")
    assert ui.calls == ["Check(all)"] and not ui.asked
    assert agent.messages[-2]["content"] == "## demo\nfine"
    assert not tools.run("security_check", {"area": "nope"}, tmp_path).ok


def test_secret_scan_finds_credentials_and_masks_their_values(tmp_path):
    (tmp_path / "settings.py").write_text('DEBUG = True\nAPI_KEY = "sk-abcdefghijklmnopqrstuvwx"\npassword = "hunter2hunter2"\n')
    (tmp_path / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n")
    (tmp_path / "clean.py").write_text("password = os.environ['PASSWORD']\n")
    r = tools.run("secret_scan", {}, tmp_path)
    (tmp_path / "db.py").write_text('URL = "postgres://shop:hunter2@db:5432/shop"\nADMIN_PASSWORD = "letmein-2024"\n')
    r = tools.run("secret_scan", {}, tmp_path)
    assert r.summary == "Found 5 possible secrets"
    assert "db.py:1: credentials in URL" in r.output and "db.py:2: hardcoded credential" in r.output
    assert "settings.py:2: API key" in r.output and "settings.py:3: hardcoded credential" in r.output
    assert "id_rsa:1: private key" in r.output and "clean.py" not in r.output
    assert "abcdefghijklmnop" not in r.output and "hunter2hunter2" not in r.output


def test_lookup_cve_uses_the_offline_cache_when_told_to(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_AGENT_OFFLINE", "1")
    r = tools.run("lookup_cve", {"cve_id": "cve-2017-0144"}, tmp_path)
    assert r.ok and "CVE-2017-0144" in r.output and r.summary == "CVE-2017-0144: found"
    assert not tools.run("lookup_cve", {"cve_id": "CVE-1999-99999"}, tmp_path).ok


def test_system_prompt_sets_the_defensive_security_role(tmp_path):
    agent, _ = agent_with(tmp_path)
    prompt = agent.messages[0]["content"]
    assert "security_check" in prompt and "only on systems the user owns" in prompt


def test_an_edit_that_cannot_apply_is_rejected_before_asking(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, ui = agent_with(tmp_path, call("edit_file", path="a.py", old_string="y = 2", new_string="y = 3"),
                           Reply(text="ok"))
    agent.run_turn("edit")
    assert not ui.asked and ui.results[0].summary == "String not found"


def test_glob_accepts_several_comma_separated_patterns(tmp_path):
    for name in ("a.py", "b.js", "c.txt"):
        (tmp_path / name).write_text("")
    assert tools.run("glob", {"pattern": "*.py, *.js"}, tmp_path).output == "a.py\nb.js"


def test_ollama_messages_use_object_arguments_and_tool_names():
    from ghostlm.code.backend import _ollama_messages
    out = _ollama_messages([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x = 1"}])
    assert out[0]["tool_calls"] == [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]
    assert out[1] == {"role": "tool", "content": "x = 1", "tool_name": "read_file"}
