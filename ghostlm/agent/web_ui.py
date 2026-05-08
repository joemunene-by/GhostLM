"""Static HTML demo UI served at GET / by the HTTP server.

A single-page chat UI with no build step, no JS framework, no
external dependencies. POSTs to /v1/agent/run, renders the trace
inline (assistant text, tool calls, tool responses, cite tags),
shows iteration count and latency. The point is "clone the repo,
start the server, open localhost in a browser, see GhostAgent work"
in three commands and zero configuration.

The HTML is exported as a single string so the server can return it
verbatim from a route handler. CSS + JS are embedded; the UI is
otherwise self-contained.
"""

from __future__ import annotations


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GhostAgent</title>
<style>
:root {
  --bg: #0b0d10;
  --panel: #14171c;
  --border: #232830;
  --text: #e6e9ee;
  --muted: #8d96a3;
  --accent: #6ad7ff;
  --tool: #ffb86c;
  --cite: #c490ff;
  --error: #ff7676;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  display: flex;
  flex-direction: column;
  height: 100vh;
  font-size: 14px;
}
header {
  padding: 14px 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 14px;
  background: var(--panel);
}
header h1 {
  font-size: 16px; font-weight: 600; margin: 0;
  letter-spacing: 0.3px;
}
header .meta {
  color: var(--muted); font-size: 12px;
}
header .badge {
  background: #1f2530; padding: 3px 8px; border-radius: 4px;
  font-size: 11px; color: var(--muted);
  font-family: ui-monospace, "SF Mono", monospace;
}
#chat {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.msg {
  max-width: 800px;
  width: 100%;
  align-self: center;
}
.msg .role {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: var(--muted);
  margin-bottom: 4px;
  font-family: ui-monospace, "SF Mono", monospace;
}
.msg .body {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  white-space: pre-wrap;
  word-wrap: break-word;
  line-height: 1.5;
}
.msg.user .body { background: #1a2230; border-color: #2a3849; }
.msg.tool .body { border-left: 3px solid var(--tool); }
.msg.tool .role { color: var(--tool); }
.msg.assistant .body { border-left: 3px solid var(--accent); }
.msg.assistant .role { color: var(--accent); }
.msg.error .body { color: var(--error); border-color: var(--error); }
.cite {
  background: rgba(196, 144, 255, 0.13);
  color: var(--cite);
  padding: 1px 5px;
  border-radius: 3px;
  font-family: ui-monospace, "SF Mono", monospace;
  font-size: 12px;
}
.tool-block {
  background: #0e1318;
  border-radius: 5px;
  padding: 8px 10px;
  margin-top: 6px;
  font-family: ui-monospace, "SF Mono", monospace;
  font-size: 12px;
  color: var(--tool);
  white-space: pre-wrap;
}
.trace-meta {
  color: var(--muted);
  font-size: 12px;
  margin-top: 6px;
  font-family: ui-monospace, "SF Mono", monospace;
}
form {
  display: flex;
  gap: 8px;
  padding: 14px 20px;
  border-top: 1px solid var(--border);
  background: var(--panel);
}
input[type=text] {
  flex: 1;
  background: #0e1318;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
  color: var(--text);
  font-family: inherit;
  font-size: 14px;
}
input[type=text]:focus { outline: none; border-color: var(--accent); }
button {
  background: var(--accent);
  color: #0b0d10;
  border: none;
  border-radius: 6px;
  padding: 10px 18px;
  font-weight: 600;
  cursor: pointer;
  font-family: inherit;
}
button:disabled { opacity: 0.5; cursor: not-allowed; }
.examples {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  padding: 8px 20px 0;
}
.examples button {
  background: transparent;
  color: var(--muted);
  border: 1px solid var(--border);
  font-size: 11px;
  padding: 4px 8px;
  font-weight: 400;
}
.examples button:hover { color: var(--text); border-color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>GhostAgent</h1>
  <span class="meta">cybersec tool-using agent</span>
  <span class="meta" id="model-badge">model: <span class="badge" id="model-name">loading...</span></span>
  <span class="meta" id="tools-badge">tools: <span class="badge" id="tools-list">…</span></span>
</header>

<div class="examples">
  <button data-q="What is CVE-2017-0144?">What is CVE-2017-0144?</button>
  <button data-q="Is CVE-2021-44228 actively exploited?">Is CVE-2021-44228 exploited?</button>
  <button data-q="Explain MITRE T1003.001">Explain T1003.001</button>
  <button data-q="What is CWE-89?">What is CWE-89?</button>
  <button data-q="Look up GreyNoise for 8.8.8.8">GreyNoise 8.8.8.8</button>
  <button data-q="What does Shodan say about 1.1.1.1?">Shodan 1.1.1.1</button>
</div>

<div id="chat"></div>

<form id="form">
  <input type="text" id="input" placeholder="Ask GhostAgent..." autocomplete="off">
  <button type="submit" id="submit">Send</button>
</form>

<script>
const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const submit = document.getElementById("submit");

async function loadHealth() {
  try {
    const r = await fetch("/healthz");
    const d = await r.json();
    document.getElementById("model-name").textContent = d.model || "ghostlm";
    document.getElementById("tools-list").textContent =
      (d.tools || []).join(", ") || "none";
  } catch (e) {
    document.getElementById("model-name").textContent = "(server)";
    document.getElementById("tools-list").textContent = "(unavailable)";
  }
}
loadHealth();

function escapeHtml(s) {
  return (s || "").replace(/[&<>]/g, c =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
}

function renderCites(text) {
  // Convert <|cite|>type:id<|/cite|> into spans.
  return escapeHtml(text).replace(
    /&lt;\\|cite\\|&gt;([^&]+?)&lt;\\|\\/cite\\|&gt;/g,
    (_, body) => `<span class="cite">[${body}]</span>`
  );
}

function appendMsg(role, body, extra) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  const r = document.createElement("div");
  r.className = "role"; r.textContent = role;
  const b = document.createElement("div");
  b.className = "body";
  b.innerHTML = body;
  el.appendChild(r); el.appendChild(b);
  if (extra) {
    const m = document.createElement("div");
    m.className = "trace-meta"; m.textContent = extra;
    el.appendChild(m);
  }
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function renderTrace(trace) {
  for (const m of trace.history || []) {
    if (m.role === "system") continue;
    if (m.role === "user") continue;  // user already shown
    if (m.role === "assistant") {
      // Pull tool_calls from metadata.
      const tcs = (m.metadata && m.metadata.tool_calls) || [];
      let body = renderCites(m.content);
      // Show the prose without the literal <|tool_call|> blocks.
      body = body.replace(
        /&lt;\\|tool_call\\|&gt;[^]*?&lt;\\|\\/tool_call\\|&gt;/g, "");
      if (tcs.length) {
        body += tcs.map(tc =>
          `<div class="tool-block">→ ${escapeHtml(tc.name)}(${
            escapeHtml(JSON.stringify(tc.args))})</div>`
        ).join("");
      }
      appendMsg("assistant", body);
    } else if (m.role === "tool") {
      // m.content is "<|tool_response|>{...}<|/tool_response|>".
      const match = (m.content || "").match(
        /<\\|tool_response\\|>([\\s\\S]*?)<\\|\\/tool_response\\|>/);
      const body = match ? match[1] : m.content;
      const name = (m.metadata && m.metadata.tool_name) || "tool";
      appendMsg("tool",
        `<div class="tool-block">${escapeHtml(name)} →
${escapeHtml(body)}</div>`);
    }
  }
}

async function runQuery(q) {
  appendMsg("user", escapeHtml(q));
  submit.disabled = true;
  input.value = "";
  try {
    const r = await fetch("/v1/agent/run", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({query: q, include_trace: true}),
    });
    if (!r.ok) {
      appendMsg("error", `HTTP ${r.status}: ${escapeHtml(await r.text())}`);
      return;
    }
    const d = await r.json();
    if (d.trace) {
      renderTrace(d.trace);
    } else {
      appendMsg("assistant", renderCites(d.final_answer || ""));
    }
    appendMsg("assistant",
      `<em style="opacity:0.6">final: ${renderCites(d.final_answer || "")}</em>`,
      `${d.terminated_reason} · ${d.iterations} iter · ${d.total_latency_ms}ms`);
  } catch (e) {
    appendMsg("error", escapeHtml(e.message || String(e)));
  } finally {
    submit.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", e => {
  e.preventDefault();
  const q = input.value.trim();
  if (q) runQuery(q);
});

document.querySelectorAll(".examples button").forEach(btn => {
  btn.addEventListener("click", () => {
    const q = btn.getAttribute("data-q");
    if (q) runQuery(q);
  });
});

input.focus();
</script>
</body>
</html>
"""
