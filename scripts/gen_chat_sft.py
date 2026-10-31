#!/usr/bin/env python3
"""Broad chat SFT conversations written by a teacher model through `opencode run` or `codex exec`.

    python scripts/gen_chat_sft.py run [--workers 3] [--rounds 4]   # resumable
    python scripts/gen_chat_sft.py status

Each job is one (area, topic, style, round) and asks the teacher for a handful of
conversations; validated ones land in data/chat_gen/<area>.jsonl as
``{"turns": [...], "source": "gen_<area>", "job": ...}``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/chat_gen"
MODEL = "opencode/deepseek-v4.1-flash-free"
BACKEND = "opencode"
TASK = "broad"
REVERSE = False
PER_JOB = 6
MAX_CHARS = 4000
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

AREAS = {
    "cybersec": [
        "SQL injection", "cross-site scripting", "CSRF", "SSRF", "XXE", "insecure deserialization",
        "authentication and session management", "OAuth 2.0 and OpenID Connect", "JWT pitfalls",
        "password hashing and storage", "TLS and certificates", "PKI and code signing",
        "network scanning with nmap", "firewall rules and segmentation", "VPNs and zero trust",
        "phishing and social engineering", "ransomware defence and recovery", "incident response process",
        "digital forensics on Windows", "Linux hardening", "Windows Active Directory attacks and defence",
        "Kerberos and NTLM", "privilege escalation on Linux", "privilege escalation on Windows",
        "malware analysis basics", "reverse engineering binaries", "buffer overflows and memory safety",
        "fuzzing", "secure code review", "threat modeling with STRIDE", "MITRE ATT&CK techniques",
        "detection engineering with Sigma rules", "YARA rules", "SIEM and log analysis",
        "cloud security on AWS", "Azure and Entra ID security", "Kubernetes and container security",
        "supply chain security and SBOMs", "secrets management", "API security", "mobile app security",
        "wireless and Wi-Fi security", "DNS security", "email security with SPF, DKIM and DMARC",
        "vulnerability management and CVSS", "CVE triage and patch prioritisation", "penetration testing methodology",
        "bug bounty hunting", "CTF challenges: web", "CTF challenges: crypto", "CTF challenges: pwn",
        "cryptography fundamentals", "OWASP Top 10", "security compliance (ISO 27001, SOC 2, PCI DSS)",
        "security awareness for non-technical staff", "OSINT and reconnaissance", "red team vs blue team",
        "EDR and endpoint detection", "web application firewalls", "browser security and same-origin policy",
        "IoT and embedded device security", "ICS and OT security", "LLM and prompt injection security",
    ],
    "code": [
        "Python basics", "Python data structures", "Python async and concurrency", "Python packaging and venvs",
        "JavaScript fundamentals", "TypeScript types", "React components and hooks", "Node.js and Express",
        "Go concurrency", "Rust ownership and borrowing", "C pointers and memory", "C++ classes and RAII",
        "Java and the JVM", "Bash scripting", "PowerShell scripting", "SQL queries and joins",
        "database indexing and performance", "REST API design", "Git workflows", "debugging techniques",
        "unit testing and mocking", "regular expressions", "algorithms: sorting and searching",
        "algorithms: graphs", "dynamic programming", "big-O complexity", "data structures: trees and heaps",
        "hash maps and sets", "recursion", "object-oriented design", "functional programming ideas",
        "error handling", "reading stack traces and error messages", "Docker and Dockerfiles", "CI/CD pipelines",
        "Linux command line", "HTTP and web fundamentals", "parsing JSON and CSV", "writing CLIs",
        "performance profiling", "refactoring legacy code", "code review feedback",
    ],
    "math": [
        "arithmetic word problems", "fractions and percentages", "algebra: linear equations",
        "algebra: quadratics", "systems of equations", "exponents and logarithms", "geometry: areas and volumes",
        "trigonometry", "probability", "statistics: mean, median, variance", "combinatorics",
        "number theory and primes", "modular arithmetic", "calculus: derivatives", "calculus: integrals",
        "limits and series", "linear algebra: vectors and matrices", "discrete math and logic",
        "proof techniques", "unit conversions and estimation", "binary and hexadecimal", "financial math and interest",
    ],
    "science": [
        "physics: motion and forces", "physics: energy and electricity", "chemistry basics", "biology: cells and genetics",
        "human body and health basics", "astronomy and space", "earth science and climate", "computer hardware",
        "how the internet works", "machine learning basics", "neural networks and transformers",
        "history of computing", "scientific method",
    ],
    "general": [
        "world history", "geography", "economics basics", "personal finance", "writing a professional email",
        "summarising a paragraph the user provides", "rewriting text to be clearer", "grammar and word usage",
        "brainstorming ideas", "study tips and learning", "career advice for developers", "productivity and planning",
        "cooking and recipes", "travel planning", "explaining a concept to a child", "everyday reasoning puzzles",
        "comparing two options and recommending one", "translation of short phrases", "fun facts and trivia",
        "health and fitness basics (general, non-medical advice)", "music, books and film",
    ],
    "assistant": [
        "greetings and small talk", "questions about who the assistant is and what it can do",
        "the user is vague and the assistant asks one clarifying question", "the user corrects the assistant",
        "the user asks for a shorter or longer answer", "the user asks something the assistant cannot know (live data, personal info)",
        "harmful requests the assistant should decline briefly and offer a safe alternative",
        "dual-use security questions answered in a defensive, educational way", "following formatting instructions exactly",
        "multi-step instructions", "admitting uncertainty instead of guessing",
    ],
}

STYLES = [
    "a beginner asking a simple question",
    "an experienced professional asking a precise, technical question",
    "a request for step-by-step instructions",
    "a request to write or fix code, a config, or a command (include it in the answer)",
    "a comparison between two things",
    "a two- or three-exchange conversation with natural follow-up questions",
    "a terse question with typos, as people really type",
    "a 'why' question that needs a short explanation of the underlying reason",
]

MULTITURN_KINDS = [
    "follow-ups that only make sense with the earlier turns (\"what about on Windows?\", \"why?\", \"show it in Python\"); GhostLM resolves it/that/this correctly",
    "a correction: in some conversations GhostLM really was wrong earlier (a realistic slip like a wrong port, flag or event ID) and admits it plainly and fixes it; in others the user is wrong and GhostLM politely holds its ground with the reason",
    "a debugging or investigation session: the user pastes a log, alert, config or error, GhostLM points at the key detail or asks for it, the user reports back, GhostLM narrows it down",
    "a genuinely ambiguous first request where GhostLM asks one short clarifying question, the user answers, and GhostLM then answers well",
    "a student working through the topic step by step, each turn building on the last",
]

PERSONAS = ["a student", "a SOC analyst", "a software developer", "a sysadmin", "a curious non-technical person",
            "a pentester", "a teacher", "a startup founder", "a data scientist", "a hobbyist"]

RULES = """You are writing supervised fine-tuning data for GhostLM, a small open-source assistant that is general-purpose with particular strength in cybersecurity. Write {n} different conversations between a user and GhostLM.

Area: {area}. Topic: {topic}. Style of the user's first message: {style}. The user can be {persona} when that fits the topic naturally; otherwise just someone interested in it. Never force the user's job into a question where it doesn't belong.

Rules for GhostLM's replies:
- Correct and specific. Only state facts you are confident are true. Answer the actual question in the first sentence.
- Concise: usually 2-8 sentences, longer only when code or steps are needed. Each whole conversation under 2500 characters.
- Code, commands and configs go in fenced code blocks with the language named, and must be correct.
- Math shows the key steps and ends with the result stated plainly. Work the answer out before writing it: never correct yourself mid-reply. If the numbers in a question don't work out, say so plainly.
- Plain ASCII punctuation: straight quotes and apostrophes, a plain hyphen, never an em or en dash.
- Plain, friendly, direct tone. No emoji. No "As an AI", no "Great question", no filler closing lines.
- GhostLM calls itself GhostLM if asked. Never mention any other assistant, model or company as its identity.
- For clearly harmful requests (malware for real targets, attacking systems without permission, weapons, self-harm), decline in one or two sentences and offer a safe alternative. Ordinary security education, defence, and authorised testing questions are answered normally. Only mention authorisation when the user is about to actively test a system; looking up public information needs no caveat.
- Vary the questions: different subtopics, specifics, names, numbers and lengths in every conversation. Never repeat a question.

Output exactly {n} lines and nothing else, each a JSON object:
{{"turns": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
Turns alternate user/assistant, start with user and end with assistant. Use \\n inside strings for newlines."""

LEAK = re.compile(r"\b(deepseek|openai|chatgpt|gpt-4|claude|anthropic|gemini|qwen|llama|mistral|"
                  r"as an ai|language model developed by)\b", re.I)
FILLER = re.compile(r"^(great question|certainly!|sure!|absolutely!)", re.I)
SELF_CORRECTION = re.compile(r"\bwait,|\bcorrection:|\u2014", re.I)
ASCII_PUNCT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2212": "-", "\u2013": "-"})
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def jobs(rounds: int) -> list:
    out = []
    if TASK == "multiturn":
        for topic in AREAS["cybersec"]:
            for ki, kind in enumerate(MULTITURN_KINDS):
                for r in range(rounds):
                    out.append({"area": "multiturn_cybersec", "topic": topic, "style": kind, "round": r,
                                "key": f"multiturn_cybersec|{topic}|{ki}|{r}"})
        random.Random(0).shuffle(out)
        return out
    for area, topics in AREAS.items():
        for topic in topics:
            for si, style in enumerate(STYLES):
                for r in range(rounds):
                    out.append({"area": area, "topic": topic, "style": style, "round": r,
                                "key": f"{area}|{topic}|{si}|{r}"})
    random.Random(0).shuffle(out)
    return out


def done_keys() -> set:
    keys = set()
    for f in OUT.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                keys.add(json.loads(line).get("job"))
    return keys


def valid(conv) -> bool:
    turns = conv.get("turns") if isinstance(conv, dict) else None
    if not isinstance(turns, list) or len(turns) < 2 or len(turns) % 2:
        return False
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or t.get("role") != ("user" if i % 2 == 0 else "assistant"):
            return False
        c = t.get("content")
        if not isinstance(c, str) or not c.strip():
            return False
        t["content"] = c = c.translate(ASCII_PUNCT)
        if t["role"] == "assistant" and (LEAK.search(c) or FILLER.match(c.strip()) or EMOJI.search(c)
                                         or SELF_CORRECTION.search(c)):
            return False
    return sum(len(t["content"]) for t in turns) <= MAX_CHARS


def parse(text: str) -> list:
    convs = []
    for line in ANSI.sub("", text).splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                convs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return convs


def ask(prompt: str, timeout: int = 300) -> str:
    try:
        if BACKEND == "opencode":
            return subprocess.run(["opencode", "run", "-m", MODEL, prompt], capture_output=True, text=True,
                                  timeout=timeout, cwd=ROOT).stdout
        with tempfile.NamedTemporaryFile("r", suffix=".txt") as out:
            subprocess.run(["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
                            "-o", out.name, prompt], capture_output=True, text=True, timeout=timeout, cwd=ROOT)
            return Path(out.name).read_text()
    except subprocess.TimeoutExpired:
        return ""


lock = threading.Lock()
seen: set = set()


def work(job: dict) -> int:
    rng = random.Random(job["key"])
    prompt = RULES.format(n=PER_JOB, area=job["area"], topic=job["topic"], style=job["style"],
                          persona=rng.choice(PERSONAS))
    if TASK == "multiturn":
        prompt = prompt.replace(f"Write {PER_JOB} different", "Write 4 different").replace(
            f"Output exactly {PER_JOB} lines", "Output exactly 4 lines").replace(
            "Each whole conversation under 2500 characters.", "Each whole conversation under 3800 characters.")
        prompt += ("\n\nEvery conversation has 3 to 5 exchanges (6 to 10 turns), so GhostLM has to use what was "
                   "said earlier. Keep each reply tight. User turns sound like real people: short, sometimes terse.")
    if job["round"]:
        prompt += f"\n\nThis is variation set {job['round'] + 1}: pick less obvious subtopics than a first attempt would."
    kept = []
    for attempt in range(3):
        convs = [c for c in parse(ask(prompt)) if valid(c)]
        if convs:
            break
        time.sleep(20 * (attempt + 1))
    if not convs:
        # Usually the free tier's usage limit; wait it out rather than burn through the job list.
        time.sleep(1800)
    with lock:
        for c in convs:
            first = " ".join(c["turns"][0]["content"].lower().split())
            if first in seen:
                continue
            seen.add(first)
            kept.append(json.dumps({"turns": [{"role": t["role"], "content": t["content"].strip()} for t in c["turns"]],
                                    "source": f"gen_{job['area']}", "job": job["key"], "teacher": BACKEND},
                                   ensure_ascii=False))
        if kept:
            with (OUT / f"{job['area']}.jsonl").open("a") as f:
                f.write("\n".join(kept) + "\n")
    return len(kept)


def run(workers: int, rounds: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                seen.add(" ".join(json.loads(line)["turns"][0]["content"].lower().split()))
    done = done_keys()
    todo = [j for j in jobs(rounds) if j["key"] not in done]
    if REVERSE:
        todo.reverse()
    print(f"{len(todo)} jobs to do", flush=True)
    total = 0
    with ThreadPoolExecutor(workers) as pool:
        for i, n in enumerate(pool.map(work, todo), 1):
            total += n
            if i % 20 == 0:
                print(f"{time.strftime('%H:%M')} {i}/{len(todo)} jobs, +{total} conversations", flush=True)
    print(f"pass finished: +{total} conversations", flush=True)


def status(rounds: int) -> None:
    counts = {f.stem: sum(1 for line in f.read_text().splitlines() if line.strip()) for f in sorted(OUT.glob("*.jsonl"))}
    print(f"{len(done_keys() & {j['key'] for j in jobs(rounds)})}/{len(jobs(rounds))} jobs done; "
          f"{sum(counts.values())} conversations {counts}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=["run", "status"])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--backend", choices=["opencode", "codex"], default="opencode")
    p.add_argument("--task", choices=["broad", "multiturn"], default="broad")
    p.add_argument("--reverse", action="store_true", help="work from the end of the job list, so two backends don't overlap")
    a = p.parse_args()
    global BACKEND, REVERSE, TASK
    BACKEND, REVERSE, TASK = a.backend, a.reverse, a.task
    if a.cmd == "status":
        return status(a.rounds)
    # Jobs that failed are not marked done, so later passes pick them up again.
    while True:
        run(a.workers, a.rounds)
        if all(j["key"] in done_keys() for j in jobs(a.rounds)):
            break


if __name__ == "__main__":
    main()
