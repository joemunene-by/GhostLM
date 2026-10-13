"""Explains pasted code by reading it, not by guessing: structure, flow and a security check.

The small models cannot explain code reliably and the retrieval index holds security references,
so for a code question the app describes what is verifiably in the code instead.
"""

from __future__ import annotations

import re
from typing import List

LANGUAGES = [
    ("Java", r"\bpublic\s+(?:static\s+)?(?:final\s+)?class\b|\bSystem\.out\.|\bimport\s+java\."),
    ("C#", r"\busing\s+System\b|\bConsole\.Write|\bnamespace\s+\w+"),
    ("Python", r"^\s*def\s+\w+\s*\(.*\)\s*(?:->[^:]+)?:|^\s*import\s+\w+\s*$|^\s*from\s+[\w.]+\s+import\b|\bprint\("),
    ("TypeScript", r":\s*(?:string|number|boolean)\b|\binterface\s+\w+\s*\{"),
    ("JavaScript", r"\bconsole\.log\(|\b(?:const|let)\s+\w+\s*=|\bfunction\s+\w+\s*\(|=>\s*\{"),
    ("Go", r"^\s*package\s+\w+|\bfunc\s+\w+\(|\bfmt\.Print"),
    ("Rust", r"\bfn\s+\w+\s*\(|\blet\s+mut\b|\bprintln!\("),
    ("C++", r"#include\s*<iostream>|\bstd::|\bcout\s*<<"),
    ("C", r"#include\s*<\w+\.h>|\bprintf\s*\(|\bint\s+main\s*\("),
    ("PHP", r"<\?php|\$\w+\s*=|\becho\s"),
    ("Ruby", r"^\s*def\s+\w+\s*$|\bputs\s|\bend\s*$"),
    ("SQL", r"(?i)^\s*(?:select|insert\s+into|update|delete\s+from|create\s+table)\b"),
    ("Shell", r"^#!/bin/(?:ba)?sh|\becho\s+\$|\bfi\s*$"),
]
NOT_NAMES = {"if", "for", "while", "switch", "catch", "return", "new", "else", "do", "try", "sizeof", "print",
             "function", "def", "class", "super", "this", "elif", "with", "assert", "lambda", "match", "case"}
READS = re.compile(r"Scanner\s*\(\s*System\.in|\binput\s*\(|readline|\bscanf\s*\(|\bcin\s*>>|process\.stdin|"
                   r"Console\.Read|\.next(?:Int|Line)?\s*\(")
PRINTS = re.compile(r"(?:println|print|printf|console\.log|puts|echo|Console\.WriteLine|cout\s*<<)\s*\(?\s*"
                    r"f?[\"']([^\"'\n]{4,60})[\"']")
RISKS = [
    (r"\beval\s*\(|\bexec\s*\(", "runs code built at run time (eval or exec)"),
    (r"shell\s*=\s*True|\bos\.system\s*\(|Runtime\.getRuntime\(\)\.exec|\bsystem\s*\(|child_process", "runs shell commands"),
    (r"(?i)(?:select|insert|update|delete)\b[^;\n]*[\"']\s*\+|\+\s*[\"'][^\"'\n]*\b(?:where|values)\b|f[\"'](?:select|insert|update|delete)\b",
     "builds SQL by joining strings, which allows SQL injection"),
    (r"verify\s*=\s*False|rejectUnauthorized\s*:\s*false|InsecureSkipVerify\s*:\s*true", "turns off TLS certificate checks"),
    (r"\b(?:strcpy|strcat|sprintf|gets)\s*\(", "uses C string functions with no bounds check"),
    (r"\.innerHTML\s*=|document\.write\s*\(", "writes raw HTML into the page, which allows cross-site scripting"),
    (r"\bpickle\.loads?\s*\(|ObjectInputStream|\byaml\.load\s*\((?![^)]*Loader)", "deserialises data that could be untrusted"),
    (r"\b(?:md5|sha1)\b|\bMD5\b|\bSHA1\b|DES/|\bRandom\s*\(\)", "uses a weak hash, cipher or random source"),
    (r"(?i)(?:password|passwd|secret|api_?key|token)\w*\s*[:=]\s*[\"'][^\"'\s]{6,}[\"']", "has a hardcoded credential"),
]


def looks_like_code(text: str) -> bool:
    """True when the message carries a real piece of code, not just a mention of one."""
    marks = len(re.findall(r"[{};]|=>|\(\)|==|!=|\+\+|--", text))
    words = len(re.findall(r"\b(?:import|class|def|function|return|public|static|void|const|let|var|if|else|for|"
                           r"while|switch|case|include|println|printf|print)\b", text))
    indented = len(re.findall(r"^\s*(?:def \w+\(|class \w+|import \w|from [\w.]+ import|for .+:\s*$|if .+:\s*$|"
                              r"while .+:\s*$|return\b|[\w.\[\]]+ = \S)", text, re.M))
    return len(text) >= 80 and ((marks >= 6 and words >= 4) or indented >= 4)


def language(code: str) -> str:
    best, hits = "", 0
    for name, pattern in LANGUAGES:
        n = len(re.findall(pattern, code, re.M))
        if n > hits:
            best, hits = name, n
    return best


def _names(pattern: str, code: str, limit: int = 8) -> List[str]:
    seen: List[str] = []
    for name in re.findall(pattern, code, re.M):
        if name not in seen and name not in NOT_NAMES:
            seen.append(name)
    return seen[:limit]


def _list(items: List[str]) -> str:
    items = [f"`{i}`" for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "es" if word.endswith(("ch", "s")) else "s")


def _imports(code: str) -> List[str]:
    found = re.findall(r"\bfrom\s+[\"']([^\"'\n]+)[\"']|\brequire\(\s*[\"']([^\"'\n]+)[\"']", code)
    bare = re.sub(r"\"[^\"\n]*\"|'[^'\n]*'", '""', code)
    found += re.findall(r"(?:^|[\s;(])(?:import|using)\s+(?:static\s+)?([\w.]+)\s*;()", bare)
    found += re.findall(r"^\s*import\s+([\w.]+)()|^\s*from\s+([\w.]+)\s+import", bare, re.M)
    found += re.findall(r"#include\s*[<\"]([\w./]+)()", code)
    names = [next((g for g in groups if g), "") for groups in found]
    return list(dict.fromkeys(n for n in names if n))[:6]


def _body(name: str, code: str) -> str:
    """The text of one function: brace matched, or by indentation for Python-style code."""
    m = re.search(rf"\b{re.escape(name)}\s*\([^)]*\)[^{{;\n]*\{{", code)
    if m:
        depth, i = 1, m.end()
        while i < len(code) and depth:
            depth += {"{": 1, "}": -1}.get(code[i], 0)
            i += 1
        return code[m.end():i - 1]
    m = re.search(rf"^([ \t]*)def\s+{re.escape(name)}\s*\(.*?\).*?:\s*$", code, re.M)
    if not m:
        return ""
    lines = []
    for line in code[m.end():].splitlines()[1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= len(m.group(1)):
            break
        lines.append(line)
    return "\n".join(lines)


def _steps(name: str, body: str, funcs: List[str]) -> List[str]:
    """What one function does, read straight from its body."""
    steps = []
    if READS.search(body):
        steps.append("reads input")
    if re.search(r"\b(?:for|while|do)\b", body):
        steps.append("loops")
    cond = re.search(r"\bif\s*\(?\s*([^{:\n]{3,40}?)\s*\)?\s*[{:]", body)
    if cond:
        steps.append(f"checks `{cond.group(1).strip()}`")
    verbs = {"++": "adds 1 to", "--": "takes 1 from", "+=": "adds to", "-=": "subtracts from"}
    for var, op in dict.fromkeys(re.findall(r"\b([A-Za-z_]\w*)\s*(\+\+|--|\+=|-=)", body)):
        steps.append(f"{verbs[op]} `{var}`")
    calls = [f for f in funcs if f != name and re.search(rf"\b{re.escape(f)}\s*\(", body)]
    if calls:
        steps.append(f"calls {_list(calls)}")
    said = [m for m in PRINTS.findall(body) if re.search(r"[A-Za-z]{3}", m)]
    if said:
        steps.append(f'prints "{said[0].strip()}"')
    return steps


def outline(code: str) -> List[str]:
    """Plain statements about the code, each one checkable against the text."""
    lang, out = language(code), []
    types = _names(r"\b(?:class|interface|struct|enum|trait)\s+([A-Za-z_]\w*)", code)
    funcs = _names(r"\b(?:def|function|func|fn)\s+([A-Za-z_]\w*)\s*\(", code)
    funcs += [n for n in _names(r"(?:\b[\w<>\[\],.]+\s+)+([A-Za-z_]\w*)\s*\([^;{)]*\)\s*(?:throws\s+[\w.,\s]+)?\{", code)
              if n not in funcs and n not in types]
    lead = f"This is {lang} code" if lang else "This is code in a language I could not identify for sure"
    if types:
        lead += f" defining {_list(types)}"
    out.append(lead + ".")
    if funcs:
        out.append(f"It has {_count(len(funcs), 'function')}: {_list(funcs)}." + (
            " `main` is where it starts." if "main" in funcs else ""))
    imports = _imports(code)
    if imports:
        out.append(f"It imports {_list(imports)}.")
    for name in funcs[:6]:
        steps = _steps(name, _body(name, code), funcs)
        if steps:
            out.append(f"`{name}` " + (steps[0] if len(steps) == 1 else ", ".join(steps[:-1]) + " and " + steps[-1]) + ".")
    constants = re.findall(r"\b(?:static\s+final\s+\w+|const(?:\s+\w+)?|#define)\s+([A-Z][A-Z0-9_]*)\s*=?\s*([^;\n,]{1,24})", code)
    constants += [c for c in re.findall(r"^([A-Z][A-Z0-9_]{2,})\s*=\s*([^\n]{1,24})", code, re.M) if c not in constants]
    if constants:
        hidden = re.compile(r"(?i)password|passwd|secret|key|token")
        out.append("Fixed values: " + ", ".join(
            f"`{n}` (value hidden, it looks like a credential)" if hidden.search(n) else f"`{n}` = `{v.strip()}`"
            for n, v in constants[:5]) + ".")

    flow = []
    if re.search(r"\bdo\s*\{", code) and re.search(r"\}\s*while\s*\(", code):
        flow.append("a do-while loop that repeats until its condition fails")
    loops = len(re.findall(r"\b(?:for|foreach)\b\s*\(?", code)) + len(re.findall(r"(?<!\}\s)\bwhile\b", code))
    loops -= 1 if flow else 0
    if loops > 0:
        flow.append(_count(loops, "loop"))
    cases = len(re.findall(r"\bcase\s+[^:\n]+:", code))
    if re.search(r"\b(?:switch|match)\b", code) and cases:
        flow.append(f"a switch with {_count(cases, 'case')}" + (" and a default" if re.search(r"\bdefault\s*:", code) else ""))
    branches = len(re.findall(r"\bif\b", code))
    if branches:
        flow.append(_count(branches, "if check"))
    if re.search(r"\b(?:try|except|catch)\b", code):
        flow.append("error handling")
    if flow:
        out.append("Control flow: " + ", ".join(flow) + ".")

    reads = READS.search(code)
    shown = PRINTS.findall(code)
    shown = [s for s in dict.fromkeys(shown) if re.search(r"[A-Za-z]{3}", s)]
    if reads or shown:
        io = "It reads input from the user" if reads else "It"
        if shown:
            io += (" and prints" if reads else " prints") + " messages such as " + ", ".join(f'"{s.strip()}"' for s in shown[:4])
        out.append(io + ".")
    changed = _names(r"\b([A-Za-z_]\w*)\s*(?:\+\+|--|\+=|-=)", code, 5)
    if changed:
        out.append(f"It changes {_list(changed)} as it runs.")

    risks = [note for pattern, note in RISKS if re.search(pattern, code)]
    if risks:
        out.append("Security check: this code " + "; ".join(risks) + ".")
    else:
        out.append("Security check: no risky calls stand out (no eval, shell commands, SQL built from strings, "
                   "disabled TLS checks or hardcoded credentials).")
    out.append("That is a reading of the code's structure. I can't yet explain the intent behind it in my own words reliably.")
    return out
