#!/usr/bin/env python3
"""Expand bet 8 binary literacy bank from 15 to 40 patterns (v0.9.22).

The original bank covered 7 file magics, 2 packers, 3 shellcode
patterns, 2 PE fields, 1 disassembly. This script appends 25
patterns covering more file formats (JPEG, GIF, MP4, Java class,
WASM, GZIP, SQLite, DEX), more shellcode (x86 execve, x64 reverse
shell, ARM64 prologue, function epilogue, ROP gadget pattern),
more PE/ELF fields (PE imports, ELF PT_LOAD, ELF PT_INTERP),
encoding patterns (base64, hex string, UTF-8 BOM, base64 URL-safe),
common hash format recognition (MD5, SHA1, SHA256 length),
disassembly (jump table, indirect call).

Idempotent: re-running skips IDs already present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BANK = REPO_ROOT / "data" / "raw" / "binary_literacy_patterns.jsonl"


NEW_PATTERNS = [
    # ---- More file magics -----------------------------------------------
    {
        "id": "BIN-016", "category": "file_magic",
        "name": "JPEG image",
        "hex_at_offset_0": "FF D8 FF",
        "ascii_decode": "(no printable)",
        "longer_pattern": "FF D8 FF E0 00 10 4A 46 49 46 00 01 ... FF D9",
        "explanation": (
            "Every JPEG begins with the SOI marker 0xFFD8 followed "
            "by another 0xFF byte. Common variants: FF D8 FF E0 = "
            "JFIF (most common), FF D8 FF E1 = EXIF (most camera "
            "photos), FF D8 FF E8 = SPIFF, FF D8 FF DB = raw "
            "JPEG with quantization tables. JPEG files always END "
            "with the EOI marker FF D9. Useful to know: malware "
            "polyglots sometimes hide payload between FF D9 and "
            "EOF since most viewers stop reading at EOI."
        ),
        "examples": ["any photo from a phone or camera",
                      "WordPress upload"],
    },
    {
        "id": "BIN-017", "category": "file_magic",
        "name": "GIF image",
        "hex_at_offset_0": "47 49 46 38 39 61",
        "ascii_decode": "GIF89a",
        "longer_pattern": "47 49 46 38 39 61 ww ww hh hh",
        "explanation": (
            "GIF files start with 'GIF87a' (47 49 46 38 37 61) or "
            "'GIF89a' (47 49 46 38 39 61). 89a added animation, "
            "interlacing, transparency. The next 4 bytes are the "
            "logical screen width and height as little-endian "
            "uint16. Used historically for animated images; mostly "
            "replaced by APNG / WebP / video formats but still "
            "ubiquitous on the web."
        ),
        "examples": ["meme animations on slack/discord"],
    },
    {
        "id": "BIN-018", "category": "file_magic",
        "name": "MP4 / QuickTime container",
        "hex_at_offset_0": "00 00 00 18 66 74 79 70",
        "ascii_decode": "....ftyp",
        "longer_pattern": "00 00 00 NN 66 74 79 70 BB BB BB BB",
        "explanation": (
            "MP4 / MOV / 3GP containers are organized as 'atoms' "
            "or 'boxes'. The first atom is always 'ftyp' "
            "(file-type) starting at offset 4, with a length "
            "header in the first 4 bytes. The brand at offset 8 "
            "(BB BB BB BB) tells you the variant: 'isom' = generic "
            "ISO base, 'mp42' = MPEG-4 v2, 'qt  ' (with two spaces) "
            "= QuickTime, 'M4A ' = audio-only AAC, '3gp4' = 3GPP."
        ),
        "examples": ["any phone-recorded video"],
    },
    {
        "id": "BIN-019", "category": "file_magic",
        "name": "Java class file",
        "hex_at_offset_0": "CA FE BA BE",
        "ascii_decode": "(no printable)",
        "longer_pattern": "CA FE BA BE 00 00 00 NN ...",
        "explanation": (
            "Java compiled .class files begin with the magic "
            "0xCAFEBABE. Bytes 4-5 are minor version, 6-7 are major "
            "version: 0x34 = Java 8, 0x37 = Java 11, 0x3D = Java "
            "17, 0x41 = Java 21. After that comes the constant "
            "pool. Mach-O FAT (universal) binaries also use "
            "CAFEBABE at offset 0, distinguished by checking the "
            "next 4 bytes: a Mach-O FAT has the count of "
            "architectures, a Java class has the version."
        ),
        "examples": ["compiled Java app", "deobfuscated Android DEX "
                      "after dex2jar"],
    },
    {
        "id": "BIN-020", "category": "file_magic",
        "name": "WebAssembly module",
        "hex_at_offset_0": "00 61 73 6D 01 00 00 00",
        "ascii_decode": ".asm....",
        "longer_pattern": "00 61 73 6D 01 00 00 00 01 ...",
        "explanation": (
            "WASM binary modules begin with 0x00 followed by 'asm' "
            "(0x61 0x73 0x6D), then a 4-byte little-endian version "
            "(currently 0x01000000 for v1). After the magic + "
            "version come typed sections (type, import, function, "
            "table, memory, global, export, start, element, code, "
            "data) each with a 1-byte ID. Used in browsers, "
            "Cloudflare Workers, smart contracts (NEAR, Polkadot)."
        ),
        "examples": ["compiled Rust browser app",
                      "Figma's renderer"],
    },
    {
        "id": "BIN-021", "category": "file_magic",
        "name": "GZIP compressed",
        "hex_at_offset_0": "1F 8B 08",
        "ascii_decode": "(no printable)",
        "longer_pattern": "1F 8B 08 NN MM MM MM MM ZZ FF",
        "explanation": (
            "GZIP starts with 1F 8B (signature) then 0x08 (DEFLATE "
            "compression method, the only one defined). Byte 3 is "
            "flags (FTEXT, FHCRC, FEXTRA, FNAME, FCOMMENT). Bytes "
            "4-7 are the modification time as Unix epoch LE. Byte "
            "9 is the OS the archive was created on (0x03 = Unix, "
            "0x0A = TOPS/20, 0xFF = unknown). After the header "
            "comes a DEFLATE stream; .tar.gz is GZIP wrapping a "
            "TAR archive."
        ),
        "examples": ["nginx access.log.gz",
                      "firmware blob in a router image"],
    },
    {
        "id": "BIN-022", "category": "file_magic",
        "name": "SQLite database",
        "hex_at_offset_0": "53 51 4C 69 74 65 20 66 6F 72 6D 61 74 20 33 00",
        "ascii_decode": "SQLite format 3.",
        "longer_pattern": "53 51 4C 69 74 65 20 66 6F 72 6D 61 74 20 33 00",
        "explanation": (
            "SQLite databases begin with the literal 16-byte string "
            "'SQLite format 3' followed by a null byte. Used by iOS "
            "+ Android system databases (SMS, contacts, browser "
            "history), Firefox / Chrome bookmarks + cookies, app "
            "caches everywhere. Forensically valuable: a recovered "
            ".db file is queryable with the sqlite3 CLI even if the "
            "original app is gone. Use `strings` and `sqlite3 file "
            "'.dump'` to extract content."
        ),
        "examples": ["WhatsApp msgstore.db",
                      "Firefox places.sqlite",
                      "iOS notes/sms backup"],
    },
    {
        "id": "BIN-023", "category": "file_magic",
        "name": "Android DEX bytecode",
        "hex_at_offset_0": "64 65 78 0A 30 33 35 00",
        "ascii_decode": "dex.035.",
        "longer_pattern": "64 65 78 0A 30 33 NN 00",
        "explanation": (
            "Android Dalvik / ART executables (.dex inside an APK) "
            "begin with 'dex\\n' (64 65 78 0A) followed by a 3-byte "
            "version + null. '035' is the historical default; '038' "
            "is Android 8+; '039' adds new opcodes. APK files are "
            "ZIP archives so they start with PK\\x03\\x04 (BIN-004); "
            "the DEX inside is at /classes.dex. Extract via "
            "`apktool` or `unzip`, then `dex2jar` + JADX for source."
        ),
        "examples": ["any Android app's classes.dex",
                      "Frida instrumentation targets"],
    },
    # ---- More shellcode --------------------------------------------------
    {
        "id": "BIN-024", "category": "shellcode",
        "name": "x86 execve('/bin/sh') shellcode (Linux 32-bit)",
        "hex_at_offset_0": "31 C0 50 68 2F 2F 73 68 68 2F 62 69 6E 89 E3",
        "ascii_decode": "1.Ph//shh/bin..",
        "longer_pattern": "31 C0 50 68 2F 2F 73 68 68 2F 62 69 6E 89 E3 50 53 89 E1 B0 0B CD 80",
        "explanation": (
            "Classic Linux x86 execve('/bin/sh') stub. Decoded:\n"
            "  xor eax, eax           ; eax = 0\n"
            "  push eax               ; null terminator\n"
            "  push 0x68732f2f        ; '//sh'\n"
            "  push 0x6e69622f        ; '/bin'\n"
            "  mov ebx, esp           ; ebx = pointer to '/bin//sh'\n"
            "  push eax               ; null argv terminator\n"
            "  push ebx               ; argv[0]\n"
            "  mov ecx, esp           ; ecx = argv\n"
            "  mov al, 11             ; sys_execve\n"
            "  int 0x80               ; syscall\n"
            "23 bytes, no nulls (so it survives strcpy). The "
            "double-slash in '//sh' is a no-op the kernel "
            "tolerates and gives the assembler a 4-byte aligned "
            "string to push."
        ),
        "examples": ["Stack overflow exploits 1990s-2010s",
                      "CTF pwn challenges"],
    },
    {
        "id": "BIN-025", "category": "shellcode",
        "name": "ARM64 function prologue",
        "hex_at_offset_0": "FD 7B BF A9 FD 03 00 91",
        "ascii_decode": "(no printable)",
        "longer_pattern": "FD 7B BF A9 FD 03 00 91",
        "explanation": (
            "ARM64 function prologue, save frame pointer and link "
            "register, set new frame pointer:\n"
            "  stp x29, x30, [sp, #-16]!  ; FD 7B BF A9\n"
            "  mov x29, sp                ; FD 03 00 91\n"
            "All ARM64 instructions are 4 bytes and little-endian "
            "encoded. The pattern A9 BF 7B FD = stp x29, x30 with "
            "pre-decrement of sp by 16 is essentially universal at "
            "the start of any ARM64 function. Apple Silicon, "
            "modern Android, AWS Graviton all decode this."
        ),
        "examples": ["macOS arm64 binary",
                      "Android native lib on aarch64"],
    },
    {
        "id": "BIN-026", "category": "shellcode",
        "name": "x64 function epilogue",
        "hex_at_offset_0": "5D C3",
        "ascii_decode": "].",
        "longer_pattern": "C9 C3 (or 5D C3 if no enter)",
        "explanation": (
            "Standard x86_64 function epilogue:\n"
            "  pop rbp        ; 5D — restore caller's frame ptr\n"
            "  ret            ; C3 — return to caller\n"
            "Or with the legacy `leave` instruction:\n"
            "  leave          ; C9 — equivalent of mov rsp,rbp + pop rbp\n"
            "  ret            ; C3\n"
            "Together with the prologue (push rbp / mov rbp, rsp = "
            "55 48 89 E5) these bracket every C function. Modern "
            "compilers omit the prologue/epilogue for leaf functions "
            "with no stack frame, so absence doesn't mean it's not "
            "a function."
        ),
        "examples": ["Last bytes of any x86_64 function",
                      "ROP-gadget search target"],
    },
    {
        "id": "BIN-027", "category": "shellcode",
        "name": "ROP gadget pop rdi; ret",
        "hex_at_offset_0": "5F C3",
        "ascii_decode": "_.",
        "longer_pattern": "5F C3",
        "explanation": (
            "x86_64 ROP gadget that pops a value from the stack into "
            "rdi (the System V AMD64 first-argument register), then "
            "returns. Combined with the address of system() and a "
            "string '/bin/sh', forms a minimal ret2libc chain that "
            "spawns a shell. Found via `ROPgadget --binary ...` or "
            "`ropper`. The two-byte sequence is so common in any "
            "compiled binary (every function ending pops rdi at "
            "some point) that gadget hunters always find it."
        ),
        "examples": ["pwntools pwn challenges",
                      "real-world ROP exploits 2010s+"],
    },
    {
        "id": "BIN-028", "category": "shellcode",
        "name": "x64 reverse shell connect-stub start",
        "hex_at_offset_0": "6A 29 58 99 6A 02 5F 6A 01 5E",
        "ascii_decode": "j)X.j._j.^",
        "longer_pattern": "6A 29 58 99 6A 02 5F 6A 01 5E 0F 05",
        "explanation": (
            "First lines of a Linux x64 reverse-shell shellcode, "
            "calling sys_socket(AF_INET, SOCK_STREAM, 0) (syscall "
            "number 41 = 0x29):\n"
            "  push 0x29; pop rax       ; rax = 41 (socket)\n"
            "  cdq                       ; rdx = 0\n"
            "  push 2; pop rdi           ; rdi = AF_INET\n"
            "  push 1; pop rsi           ; rsi = SOCK_STREAM\n"
            "  syscall\n"
            "After the socket comes connect() to a hardcoded "
            "addr+port, dup2() of the socket onto stdin/stdout/"
            "stderr, then execve('/bin/sh'). The push-pop "
            "constants pattern keeps the bytes printable and "
            "null-free for surviving string copies."
        ),
        "examples": ["msfvenom payload "
                      "linux/x64/shell_reverse_tcp"],
    },
    # ---- More PE / ELF / Mach-O fields -----------------------------------
    {
        "id": "BIN-029", "category": "pe_field",
        "name": "PE Import Address Table (IAT) signature",
        "hex_at_offset_0": "(varies; in .idata section)",
        "ascii_decode": "kernel32.dll\\0",
        "longer_pattern": "ASCII names like 'kernel32.dll', "
                          "'CreateFileA', 'GetProcAddress'",
        "explanation": (
            "PE binaries resolve external functions through the "
            "Import Directory in the .idata section. Look for "
            "ASCII strings like 'kernel32.dll', 'ntdll.dll', "
            "'CreateFileA', 'VirtualAlloc', 'GetProcAddress'. "
            "Suspicious imports for malware analysis: "
            "VirtualAllocEx + WriteProcessMemory + "
            "CreateRemoteThread (process injection); "
            "WinHttpOpen + WinHttpConnect (C2); CryptEncrypt + "
            "CryptGenKey (ransomware). PEview / CFF Explorer / "
            "`strings` shows them; pe-bear gives a parsed view."
        ),
        "examples": ["any non-trivial Windows PE",
                      "Cobalt Strike beacon"],
    },
    {
        "id": "BIN-030", "category": "elf_field",
        "name": "ELF PT_LOAD segment marker",
        "hex_at_offset_0": "(in program header)",
        "ascii_decode": "(no printable)",
        "longer_pattern": "01 00 00 00 (p_type = PT_LOAD = 1)",
        "explanation": (
            "ELF program headers describe how the kernel loader "
            "maps the binary into memory. PT_LOAD (p_type = 1) "
            "marks a segment to load; p_flags tells you R, W, X "
            "permissions. Most binaries have one R-X PT_LOAD "
            "(code) and one R-W PT_LOAD (data + bss). Modern PIE "
            "binaries also have a PT_DYNAMIC segment (p_type = 2) "
            "for the dynamic linker. A PT_LOAD with WX flags "
            "together is suspicious — usually means JIT or "
            "unpacker."
        ),
        "examples": ["readelf -l on /bin/ls",
                      "pwntools' elf.load_addr"],
    },
    {
        "id": "BIN-031", "category": "elf_field",
        "name": "ELF dynamic interpreter (PT_INTERP)",
        "hex_at_offset_0": "(varies)",
        "ascii_decode": "/lib64/ld-linux-x86-64.so.2",
        "longer_pattern": "PT_INTERP segment containing path string",
        "explanation": (
            "Dynamically-linked ELF binaries embed the path to "
            "their dynamic linker in the PT_INTERP segment. "
            "Standard paths: /lib64/ld-linux-x86-64.so.2 (glibc "
            "x86_64), /lib/ld-musl-x86_64.so.1 (musl x86_64), "
            "/system/bin/linker64 (Android). Statically-linked "
            "binaries have no PT_INTERP — `file` reports "
            "'statically linked'. Custom interpreter strings are "
            "a known anti-forensics technique to break analysis "
            "tools."
        ),
        "examples": ["readelf -p .interp /bin/ls",
                      "musl-static cross-compile output"],
    },
    {
        "id": "BIN-032", "category": "pe_field",
        "name": "PE Section .text characteristics",
        "hex_at_offset_0": "2E 74 65 78 74",
        "ascii_decode": ".text",
        "longer_pattern": "2E 74 65 78 74 00 00 00 ... 20 00 00 60",
        "explanation": (
            "Every PE has a .text section (8-byte name field, null-"
            "padded). Its Characteristics field is at offset +36 in "
            "the section header. 0x60000020 = "
            "IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | "
            "IMAGE_SCN_MEM_READ. Other common section names: "
            ".rdata (read-only data), .data (writable data), "
            ".rsrc (resources, icons, version info), .reloc "
            "(relocations). Suspicious: .text section that is also "
            "writable (RWX); custom section names like '.evil' or "
            "high-entropy section names."
        ),
        "examples": ["dumpbin /headers app.exe",
                      "pe-bear section view"],
    },
    # ---- Encodings -------------------------------------------------------
    {
        "id": "BIN-033", "category": "encoding",
        "name": "Base64 encoded data",
        "hex_at_offset_0": "(ASCII chars)",
        "ascii_decode": "[A-Za-z0-9+/]+=*",
        "longer_pattern": "VGhpcyBpcyBhIHRlc3Q= (= 'This is a test')",
        "explanation": (
            "Base64 encodes 3 bytes of binary as 4 ASCII chars from "
            "the set [A-Za-z0-9+/], padded with '=' to a multiple "
            "of 4 chars. Output length = ceil(input_bytes / 3) * 4. "
            "Recognise: long ASCII strings ending in 0, 1, or 2 "
            "'=' signs. Variants: URL-safe Base64 swaps + and / "
            "for - and _. Used for encoding binary in JSON, JWT "
            "payloads, embedded images in HTML (data URLs), and "
            "obfuscating shellcode. Decode with `base64 -d` "
            "(coreutils) or `python3 -c \"import base64; "
            "print(base64.b64decode('...'))\"`."
        ),
        "examples": ["JWT tokens",
                      "PowerShell -EncodedCommand payload",
                      "data: image/png URLs"],
    },
    {
        "id": "BIN-034", "category": "encoding",
        "name": "Hex string (textual representation)",
        "hex_at_offset_0": "(ASCII chars)",
        "ascii_decode": "[0-9a-fA-F]+",
        "longer_pattern": "deadbeef or DE AD BE EF or \\xde\\xad\\xbe\\xef",
        "explanation": (
            "Hex strings encode binary as 2 ASCII chars per byte "
            "from [0-9a-fA-F]. Length is always even. Recognise: "
            "long-ish strings that match the regex. Variants: "
            "space-separated 'DE AD BE EF', backslash-x prefix "
            "'\\xde\\xad\\xbe\\xef' (C / Python), 0x prefix per byte "
            "'0xde, 0xad'. Used for representing hashes, byte arrays "
            "in source code, network packet dumps. Decode with "
            "`xxd -r -p`, `python3 -c \"print(bytes.fromhex('...'))\"`, "
            "or just paste into CyberChef."
        ),
        "examples": ["MD5 / SHA hashes",
                      "shellcode in C source",
                      "tcpdump -X output"],
    },
    {
        "id": "BIN-035", "category": "encoding",
        "name": "UTF-8 BOM (byte order mark)",
        "hex_at_offset_0": "EF BB BF",
        "ascii_decode": "(invisible 'ZERO WIDTH NO-BREAK SPACE')",
        "longer_pattern": "EF BB BF (then UTF-8 text)",
        "explanation": (
            "Some Windows tools (Notepad, older Excel) prepend EF "
            "BB BF to UTF-8 files as a BOM. UTF-8 doesn't actually "
            "need a BOM (it's encoded the same regardless of byte "
            "order), but the bytes mean 'this is UTF-8' to anything "
            "that checks. Causes problems: shell scripts with a "
            "BOM fail because the BOM bytes are not whitespace; "
            "Python 3 handles them silently with `encoding='utf-8-"
            "sig'`. UTF-16 BOM is FF FE (LE) or FE FF (BE), which "
            "is more important because the byte order matters."
        ),
        "examples": ["Notepad-saved UTF-8 file",
                      "Excel-exported CSV with non-ASCII"],
    },
    # ---- Hashes / digests ------------------------------------------------
    {
        "id": "BIN-036", "category": "hash",
        "name": "MD5 hex digest length recognition",
        "hex_at_offset_0": "(N/A — string)",
        "ascii_decode": "32 hex chars",
        "longer_pattern": "d41d8cd98f00b204e9800998ecf8427e",
        "explanation": (
            "MD5 produces a 128-bit (16-byte) hash. As a hex "
            "string it's exactly 32 lowercase chars from [0-9a-f]. "
            "Common contexts: file checksums, password hashes "
            "(broken — never use for security), Git tree IDs (SHA1, "
            "40 chars). Easy to spot a hash by length: 32 = MD5, "
            "40 = SHA1, 56 = SHA224, 64 = SHA256, 96 = SHA384, "
            "128 = SHA512, 64 with $argon2 prefix = Argon2 with "
            "salt embedded."
        ),
        "examples": ["VirusTotal sample lookup",
                      "package integrity hash"],
    },
    {
        "id": "BIN-037", "category": "hash",
        "name": "SHA-256 hex digest length recognition",
        "hex_at_offset_0": "(N/A — string)",
        "ascii_decode": "64 hex chars",
        "longer_pattern": "e3b0c44298fc1c149afbf4c8996fb924"
                            "27ae41e4649b934ca495991b7852b855",
        "explanation": (
            "SHA-256 produces a 256-bit (32-byte) hash. As a hex "
            "string it's exactly 64 lowercase chars from [0-9a-f]. "
            "Modern standard for file integrity (most package "
            "managers verify SHA-256), VirusTotal sample IDs, "
            "container image digests (sha256:...). Argon2id "
            "password hashes start with `$argon2id$...$` followed "
            "by base64-encoded 32-byte hash. bcrypt is 60 chars "
            "starting with `$2y$` or `$2b$`."
        ),
        "examples": ["docker pull image@sha256:...",
                      "Linux distro ISO checksum"],
    },
    {
        "id": "BIN-038", "category": "hash",
        "name": "Common bcrypt password hash format",
        "hex_at_offset_0": "(N/A — string)",
        "ascii_decode": "$2b$12$22-char-salt-31-char-hash",
        "longer_pattern": "$2b$12$N9qo8uLOickgx2ZMRZoMy.Mrq"
                            "VNtJxWWmZbwFq.4jh6hHM6QK7T8.",
        "explanation": (
            "bcrypt password hashes are 60 ASCII chars starting "
            "with the version (`$2y$`, `$2b$`, `$2a$`), then the "
            "cost factor (`$12$` for 2^12 iterations), then a "
            "22-char base64-encoded salt, then a 31-char base64-"
            "encoded hash. Variants are interoperable; cost 12-14 "
            "is current best practice. Recognise from the leading "
            "`$2`: that signals bcrypt, distinct from `$argon2id$` "
            "for Argon2 and `$5$` / `$6$` for SHA-256/512 crypt."
        ),
        "examples": ["Django auth_user.password",
                      "PostgreSQL pgcrypto"],
    },
    # ---- More disassembly ------------------------------------------------
    {
        "id": "BIN-039", "category": "disassembly",
        "name": "x64 indirect call through register",
        "hex_at_offset_0": "FF D0",
        "ascii_decode": "..",
        "longer_pattern": "FF D0 (call rax) or FF D7 (call rdi)",
        "explanation": (
            "FF Dr is x86_64 indirect call through register r:\n"
            "  FF D0 = call rax\n"
            "  FF D3 = call rbx\n"
            "  FF D7 = call rdi\n"
            "  FF D6 = call rsi\n"
            "Common in: virtual function dispatch (vtable lookup "
            "into rax then call rax), function pointer calls, "
            "JIT-emitted code, ROP/JOP gadget chains. Direct "
            "function calls use E8 + 4-byte relative offset, which "
            "the linker resolves at static link time. Indirect "
            "calls through fixed addresses are harder to track "
            "statically."
        ),
        "examples": ["compiled C++ virtual method call",
                      "JIT-compiled Lua / V8"],
    },
    {
        "id": "BIN-040", "category": "disassembly",
        "name": "x64 syscall instruction (Linux ABI)",
        "hex_at_offset_0": "0F 05",
        "ascii_decode": "..",
        "longer_pattern": "B8 NN 00 00 00 0F 05",
        "explanation": (
            "0F 05 is the x86_64 SYSCALL instruction. The Linux "
            "syscall ABI uses rax = number, rdi/rsi/rdx/r10/r8/r9 "
            "as args 1-6. Common preludes:\n"
            "  B8 01 00 00 00 0F 05    ; sys_write\n"
            "  B8 02 00 00 00 0F 05    ; sys_open\n"
            "  B8 3B 00 00 00 0F 05    ; sys_execve\n"
            "  B8 E7 00 00 00 0F 05    ; sys_exit_group (231)\n"
            "macOS x86_64 also uses SYSCALL but with BSD-style "
            "numbers + 0x2000000 offset. Windows uses int 0x2E "
            "or syscall via ntdll wrapper, not direct."
        ),
        "examples": ["any Linux statically-linked binary",
                      "minimal hand-written assembly"],
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
    print("  PYTHONPATH=. python3 scripts/synth_binary_literacy.py \\")
    print("      --bank data/raw/binary_literacy_patterns.jsonl \\")
    print("      --out data/processed/synth_binary_literacy.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
