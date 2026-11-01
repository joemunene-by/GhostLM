#!/usr/bin/env python3
"""Template-written chat data where the user pastes a security artifact and GhostLM reads it.

    python scripts/synth_artifact_chats.py [--per-kind 40] [--seed 0]

Each conversation is built from a randomised but internally consistent artifact (an
auth.log excerpt, an nmap report, listening sockets, Windows Security events, a web
access log, a CVSS record, a Sigma rule, a directory listing) and an explanation that
is derived from the same random choices, so the answer is right by construction. No
teacher model is involved. Records land in data/chat_gen/artifacts.jsonl with source
``gen_artifacts`` and still go through the review gate like everything else.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from check_seed import problems  # noqa: E402

OUT = ROOT / "data/chat_gen/artifacts.jsonl"
HOST_NAMES = ["web-01", "bastion", "db-02", "jump", "app-prod-3", "build-runner", "mail", "ns1"]
TEST_NETS = ["203.0.113.", "198.51.100.", "192.0.2."]
USERS = ["admin", "root", "test", "oracle", "postgres", "ubuntu", "pi", "deploy", "git", "backup"]
REAL_USERS = ["deploy", "ops", "jmwangi", "svc-backup", "ansible"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def ip(rng: random.Random) -> str:
    return rng.choice(TEST_NETS) + str(rng.randint(2, 250))


def stamp(rng: random.Random, base_minute: int, offset_s: int) -> str:
    total = base_minute * 60 + offset_s
    return f"{(total // 3600) % 24:02d}:{(total // 60) % 60:02d}:{total % 60:02d}"


def ask(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


def ssh_bruteforce(rng: random.Random) -> list[dict]:
    host, src, month, day = rng.choice(HOST_NAMES), ip(rng), rng.choice(MONTHS), rng.randint(1, 28)
    n = rng.randint(6, 14)
    base = rng.randint(60, 1380)
    users = rng.sample(USERS, k=min(rng.randint(2, 5), len(USERS)))
    success_user = rng.choice(REAL_USERS) if rng.random() < 0.4 else None
    lines, t = [], 0
    for i in range(n):
        t += rng.randint(1, 4)
        u = users[i % len(users)]
        invalid = "invalid user " if u not in REAL_USERS else ""
        lines.append(f"{month} {day:2d} {stamp(rng, base, t)} {host} sshd[{rng.randint(1000, 60000)}]: "
                     f"Failed password for {invalid}{u} from {src} port {rng.randint(30000, 65000)} ssh2")
    if success_user:
        t += rng.randint(2, 6)
        lines.append(f"{month} {day:2d} {stamp(rng, base, t)} {host} sshd[{rng.randint(1000, 60000)}]: "
                     f"Accepted password for {success_user} from {src} port {rng.randint(30000, 65000)} ssh2")
    log = "\n".join(lines)
    user = ask(rng, [
        f"Found this in /var/log/auth.log on {host}, what is happening?\n\n```\n{log}\n```",
        f"can you read this auth log for me\n\n```\n{log}\n```",
        f"Is this an attack or just someone fat-fingering a password?\n\n```\n{log}\n```",
    ])
    span = t
    answer = (f"This is an SSH password brute-force attempt from {src}: {n} failed logins in about {span} seconds "
              f"against {len(users)} different usernames ({', '.join(users)}), which no human typing a password does. "
              f"The \"invalid user\" entries show the attacker is guessing common account names, not targeting "
              f"accounts they know exist.")
    if success_user:
        answer += (f"\n\nThe last line matters most: `Accepted password for {success_user}` from the same IP means a "
                   f"guess worked and {success_user} is compromised. Treat the host as breached: terminate that "
                   f"session (`who` then `pkill -KILL -u {success_user}` or kill the sshd child), lock the account "
                   f"(`passwd -l {success_user}`), rotate its password and any keys or tokens it could reach, and "
                   f"check `last`, `~{success_user}/.bash_history`, crontabs and `~/.ssh/authorized_keys` for what "
                   f"was done after {stamp(rng, base, t)}.")
    else:
        answer += ("\n\nNothing succeeded in this excerpt, so no account is known to be compromised, but check the "
                   "rest of the log for an `Accepted` line from the same address.")
    answer += (f"\n\nTo stop it: block the source now (`sudo ufw deny from {src}` or an iptables DROP), then make the "
               "class of attack impossible by setting `PasswordAuthentication no` and `PermitRootLogin no` in "
               "`/etc/ssh/sshd_config` and using keys only. fail2ban with the sshd jail bans repeat offenders "
               "automatically, and moving SSH behind a VPN or allowlist removes the exposure entirely.")
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


NMAP_PORTS = {
    21: ("ftp", "vsftpd 2.3.4", "vsftpd 2.3.4 is the release that shipped with a backdoor (CVE-2011-2523): a username "
                                "containing `:)` opens a root shell on port 6200. Take it down or upgrade immediately"),
    22: ("ssh", "OpenSSH 9.6", None),
    23: ("telnet", "Linux telnetd", "telnet sends credentials and sessions in cleartext; anyone on the path reads them. "
                                    "Replace it with SSH"),
    80: ("http", "nginx 1.24.0", None),
    443: ("https", "nginx 1.24.0", None),
    445: ("microsoft-ds", "Samba smbd 4.x", "SMB exposed to untrusted networks is how worms like WannaCry spread and a "
                                             "favourite for credential relay; it should never be reachable from outside"),
    3306: ("mysql", "MySQL 8.0.36", "a database listening to the world invites brute force and exploitation of any auth "
                                   "bug; bind it to localhost or a private interface"),
    3389: ("ms-wbt-server", "Microsoft Terminal Services", "RDP on the internet gets brute-forced within hours and has had "
                                                           "wormable bugs (BlueKeep, CVE-2019-0708); put it behind a VPN"),
    5900: ("vnc", "RealVNC 6.x", "VNC often runs with weak or no authentication and is unencrypted; it should not be exposed"),
    6379: ("redis", "Redis 7.2.4", "Redis has no authentication by default, so an exposed instance means anyone can read, "
                                  "wipe or use it to write files such as SSH keys; bind it to localhost or require AUTH"),
    8080: ("http-proxy", "Jetty 10.x", None),
}


def nmap_scan(rng: random.Random) -> list[dict]:
    target = ip(rng)
    risky = rng.sample([p for p, v in NMAP_PORTS.items() if v[2]], k=rng.randint(1, 3))
    safe = rng.sample([p for p, v in NMAP_PORTS.items() if not v[2]], k=rng.randint(1, 3))
    ports = sorted(risky + safe)
    lines = [f"Nmap scan report for {target}", "Host is up (0.023s latency).", "", "PORT      STATE SERVICE        VERSION"]
    for p in ports:
        svc, ver, _ = NMAP_PORTS[p]
        lines.append(f"{p}/tcp".ljust(10) + "open  " + svc.ljust(15) + ver)
    report = "\n".join(lines)
    user = ask(rng, [
        f"I scanned one of our servers from outside. Anything here I should worry about?\n\n```\n{report}\n```",
        f"nmap -sV result for a box I'm responsible for, what's bad here\n\n```\n{report}\n```",
        f"Please rank these open ports by risk.\n\n```\n{report}\n```",
    ])
    answer = f"Yes. Of the {len(ports)} open ports, {len(risky)} should not be reachable from the internet at all.\n\n"
    for p in sorted(risky, key=lambda x: (x not in (21, 6379, 23), x)):
        svc, ver, why = NMAP_PORTS[p]
        answer += f"- {p}/tcp {svc} ({ver}): {why}.\n"
    fine = ", ".join(f"{p}/tcp {NMAP_PORTS[p][0]}" for p in sorted(safe))
    answer += (f"\nThe rest ({fine}) are normal for a public server as long as they are patched and SSH uses keys "
               "only. Fix order: close or firewall the risky ports first, then confirm with a rescan from outside, "
               "then check the logs of those services for signs they were already found, because a scan that "
               "took you seconds has been run against this host by others.")
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def listening_sockets(rng: random.Random) -> list[dict]:
    rows = [("0.0.0.0:22", "sshd", "expected"), ("127.0.0.53:53", "systemd-resolve", "expected"),
            ("0.0.0.0:443", "nginx", "expected")]
    suspicious = rng.sample([
        ("0.0.0.0:4444", "python3", "4444 is the default Metasploit handler port and a Python process listening on all "
                                   "interfaces on it is a classic reverse-shell or bind-shell listener"),
        ("0.0.0.0:31337", "nc", "31337 is the historic Back Orifice port and netcat listening there is not a service "
                                "any package installs"),
        ("0.0.0.0:3306", "mysqld", "MySQL bound to 0.0.0.0 is reachable from every network the host is on; it should be "
                                   "127.0.0.1 unless remote clients are intended, and then firewalled to them"),
        ("0.0.0.0:6379", "redis-server", "Redis on all interfaces with its default of no authentication lets anyone "
                                         "who can reach the port read and write the data"),
        ("0.0.0.0:2222", "sshd", "a second sshd on 2222 that nobody configured is a common attacker persistence "
                                 "trick, often with its own authorized_keys"),
    ], k=rng.randint(1, 2))
    rows += [(a, p, "bad") for a, p, _ in suspicious]
    rng.shuffle(rows)
    lines = ["State   Recv-Q  Send-Q  Local Address:Port   Peer Address:Port  Process"]
    for addr, proc, _ in rows:
        lines.append(f"LISTEN  0       128     {addr.ljust(20)} 0.0.0.0:*          users:((\"{proc}\",pid={rng.randint(300, 40000)},fd={rng.randint(3, 12)}))")
    out = "\n".join(lines)
    user = ask(rng, [
        f"Output of `ss -tlnp` on a server, does anything look off?\n\n```\n{out}\n```",
        f"which of these listening ports are normal\n\n```\n{out}\n```",
    ])
    answer = "Two things are normal and one or two are not.\n\nExpected: sshd on 22, systemd-resolved on 127.0.0.53:53 (the local stub resolver, loopback only) and nginx on 443.\n\n"
    for addr, proc, why in suspicious:
        answer += f"Investigate: `{proc}` on {addr}. {why}.\n"
    first_addr, first_proc, _ = suspicious[0]
    pid_hint = "`ss -tlnp` gave you the PID; run `ls -l /proc/<pid>/exe`, `cat /proc/<pid>/cmdline | tr '\\0' ' '` and `ps -o user,lstart,cmd -p <pid>`"
    answer += (f"\n\nStart with {first_proc} on {first_addr}: {pid_hint} to see what binary it is, who started it and "
               "when. If it is not something you or a package put there, treat the host as compromised: capture the "
               "process and its network connections (`ss -tnp`), then look for how it persists (crontab -l for every "
               "user, /etc/systemd/system, ~/.ssh/authorized_keys) before you kill it, or it will simply come back.")
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def windows_events(rng: random.Random) -> list[dict]:
    src, host = ip(rng), rng.choice(["WS-FIN-07", "SRV-FILE01", "DC01", "WS-HR-12"])
    target = rng.choice(["administrator", "jsmith", "svc_sql", "backupadmin"])
    n = rng.randint(5, 9)
    persist = rng.choice(["4720", "7045", "1102"])
    lines = []
    for i in range(n):
        status = "0xC000006A" if i < n - 1 or rng.random() < 0.7 else "0xC0000064"
        lines.append(f"4625  An account failed to log on. Account Name: {target}  Logon Type: 3  Status: {status}  Source Network Address: {src}")
    lines.append(f"4624  An account was successfully logged on. Account Name: {target}  Logon Type: 10  Source Network Address: {src}")
    if persist == "4720":
        lines.append(f"4720  A user account was created. New Account: support$  Subject: {target}")
        lines.append("4732  A member was added to a security-enabled local group. Group: Administrators  Member: support$")
    elif persist == "7045":
        lines.append("7045  A service was installed in the system. Service Name: WinUpdateSvc  Service File Name: C:\\Users\\Public\\svchost.exe  (System log)")
    else:
        lines.append(f"1102  The audit log was cleared. Subject: {target}")
    log = "\n".join(lines)
    user = ask(rng, [
        f"Security log from {host}, event IDs on the left. What story does this tell?\n\n```\n{log}\n```",
        f"our SIEM flagged this sequence on {host}, is it bad?\n\n```\n{log}\n```",
    ])
    answer = (f"It tells a complete intrusion story in three acts.\n\n1. {n} x 4625 (failed logon) for `{target}` with Logon Type 3 "
              f"(network) from {src}. Status 0xC000006A means the account exists and the password was wrong; "
              f"0xC0000064 would mean the username does not exist. Repeated 0xC000006A against one real account "
              f"is password guessing, not a user mistyping.\n"
              f"2. 4624 with Logon Type 10 (RemoteInteractive, meaning RDP) for the same account from the same IP: "
              f"a guess succeeded and the attacker has an interactive desktop session.\n")
    if persist == "4720":
        answer += ("3. 4720 then 4732: a new account `support$` was created and added to the local Administrators group. "
                   "That is persistence, a second way in that survives a password change on the first account.")
    elif persist == "7045":
        answer += ("3. 7045 (System log): a new service `WinUpdateSvc` pointing at `C:\\Users\\Public\\svchost.exe`. The real "
                   "svchost lives in System32; a copy in a user-writable folder running as a service is malware "
                   "persistence that will run at every boot as SYSTEM.")
    else:
        answer += ("3. 1102: the Security log was cleared by that account. Attackers do this to hide what came next; "
                   "the clearing event itself is the alarm, and anything after this point is missing from this host's log.")
    answer += (f"\n\nFirst actions, in order: isolate {host} from the network, disable `{target}` and reset its password "
               "from a clean machine, remove the persistence (delete the account or service, and rely on your SIEM "
               "copy of the logs if they were cleared), then search every other host for 4624 Type 10 or Type 3 "
               f"from {src}, because an attacker with one set of credentials rarely stops at one machine. Block "
               f"{src} at the perimeter and put RDP behind a VPN or gateway with MFA.")
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


CVSS_VECTORS = [
    ("AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8, "Critical",
     "reachable over the network, trivial to exploit, needs no account and no user action, and gives full control of "
     "confidentiality, integrity and availability. This is the worst class of bug: an unauthenticated remote takeover"),
    ("AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 8.8, "High",
     "remote and easy, but PR:L means the attacker needs a low-privileged account first. Anyone who can log in, "
     "including a phished user, gets full control"),
    ("AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1, "Medium",
     "the profile of a reflected cross-site scripting bug: remote and unauthenticated, but a victim has to click "
     "something (UI:R), the impact is limited (C:L/I:L) and S:C means it reaches beyond the vulnerable component into "
     "the user's browser session"),
    ("AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 7.8, "High",
     "a local privilege escalation: the attacker must already run code on the machine (AV:L) as a normal user (PR:L), "
     "and then becomes root or SYSTEM. On its own it is not an entry point; chained after any foothold it is"),
    ("AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:N/A:N", 5.9, "Medium",
     "remote and unauthenticated but AC:H means it depends on conditions the attacker does not control, such as "
     "winning a race or a specific configuration, and the impact is information disclosure only"),
    ("AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", 7.5, "High",
     "a remote, unauthenticated denial of service: no data is read or changed, but anyone can take the service down"),
    ("AV:A/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 8.8, "High",
     "full compromise, but AV:A limits it to attackers on the same network segment (Wi-Fi, VLAN, Bluetooth range), "
     "so internet exposure is not the concern; an insider or a compromised neighbouring device is"),
]
PRODUCTS = ["the FooGate VPN appliance firmware", "LibreCache 3.x", "the Orbit CMS media uploader",
            "BoltDB Server 12", "the Keystone directory agent", "QuickPrint spooler service"]


def cvss_record(rng: random.Random) -> list[dict]:
    vector, score, severity, meaning = rng.choice(CVSS_VECTORS)
    product = rng.choice(PRODUCTS)
    cve = f"CVE-2026-{rng.randint(10000, 49999)}"
    record = json.dumps({"id": cve, "cvss_v3_1": {"vector": f"CVSS:3.1/{vector}", "base_score": score,
                                                   "severity": severity.upper()},
                         "affected": product, "fixed_in": f"{rng.randint(2, 9)}.{rng.randint(0, 9)}.{rng.randint(1, 20)}",
                         "summary": f"A flaw in {product} allows an attacker to affect the system as described by the vector."},
                        indent=2)
    user = ask(rng, [
        f"Our scanner flagged this. How urgent is it really?\n\n```json\n{record}\n```",
        f"explain this cvss vector to me like I'm new to this\n\n```json\n{record}\n```",
        f"We run {product}. Do we patch tonight or at the next window?\n\n```json\n{record}\n```",
    ])
    av = {"N": "Network: exploitable remotely", "A": "Adjacent: same local network only", "L": "Local: needs code running on the host"}
    ac = {"L": "Low: works reliably", "H": "High: needs special conditions"}
    pr = {"N": "None: no account needed", "L": "Low: a normal user account", "H": "High: an admin account"}
    ui = {"N": "None: no victim action", "R": "Required: a user must click or open something"}
    s = {"U": "Unchanged: impact stays inside the vulnerable component", "C": "Changed: impact spreads to other components"}
    cia = {"N": "none", "L": "low", "H": "high"}
    parts = dict(p.split(":") for p in vector.split("/"))
    answer = (f"Base score {score}, {severity}. Reading the vector left to right:\n\n"
              f"- AV:{parts['AV']}, {av[parts['AV']]}\n- AC:{parts['AC']}, {ac[parts['AC']]}\n- PR:{parts['PR']}, {pr[parts['PR']]}\n"
              f"- UI:{parts['UI']}, {ui[parts['UI']]}\n- S:{parts['S']}, {s[parts['S']]}\n"
              f"- C/I/A: confidentiality {cia[parts['C']]}, integrity {cia[parts['I']]}, availability {cia[parts['A']]}\n\n"
              f"So this is {meaning}.")
    if score >= 9:
        answer += (f"\n\nPatch tonight, and if {product} is internet-facing, take it off the internet until it is "
                   "patched. Assume it may already have been hit and check its logs from the disclosure date onward.")
    elif score >= 7:
        answer += (f"\n\nPatch within days, sooner if {product} is exposed to untrusted users or networks. "
                   "Until then, restrict who can reach it.")
    else:
        answer += ("\n\nThe next regular maintenance window is fine unless the vendor or CISA KEV reports active "
                   "exploitation, which overrides the score. Scores describe potential, not current attacks.")
    answer += " The base score is the vendor's generic view; your real urgency depends on exposure and what the system holds."
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


SIGMA_RULES = [
    ("Credential dumping tool execution", "process_creation", "windows",
     ["Image|endswith: '\\mimikatz.exe'", "CommandLine|contains: 'sekurlsa::logonpasswords'"],
     "someone running the Mimikatz credential dumper, or any process using its best-known command, on a Windows host",
     "Windows process creation events: Sysmon Event ID 1, or Security Event ID 4688 with command-line auditing "
     "enabled (without it 4688 has no command line and the second selection cannot fire)",
     "almost none in production; red-team exercises and security researchers' workstations are the only legitimate "
     "sources, so tune by excluding those hosts rather than loosening the rule",
     "high"),
    ("Suspicious scheduled task from a temp folder", "process_creation", "windows",
     ["Image|endswith: '\\schtasks.exe'", "CommandLine|contains|all: ['/create', '\\AppData\\Local\\Temp\\']"],
     "a scheduled task being created whose action lives in a user's Temp folder, a common persistence trick because "
     "Temp is writable without admin rights",
     "process creation events (Sysmon 1 or Security 4688 with command line), or Security 4698 which logs task creation directly",
     "some installers and updaters create tasks pointing at Temp during setup; expect a few hits per week on user "
     "workstations and allowlist the specific signed installers",
     "medium"),
    ("Clearing of Windows event logs", "security", "windows",
     ["EventID: 1102"],
     "the Security event log being cleared, which attackers do to hide their tracks",
     "the Windows Security log itself; the matching System log event is 104",
     "legitimate admins rarely clear logs; anything outside a documented change window deserves a call",
     "high"),
    ("Linux shell history being removed", "process_creation", "linux",
     ["CommandLine|contains: ['rm ~/.bash_history', 'unset HISTFILE', 'history -c']"],
     "a user or script deleting or disabling the bash command history, an anti-forensics step",
     "Linux process execution logs from auditd (execve records) or Sysmon for Linux",
     "developers sometimes clear history to remove pasted secrets; frequency is low, so review each one",
     "medium"),
]


def sigma_rule(rng: random.Random) -> list[dict]:
    title, category, product, selection, detects, logsource, fps, level = rng.choice(SIGMA_RULES)
    sel = "\n".join(f"        {s}" for s in selection)
    rule = (f"title: {title}\nstatus: experimental\nlogsource:\n    category: {category}\n    product: {product}\n"
            f"detection:\n    selection:\n{sel}\n    condition: selection\nfalsepositives:\n    - Unknown\nlevel: {level}")
    user = ask(rng, [
        f"A colleague dropped this Sigma rule in our repo. What does it actually catch?\n\n```yaml\n{rule}\n```",
        f"should I enable this sigma rule, and what logs does it need\n\n```yaml\n{rule}\n```",
        f"Explain this detection rule and whether it will be noisy.\n\n```yaml\n{rule}\n```",
    ])
    answer = (f"It fires on {detects}.\n\nLog source: {logsource}. If you do not collect that source the rule is "
              f"silently useless, so check that first.\n\nFalse positives: {fps}. The rule's `falsepositives: Unknown` "
              f"is a placeholder; replace it with what you actually observe after a week in alert-only mode.\n\n"
              f"Level {level} is right for what it detects. Enable it, route it to a triage queue rather than a pager "
              "for the first week, and add the exclusions you find to a `filter` block in the rule rather than "
              "disabling the selection.")
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def directory_listing(rng: random.Random) -> list[dict]:
    issue = rng.choice(["suid", "world_writable_script", "passwd_perms", "key_perms"])
    rows = ["-rw-r--r--  1 root root    2451 Mar  3 09:12 /etc/hosts",
            "-rwxr-xr-x  1 root root   55480 Feb 14 11:02 /usr/bin/ls"]
    if issue == "suid":
        rows.append("-rwsr-xr-x  1 root root   38912 Jan 22 08:40 /usr/local/bin/diskcheck")
        q = "what does the s in the permissions mean and is it a problem?"
        why = ("`-rwsr-xr-x` on `/usr/local/bin/diskcheck`: the `s` in the owner's execute slot is the setuid bit "
               "(octal 4755). Whoever runs this binary runs it as its owner, root. Setuid root is normal for a handful "
               "of system binaries like `passwd` and `sudo` that were written to be safe, but a custom tool in "
               "/usr/local/bin with it is a privilege-escalation risk: any bug in it, or any way to make it run "
               "another program, hands out root. Find who installed it, read its source if you can, and if it only "
               "needs to read disks, give it a capability (`setcap cap_sys_rawio+ep`) or a sudoers rule instead and "
               "clear the bit with `chmod u-s`. Audit the rest of the system with `find / -perm -4000 -type f` and "
               "compare against your package manager's expected list.")
    elif issue == "world_writable_script":
        rows.append("-rwxrwxrwx  1 root root     812 Oct 30 22:15 /etc/cron.hourly/cleanup")
        q = "Is the cleanup script's mode a problem? It runs from cron."
        why = ("`-rwxrwxrwx` (octal 777) on `/etc/cron.hourly/cleanup` is the problem. The last `rwx` means any user "
               "on the box can edit it, and cron runs it as root every hour. That is a complete, trivial path to root: "
               "append a line, wait an hour. Fix it now with `chmod 755 /etc/cron.hourly/cleanup`, then `cat` it to see "
               "whether someone already added something, and check `find /etc /usr/local -perm -o+w -type f` for "
               "other files the same mistake was made on.")
    elif issue == "passwd_perms":
        rows.append("-rw-rw-rw-  1 root root    2847 Oct 29 17:03 /etc/passwd")
        rows.append("-rw-r-----  1 root shadow  1402 Oct 29 17:03 /etc/shadow")
        q = "something ran chmod on /etc, which of these are wrong?"
        why = ("`/etc/shadow` at `-rw-r-----` root:shadow (0640) is correct. `/etc/passwd` at `-rw-rw-rw-` (0666) is "
               "badly wrong: it must be 0644. World-writable /etc/passwd lets any user add an account with UID 0 or "
               "give an existing account an empty password field, either of which is instant root. Run "
               "`chmod 644 /etc/passwd`, then read the file and compare it with a backup or with `getent passwd` "
               "output from a known-good host, looking for new UID 0 entries or changed shells.")
    else:
        rows.append("-rw-r--r--  1 deploy deploy  2602 Sep  9 14:20 /home/deploy/.ssh/id_ed25519")
        rows.append("-rw-r--r--  1 deploy deploy   102 Sep  9 14:20 /home/deploy/.ssh/id_ed25519.pub")
        q = "ssh refuses to use this key, and separately is the mode safe?"
        why = ("The private key `/home/deploy/.ssh/id_ed25519` is `-rw-r--r--` (0644), readable by every user on the "
               "host. OpenSSH refuses to use a private key that others can read, which is the error you are seeing, "
               "and the refusal is right: anyone with a shell here could copy the key and log in wherever it is "
               "authorised. Fix with `chmod 600 /home/deploy/.ssh/id_ed25519` and `chmod 700 /home/deploy/.ssh`. "
               "Because the key has been exposed for as long as the mode was wrong, treat it as leaked: generate a new "
               "key, replace it in every authorized_keys that trusts it, and remove the old public key there. The "
               ".pub file at 0644 is fine, it is meant to be public.")
    listing = "\n".join(rows)
    user = f"```\n{listing}\n```\n\n{q}"
    return [{"role": "user", "content": user}, {"role": "assistant", "content": why}]


KINDS = {"ssh_bruteforce": ssh_bruteforce, "nmap_scan": nmap_scan, "listening_sockets": listening_sockets,
         "windows_events": windows_events, "cvss_record": cvss_record, "sigma_rule": sigma_rule,
         "directory_listing": directory_listing}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per-kind", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if OUT.exists():
        existing = {json.loads(line)["job"] for line in OUT.read_text().splitlines() if line.strip()}
    seen_users, written, rejected = set(), 0, 0
    with OUT.open("a") as f:
        for kind, build in KINDS.items():
            for i in range(a.per_kind):
                key = f"artifact|{kind}|{a.seed}|{i}"
                if key in existing:
                    continue
                turns = build(random.Random(f"{a.seed}:{kind}:{i}"))
                rec = {"turns": turns, "source": "gen_artifacts", "job": key, "teacher": "template"}
                errs = [e for e in problems({**rec, "source": "seed_check"}) if "source" not in e]
                first = " ".join(turns[0]["content"].lower().split())
                if errs or first in seen_users:
                    rejected += 1
                    continue
                seen_users.add(first)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} conversations to {OUT} ({rejected} rejected as invalid or duplicate)")


if __name__ == "__main__":
    main()
