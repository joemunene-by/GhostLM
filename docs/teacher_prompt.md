You are writing training data for GhostLM, a small open-source cybersecurity language model. It will learn from your examples how to answer users from retrieved sources, so precision matters more than style.

First run `cd /Users/ghost/Desktop/GhostLM`. Always use these exact commands and paths; do not explore other files, do not read scripts, do not use `~`:
- Status: `/Users/ghost/Desktop/GhostLM/.venv/bin/python scripts/teacher_sft.py status` (prints the next batch number NNNN)
- Read: `data/teacher/queue/batch_NNNN.jsonl` (10 lines, one passage each: pid, source, ref, text)
- Write: `data/teacher/out/batch_NNNN.jsonl` (create the folder if needed)
- Check: `/Users/ghost/Desktop/GhostLM/.venv/bin/python scripts/teacher_sft.py validate NNNN`
Do not edit, create or delete any other file.

Repeat this loop for BATCHES batches, then stop and report how many you finished:
1. Run status to get the next batch number.
2. Read that queue batch.
3. Write the output file: one JSON object per line, nothing else (no prose, no code fences, no blank lines).
4. Run validate. If it prints errors, fix only the lines it names and validate again until it prints OK.

Each output line is:
{"pid": "<pid from the queue>", "type": "answer", "question": "...", "answer": "..."}

Per batch, write:
- For every passage: one "answer" example. For a rich passage you may add a second one with a different question.
- Exactly one "unanswerable" example: a realistic security question that none of the batch's passages answers. Use any pid from the batch and the answer exactly: My sources don't cover that.
- For a passage that is junk (navigation text, a cookie banner, code with no explanation): {"pid": "...", "type": "skip"} instead. At most 3 per batch.

Questions:
- Write what a real person would ask: a SOC analyst, student, developer or pentester. Be specific to the passage's content (names, IDs, products, techniques).
- Vary the form: what is, how does, why, how do I detect, how do I mitigate, what is the difference between, which versions.
- Never mention "the passage", "the text" or "the reference". The user cannot see it.

Answers:
- Use only facts stated in the passage. Do not add outside knowledge, even if you are sure it is true.
- The first sentence answers the question directly. Then 1-3 sentences of supporting detail. 40-900 characters.
- Reuse the passage's exact terms, names, IDs, versions and numbers.
- Put [1] right after the sentence that uses the passage's facts. Every answer cites [1] at least once.
- Plain sentences. No headings, no bullet lists, no "As an AI", no hedging filler.
- If the passage only partly answers the question, answer the part it supports and say what it does not cover.

Example (passage pid p123: "CAPEC-322: TCP (ISN) Greatest Common Divisor Probe. This OS fingerprinting probe sends a number of TCP SYN packets to an open port ... the smallest number that the target host uses when incrementing sequence numbers ..."):
{"pid": "p123", "type": "answer", "question": "How does the TCP ISN greatest common divisor probe fingerprint an operating system?", "answer": "It sends several TCP SYN packets to an open port and analyzes the Initial Sequence Number in each SYN/ACK reply to find the smallest value the host uses when incrementing sequence numbers [1]. Because operating systems and versions increment sequence numbers by different values, that result is compared against a database of OS behaviors to identify the OS type or version [1]."}
{"pid": "p123", "type": "unanswerable", "question": "What ports does the Mirai botnet scan for Telnet access?", "answer": "My sources don't cover that."}

Start now with BATCHES = 5.
