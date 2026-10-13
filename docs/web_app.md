# GhostLM web app

A chat app for GhostLM, served from the machine that holds the model and the
RAG index.

```bash
pip install -e '.[serve]'
python -m ghostlm.app --port 8091          # or: scripts/background/install_app.sh
```

Open `http://localhost:8091`, or `http://<mac-ip>:8091` from another device on
the same network or over a VPN such as Twingate. There is no login, so keep it
off the public internet.

## What it does

- Streams answers token by token. Generation runs on the CPU, so it can share a
  Mac with a training run on the GPU.
- Retrieves passages with `ghostlm.rag.HybridRetriever`: CVE, CWE, CAPEC and
  ATT&CK IDs in the question are looked up exactly, the rest by dense search.
  The index is memory-mapped and passages are read on demand, so a 500K-passage
  index costs little RAM.
- For exact-ID hits, shows a "From the sources" card that quotes the record's
  opening sentences with its citation. Model text cites passages as `[n]`;
  clicking a citation highlights the source.
- Explains pasted code by reading it. For a message that contains a program, retrieval is skipped
  (it would only match keywords against security references) and the answer is an outline taken
  from the code itself: language, functions and what each one does, constants, control flow,
  input and output, and a security check for risky calls and hardcoded credentials.
- Lets you delete chats: a cross on each one in the sidebar, and "Delete others" (or "Delete all"
  when no chat is open) at the top of the list. Both ask first.
- Shows the token ghost as its logo: in the sidebar, floating on the welcome screen, and beside each
  reply, where it bobs while an answer is streaming. Scrollbars are hidden; every panel still scrolls.
- Keeps chats on the server (`~/Library/Application Support/GhostLM/chats`), so
  every device sees the same history.
- Offers `ghost-small-gen` (downloaded from Hugging Face) and the current best
  checkpoint of the background ghost-base run, and shows that run's progress.

Small base models answer weakly on their own; until ghost-base finishes and is
fine-tuned for chat, the quoted facts and sources carry most of the value.
