# Ghost Code

A coding agent for your terminal. It reads, searches, edits and runs code in the
folder you start it in, asking before it changes anything.

It talks to any OpenAI-compatible endpoint, so the model can be a local one served
by Ollama, the GhostLM agent server, vLLM, or a hosted API.

```bash
pip install -e '.[code]'
ghost                         # interactive, in the current folder
ghost "why is test_auth failing?"
ghost -p "summarise this repo" # answer and exit
ghost -c                      # resume the last session in this folder
```

## Choosing a model

By default `ghost` connects to Ollama at `http://localhost:11434/v1` and picks an
installed model, preferring one with "coder" in its name.

| Setting | Flag | Environment | `~/.ghostcode/config.json` |
|---|---|---|---|
| Endpoint | `--base-url` | `GHOST_BASE_URL` | `base_url` |
| Model | `--model` | `GHOST_MODEL` | `model` |
| API key | | `GHOST_API_KEY` | `api_key` |

The model must support tool calling. To use a GhostLM checkpoint, serve it with
`python -m ghostlm.agent.server --checkpoint <checkpoint.pt> --port 8000` and pass
`--base-url http://localhost:8000/v1`. The small GhostLM checkpoints are not yet
strong enough to drive an agent loop; a tool-capable model of 7B parameters or more
is a realistic floor.

Ollama truncates long conversations to its context length without warning. For
agent work, start it with a larger one: `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`.

## Interface

| Key | Action |
|---|---|
| `enter` | Send |
| `\` then `enter`, or `esc` then `enter` | New line |
| `shift+tab` | Cycle permission mode |
| `esc` | Interrupt the model |
| `! command` | Run a shell command yourself; the output joins the conversation |
| `@ path` | Complete a file path |
| `ctrl+c` twice | Quit |

| Command | |
|---|---|
| `/help` | Commands and shortcuts |
| `/clear` | Start a fresh conversation |
| `/compact` | Replace the conversation with a summary to free context |
| `/model [name]` | Show or switch the model |
| `/mode [name]` | Show or set the permission mode |
| `/status` | Model, endpoint, mode, context size |
| `/init` | Have the agent write a `GHOST.md` for the project |
| `/exit` | Quit |

## Permissions

Reading and searching never ask. Everything else depends on the mode.

| Mode | Edits in the working folder | Edits elsewhere | Shell commands |
|---|---|---|---|
| default | Ask | Ask | Ask |
| accept edits | Allowed | Ask | Ask |
| plan | Refused | Refused | Refused |

Each prompt offers "yes", "yes for this session" and "no". Session approval for a
shell command covers its first two words, so approving `git status` also covers
`git status --short` but not `git push`. `--allow-all` skips every prompt and is
meant for throwaway sandboxes.

With `--print` there is nobody to ask, so anything that would prompt is refused
unless `--mode "accept edits"` or `--allow-all` permits it.

## Project instructions

If the working folder has a `GHOST.md` (or `AGENTS.md`, or `CLAUDE.md`), its
contents are added to the system prompt.

## Tools

`read_file`, `write_file`, `edit_file`, `bash`, `grep`, `glob`, `list_dir`. Search
skips `.git`, `node_modules`, virtual environments and build folders. Tool output
over 20,000 characters is truncated.

## Files

State lives in `~/.ghostcode/` (override with `GHOST_HOME`): `config.json`, prompt
`history`, and one saved session per working folder under `sessions/`.

## Layout

| File | |
|---|---|
| `ghostlm/code/cli.py` | Prompt, slash commands, `--print` mode |
| `ghostlm/code/agent.py` | The loop, permission modes, system prompt |
| `ghostlm/code/tools.py` | Tool definitions and handlers |
| `ghostlm/code/backend.py` | Streaming chat client |
| `ghostlm/code/ui.py` | Transcript, status line, permission prompt |
| `ghostlm/code/mascot.py` | The animated token ghost |
