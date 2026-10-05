# Teacher-written chat data

`scripts/teacher_sft.py` has a stronger chat model write grounded question and
answer pairs from GhostLM's own corpus, in the "Reference passages / Question"
layout that RAFT and the web app use. The teacher only rephrases what each
passage states; the validator rejects answers that do not draw on it.

```bash
python scripts/teacher_sft.py queue --batches 300      # sample passages (needs data/rag)
python scripts/teacher_sft.py run --batches 300        # unattended, one `opencode run` call per batch
python scripts/teacher_sft.py status
python scripts/teacher_sft.py merge                    # -> data/processed/teacher_sft.jsonl
```

Sampling uses per-source quotas so small curated sources (CWE, CAPEC, OWASP,
MITRE, CISA KEV) are well represented next to web text.

`run` defaults to OpenCode's free `mimo-v2.6-flash-free` (measured: about 77 s
per 10-passage batch, 17 of 18 examples accepted on first try). Nemotron 3.5
Lightning stalled and degenerated on the same task. Rejected batches are moved
to `*.rejected.jsonl` and retried on the next run.

To drive an interactive coding agent instead, paste `docs/teacher_prompt.md`
into it.

Free model tiers log prompts and may use them to improve their models, so only
public corpus text should be sent. Check the provider's terms on using outputs
for training before publishing a model trained on this data.
