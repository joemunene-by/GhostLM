# GhostLM Dataset Guide

> **Canonical corpus reference is now [CORPUS.md](../CORPUS.md)** at the repo root.
> This doc is a brief technical pointer kept for backward compatibility.

For current corpus composition, expansion targets, and licensing notes, see [CORPUS.md](../CORPUS.md).

For the training-data section of the model card (record counts, token counts, splits), see [MODEL_CARD.md](../MODEL_CARD.md#training-data).

For corpus diagnostics (length percentiles, dedup rate, year/category distributions, train/val leakage check), run `scripts/data_audit.py`.

---

## Adding a new data source

Add a `collect_<source>` function to `data/collect.py`:

```python
def collect_my_source(output_path="data/raw/my_source.jsonl", max_records=5000):
    records = []
    # ... collect data ...
    for item in data:
        cleaned = clean_text(item["text"])
        if len(cleaned) >= 50:
            records.append({"id": item["id"], "text": cleaned, "source": "my_source"})
    save_jsonl(records, output_path)
```

Then add the output path to `merge_datasets()` in `main()`. The merge step uses a deterministic MD5-bucket split, so no further config is needed — identical/near-duplicate texts will land in the same train/val bucket automatically.

After adding, run `python scripts/data_audit.py` and inspect `logs/data_audit.png` before kicking off a training run. License-check the source against the principles in [CORPUS.md](../CORPUS.md#licensing-principles).
