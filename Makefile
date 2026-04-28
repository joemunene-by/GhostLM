PYTHON ?= python3

.PHONY: all install test data data-nvd-full data-ctf-repos data-ctftime data-mitre data-capec data-exploitdb data-exploitdb-audit data-arxiv-full data-diversity data-rebuild data-audit train-tiny train-small generate chat demo demo-compare benchmark eval-security eval-security-phase1 eval-security-phase2 eval-security-phase3 eval-security-all-phases eval-compare-phases eval-perplexity-by-source plot export clean help

help:
	@echo "GhostLM — Cybersecurity Language Model"
	@echo "Usage: make [target]"
	@echo ""
	@echo "  install         Install all dependencies"
	@echo "  test            Run all unit tests"
	@echo "  data            Download and prepare training data (full pipeline)"
	@echo "  data-nvd-full   Pull the full NVD CVE corpus (Phase 3 — uses NVD_API_KEY)"
	@echo "  data-ctf-repos  Pull CTF writeups from a JSON-config'd list of permissive repos"
	@echo "  data-ctftime    Pull CTFtime inline writeups for a JSON-config'd list of events"
	@echo "  data-mitre      Pull MITRE ATT&CK techniques (Apache 2.0)"
	@echo "  data-capec      Pull CAPEC attack patterns (public)"
	@echo "  data-exploitdb  Pull Exploit-DB records (persistent mirror, resume-aware, GPL-2.0)"
	@echo "  data-exploitdb-audit  Print structural audit of data/raw/exploitdb.jsonl"
	@echo "  data-arxiv-full Pull arXiv cs.CR full-text PDFs (needs pymupdf, ~1 req/sec)"
	@echo "  data-diversity  Run all the corpus-diversity collectors (mitre + capec)"
	@echo "  data-rebuild    Re-merge data/raw/ into train/val (after a corpus pull)"
	@echo "  data-audit      Run pre-training corpus diagnostics + chart"
	@echo "  train-tiny      Train ghost-tiny (14.7M params, CPU-friendly)"
	@echo "  train-small     Train ghost-small (55M params, GPU recommended)"
	@echo "  generate        Generate text from trained checkpoint"
	@echo "  chat            Interactive chat with trained model"
	@echo "  demo            Launch the Gradio web demo (single checkpoint)"
	@echo "  demo-compare    Launch Gradio with the Phase 3.5 vs 3.6 compare tab"
	@echo "  benchmark       Compare GhostLM vs GPT-2 perplexity"
	@echo "  eval-security   Run the 5-task PMI security classification suite (Phase 3.5 checkpoint)"
	@echo "  eval-security-all-phases  Re-score every preserved checkpoint on the new suite"
	@echo "  eval-compare-phases       Print a cross-phase comparison table from saved JSONs"
	@echo "  eval-perplexity-by-source  Per-source held-out perplexity breakdown"
	@echo "  plot            Plot training loss curve"
	@echo "  clean           Remove cache files"
	@echo "  help            Show this help message"

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt
	pip install -e .

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v

data:
	$(PYTHON) data/collect.py

data-nvd-full:
	$(PYTHON) scripts/collect_nvd_full.py

data-ctf-repos:
	$(PYTHON) scripts/collect_ctf_repos.py --config data/ctf_repos.json

data-ctftime:
	$(PYTHON) scripts/collect_ctftime.py --config data/ctftime_events.json

data-mitre:
	$(PYTHON) -c "from data.collect import collect_mitre_attack; collect_mitre_attack()"

data-capec:
	$(PYTHON) -c "from data.collect import collect_capec; collect_capec()"

data-exploitdb:
	$(PYTHON) scripts/collect_exploitdb.py

data-exploitdb-audit:
	$(PYTHON) scripts/audit_exploitdb.py

data-arxiv-full:
	$(PYTHON) scripts/collect_arxiv_full.py

data-diversity: data-mitre data-capec

data-rebuild:
	$(PYTHON) scripts/rebuild_corpus.py

data-audit:
	$(PYTHON) scripts/data_audit.py --plot

train-tiny:
	$(PYTHON) scripts/train.py --preset ghost-tiny --max-steps 2000 --batch-size 2 --device cpu

train-small:
	$(PYTHON) scripts/train.py --preset ghost-small --max-steps 100000 --batch-size 32

# Default checkpoint for the interactive scripts. Phase 3.5 is the
# canonical model — the Phase 2 default in older versions of this
# Makefile pointed at checkpoints/best_model.pt which is the
# pre-rebalance archive. Override with CHECKPOINT=... on the command
# line for a different one (e.g. CHECKPOINT=checkpoints/phase3.6_exploitdb/best_model.pt).
CHECKPOINT ?= checkpoints/phase3.5_balanced/best_model.pt

generate:
	PYTHONPATH=. $(PYTHON) scripts/generate.py --checkpoint $(CHECKPOINT) --prompt "A SQL injection attack works by" --max-tokens 150

chat:
	PYTHONPATH=. $(PYTHON) scripts/chat.py --checkpoint $(CHECKPOINT)

demo:
	PYTHONPATH=. $(PYTHON) demo/app.py --checkpoint $(CHECKPOINT)

# Compare tab visible — Phase 3.5 vs Phase 3.6 side-by-side.
demo-compare:
	PYTHONPATH=. $(PYTHON) demo/app.py \
		--checkpoint checkpoints/phase3.5_balanced/best_model.pt \
		--compare-checkpoint checkpoints/phase3.6_exploitdb/best_model.pt

benchmark:
	$(PYTHON) scripts/benchmark.py --checkpoint $(CHECKPOINT)

eval-security:
	$(PYTHON) scripts/eval_security.py --checkpoint checkpoints/phase3.5_balanced/best_model.pt --output logs/eval_security_phase3.5_expanded.json

eval-security-phase1:
	$(PYTHON) scripts/eval_security.py --checkpoint checkpoints/_backup-20260425-1310/_backup-20260425-1309/best_model.pt --output logs/eval_security_phase1_expanded.json

eval-security-phase2:
	$(PYTHON) scripts/eval_security.py --checkpoint checkpoints/best_model.pt --output logs/eval_security_phase2_expanded.json

eval-security-phase3:
	$(PYTHON) scripts/eval_security.py --checkpoint checkpoints/phase3_refresh/best_model.pt --output logs/eval_security_phase3_expanded.json

eval-security-phase3.6:
	$(PYTHON) scripts/eval_security.py --checkpoint checkpoints/phase3.6_exploitdb/best_model.pt --output logs/eval_security_phase3.6_expanded.json

eval-security-all-phases: eval-security-phase1 eval-security-phase2 eval-security-phase3 eval-security eval-security-phase3.6
	@$(PYTHON) scripts/compare_phase_evals.py

eval-compare-phases:
	@$(PYTHON) scripts/compare_phase_evals.py

eval-perplexity-by-source:
	$(PYTHON) scripts/eval_perplexity_by_source.py --checkpoint checkpoints/phase3.5_balanced/best_model.pt --output logs/eval_perplexity_by_source_phase3.5.json

export:
	$(PYTHON) scripts/export.py --checkpoint checkpoints/best_model.pt --format both

plot:
	$(PYTHON) scripts/plot_training.py --output logs/training_curve.png

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
