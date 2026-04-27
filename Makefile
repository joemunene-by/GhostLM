PYTHON ?= python3

.PHONY: all install test data data-nvd-full data-ctf-repos data-ctftime data-mitre data-capec data-diversity data-rebuild data-audit train-tiny train-small generate chat benchmark eval-security eval-perplexity-by-source plot export clean help

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
	@echo "  data-diversity  Run all the corpus-diversity collectors (mitre + capec)"
	@echo "  data-rebuild    Re-merge data/raw/ into train/val (after a corpus pull)"
	@echo "  data-audit      Run pre-training corpus diagnostics + chart"
	@echo "  train-tiny      Train ghost-tiny (14.7M params, CPU-friendly)"
	@echo "  train-small     Train ghost-small (55M params, GPU recommended)"
	@echo "  generate        Generate text from trained checkpoint"
	@echo "  chat            Interactive chat with trained model"
	@echo "  benchmark       Compare GhostLM vs GPT-2 perplexity"
	@echo "  eval-security   Run the 5-task PMI security classification suite"
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

data-diversity: data-mitre data-capec

data-rebuild:
	$(PYTHON) scripts/rebuild_corpus.py

data-audit:
	$(PYTHON) scripts/data_audit.py --plot

train-tiny:
	$(PYTHON) scripts/train.py --preset ghost-tiny --max-steps 2000 --batch-size 2 --device cpu

train-small:
	$(PYTHON) scripts/train.py --preset ghost-small --max-steps 100000 --batch-size 32

generate:
	$(PYTHON) scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "A SQL injection attack works by" --max-tokens 150

chat:
	$(PYTHON) scripts/chat.py --checkpoint checkpoints/best_model.pt

benchmark:
	$(PYTHON) scripts/benchmark.py --checkpoint checkpoints/best_model.pt

eval-security:
	$(PYTHON) scripts/eval_security.py --checkpoint checkpoints/phase3.5_balanced/best_model.pt --output logs/eval_security_phase3.5_expanded.json

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
