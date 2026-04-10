.PHONY: all install test data train-tiny train-small generate chat benchmark plot export clean help

help:
	@echo "GhostLM — Cybersecurity Language Model"
	@echo "Usage: make [target]"
	@echo ""
	@echo "  install       Install all dependencies"
	@echo "  test          Run all unit tests"
	@echo "  data          Download and prepare training data"
	@echo "  train-tiny    Train ghost-tiny (3M params, CPU-friendly)"
	@echo "  train-small   Train ghost-small (55M params, GPU recommended)"
	@echo "  generate      Generate text from trained checkpoint"
	@echo "  chat          Interactive chat with trained model"
	@echo "  benchmark     Compare GhostLM vs GPT-2 perplexity"
	@echo "  plot          Plot training loss curve"
	@echo "  clean         Remove cache files"
	@echo "  help          Show this help message"

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt
	pip install -e .

test:
	PYTHONPATH=. pytest tests/ -v

data:
	python data/collect.py

train-tiny:
	python scripts/train.py --preset ghost-tiny --max-steps 2000 --batch-size 2 --device cpu

train-small:
	python scripts/train.py --preset ghost-small --max-steps 100000 --batch-size 32

generate:
	python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "A SQL injection attack works by" --max-tokens 150

chat:
	python scripts/chat.py --checkpoint checkpoints/best_model.pt

benchmark:
	python scripts/benchmark.py --checkpoint checkpoints/best_model.pt

export:
	python scripts/export.py --checkpoint checkpoints/best_model.pt --format both

plot:
	python scripts/plot_training.py --output logs/training_curve.png

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
