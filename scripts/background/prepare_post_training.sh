#!/bin/zsh
# Builds everything post-pretraining needs (RAG index, chat SFT set, RAFT set,
# tool-use set) on the CPU, so fine-tuning can start as soon as ghost-base ends.
# Idempotent: finished outputs are skipped.
cd "${0:A:h}/../.."
export PYTHONPATH=.
PY=.venv/bin/python
mkdir -p .bg/logs
exec >>.bg/logs/post_training_prep.log 2>&1

run() {
  local out=$1; shift
  if [[ -s $out ]]; then echo "[$(date '+%F %T')] skip $out"; return 0; fi
  echo "[$(date '+%F %T')] START $*"
  nice -n 15 "$@" && echo "[$(date '+%F %T')] DONE $out" || { echo "[$(date '+%F %T')] FAILED $*"; return 1; }
}

run data/rag/index.npy                   $PY scripts/build_rag_index.py --domains cybersec,knowledge --device cpu
run data/processed/chat_train.jsonl      $PY scripts/build_chat_dataset.py --small-talk-multiplier 30 --mcq-multiplier 2
run data/processed/raft_train.jsonl      $PY scripts/build_raft_data.py --device cpu
run data/processed/synth_tool_use.jsonl  $PY scripts/synth_tool_use.py
echo "[$(date '+%F %T')] post-training data ready"
