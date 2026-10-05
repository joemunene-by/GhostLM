#!/bin/zsh
# Pulls every corpus source, then builds the generalist mix: rebuild, near-dedup,
# decontaminate, pretokenize by domain. A step whose output already exists is
# skipped, so rerunning resumes.
cd "${0:A:h}/../.."
PY=.venv/bin/python
mkdir -p .bg/logs
LOG=.bg/logs/collect.log
STATUS=.bg/collect.status
exec >>"$LOG" 2>&1

step() {
  local out=$1; shift
  if [[ -s $out ]]; then
    echo "[$(date '+%F %T')] skip $out (exists)"
    return
  fi
  echo "[$(date '+%F %T')] START $* -> $out"
  echo "running: $*" > $STATUS
  if "$@"; then
    echo "[$(date '+%F %T')] DONE $out ($(du -h $out 2>/dev/null | cut -f1))"
  else
    echo "[$(date '+%F %T')] FAILED $*, continuing"
    echo "$*" >> .bg/collect.failed
  fi
}

step data/raw/cwe.jsonl                 $PY scripts/collect_cwe.py
step data/raw/mitre_full.jsonl          $PY scripts/collect_mitre_full.py
step data/raw/capec.jsonl               $PY -c "from data.collect import collect_capec; collect_capec()"
step data/raw/owasp_top10.jsonl         $PY scripts/collect_owasp_top10.py
step data/raw/owasp_asvs.jsonl          $PY scripts/collect_owasp_asvs.py
mkdir -p data/raw/.src
for repo in CheatSheetSeries wstg; do
  [[ -d data/raw/.src/$repo ]] || git clone -q --depth 1 https://github.com/OWASP/$repo.git data/raw/.src/$repo
done
step data/raw/owasp_cheatsheets.jsonl   $PY scripts/collect_owasp_cheatsheets.py --src data/raw/.src/CheatSheetSeries
step data/raw/owasp_wstg.jsonl          $PY scripts/collect_owasp_wstg.py --src data/raw/.src/wstg
step data/raw/cisa_kev.jsonl            $PY scripts/collect_cisa_kev.py
step data/raw/cisa_advisories.jsonl     $PY scripts/collect_cisa_advisories.py
step data/raw/rfcs.jsonl                $PY scripts/collect_rfcs.py
step data/raw/nist_sp800.jsonl          $PY scripts/collect_nist_sp800.py
step data/raw/wikipedia_cyber.jsonl     $PY scripts/collect_wikipedia_cyber.py --request-delay 3
step data/raw/security_blogs.jsonl      $PY scripts/collect_security_blogs.py
step data/raw/vendor_research.jsonl     $PY scripts/collect_vendor_research.py
step data/raw/exploitdb.jsonl           $PY scripts/collect_exploitdb.py
step data/raw/ctftime.jsonl             $PY scripts/collect_ctftime.py --config data/ctftime_events.json
step data/raw/cve_full.jsonl            $PY scripts/collect_nvd_full.py
step data/raw/primus_fineweb.jsonl      $PY scripts/collect_primus.py
step data/raw/arxiv_full.jsonl          $PY scripts/collect_arxiv_full.py
step data/raw/security_code.jsonl       $PY scripts/collect_security_code.py
step data/raw/instruction.jsonl         $PY scripts/collect_instruction.py
step data/raw/wikipedia_general.jsonl   $PY scripts/collect_wikipedia_general.py --max-records 50000 --sample-every 1
step data/raw/math_reasoning.jsonl      $PY scripts/collect_math_reasoning.py --repo HuggingFaceTB/finemath --config finemath-4plus --max-records 60000
step data/raw/fineweb_edu.jsonl         $PY scripts/collect_fineweb_edu.py --max-records 150000
step data/raw/code_corpus.jsonl         $PY scripts/collect_code_corpus.py

echo "running: rebuild + pretokenize" > $STATUS
echo "[$(date '+%F %T')] rebuild generalist corpus"
$PY scripts/rebuild_corpus.py --profile generalist --max-cve-tokens 6000000 || { echo "rebuild FAILED"; echo "rebuild failed" > $STATUS; exit 1; }
echo "[$(date '+%F %T')] near-duplicate removal"
$PY scripts/dedup_near.py --train data/processed/train.jsonl --val data/processed/val.jsonl --out data/processed/train.dedup.jsonl \
  && mv data/processed/train.dedup.jsonl data/processed/train.jsonl \
  || echo "near-dedup FAILED, keeping exact-deduped train split"
step data/raw/general_mcq_bench.jsonl  $PY scripts/fetch_general_mcq.py
step data/raw/secqa.jsonl               $PY scripts/fetch_secqa.py
step data/raw/math_mcq_bench.jsonl      $PY scripts/build_math_eval.py --n 120 --out data/raw/math_mcq_bench.jsonl
$PY scripts/decontaminate.py --write-clean data/processed/train.clean.jsonl --report .bg/decontamination_report.md \
  && mv data/processed/train.clean.jsonl data/processed/train.jsonl \
  || echo "decontaminate FAILED, keeping uncleaned train split"
$PY scripts/pretokenize.py --by-domain --train data/processed/train.jsonl --val data/processed/val.jsonl || { echo "pretokenize FAILED"; echo "pretokenize failed" > $STATUS; exit 1; }
echo "done $(date '+%F %T')" > $STATUS
echo "[$(date '+%F %T')] ALL DONE"
