#!/usr/bin/env bash
# Full IRBench re-run on the reference AWS machine (c7i.4xlarge), in three phases.
#
#   scripts/vm_bench.sh setup     prepare + embed every scenario, build the load generator
#   scripts/vm_bench.sh sweeps    three engine lanes in parallel; each runs its engines' default
#                                 track (relevance only, not published for speed) then their sweep
#                                 variants (tune half, relevance only, aws-sweep profile)
#   scripts/vm_bench.sh lane E...  rerun the sweeps lane of the given engines
#   scripts/vm_bench.sh finals    tuned track, every stage including speed, one run at a time
#                                 on the pinned aws-c7i-4xlarge profile
#
# Logs go to .irbench/logs/; each phase writes .irbench/logs/<phase>.done when it ends.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

MODEL=bge-small-en-v1.5
SCENARIOS=(
  ecommerce/product-search
  knowledge-base/community-qa
  knowledge-base/covid-papers
  knowledge-base/covid-papers-abstracts
  knowledge-base/web-questions
)
LOGS=.irbench/logs
mkdir -p "$LOGS"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGS/progress.log"; }

run() { # engine scenario track hardware stages
  local name="${1}__${2//\//__}__${3}__${4}"
  log "start $name"
  if uv run irbench run --engine "$1" --scenario "$2" --model "$MODEL" --track "$3" \
      --hardware "$4" --stages "$5" >"$LOGS/$name.log" 2>&1; then
    log "ok    $name"
  else
    log "FAIL  $name (see $LOGS/$name.log)"
  fi
}

variants() { # engine scenario
  uv run python -c "import sys, yaml; print(' '.join(yaml.safe_load(open(sys.argv[1]))['variants']))" \
    "engines/$1/sweeps/$2.yaml"
}

lane() { # engines...: the default track, then every sweep variant, one engine at a time
  # (an engine's container and port are shared by all its runs).
  for engine in "$@"; do
    for s in "${SCENARIOS[@]}"; do
      run "$engine" "$s" default aws-c7i-4xlarge up,index,render,relevance,score,down
    done
    for s in "${SCENARIOS[@]}"; do
      for v in $(variants "$engine" "$s"); do
        run "$engine" "$s" "tuned+$v" aws-sweep up,index,render,relevance,score,down
      done
    done
  done
}

case "${1:-}" in
  setup)
    uv sync --extra embed-local
    cargo build --release --manifest-path loadgen/Cargo.toml
    for s in "${SCENARIOS[@]}"; do
      uv run irbench prepare --scenario "$s" && uv run irbench embed --scenario "$s" --model "$MODEL"
    done
    touch "$LOGS/setup.done"
    ;;
  sweeps)
    lane postgres &
    lane typesense qdrant &
    lane meilisearch elasticsearch postgres-paradedb &
    wait
    touch "$LOGS/sweeps.done"
    ;;
  lane)  # rerun one lane: scripts/vm_bench.sh lane <engine>...
    shift
    lane "$@"
    touch "$LOGS/lane-${1}.done"
    ;;
  finals)
    for s in "${SCENARIOS[@]}"; do
      for engine in meilisearch elasticsearch typesense qdrant postgres-paradedb postgres; do
        if [ "$s" = knowledge-base/covid-papers-abstracts ]; then
          stages=prepare,embed,up,index,render,relevance,score,down  # no speed scenario
        else
          stages=prepare,embed,up,index,render,relevance,perf,score,down
        fi
        run "$engine" "$s" tuned aws-c7i-4xlarge "$stages"
      done
    done
    touch "$LOGS/finals.done"
    ;;
  *)
    sed -n 2,13p "$0"
    exit 2
    ;;
esac
