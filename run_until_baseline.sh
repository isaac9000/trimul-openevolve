#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_BASE="trimul_kernel/openevolve_runs"

set -a
source .env
set +a

mkdir -p "$RUN_BASE"

RUN_NUM=$(ls -d "$RUN_BASE"/run* 2>/dev/null | wc -l)
RUN_NUM=$((RUN_NUM + 1))
RUN_OUT="$RUN_BASE/run$RUN_NUM"
mkdir -p "$RUN_OUT"

tmux kill-session -t openevolve-trimul 2>/dev/null || true
tmux new-session -d -s openevolve-trimul

tmux send-keys -t openevolve-trimul \
    "cd $SCRIPT_DIR && set -a && source .env && set +a && export OPENAI_API_KEY=\$ANTHROPIC_API_KEY && bash _baseline_loop.sh $RUN_OUT 2>&1 | tee ${RUN_OUT}.log" Enter

echo "tmux session : openevolve-trimul"
echo "Output dir   : $RUN_OUT"
echo "Log          : ${RUN_OUT}.log"
echo ""
echo "Monitor : tmux attach -t openevolve-trimul"
echo "Plot    : uv run python trimul_kernel/plot_run.py $RUN_OUT"
