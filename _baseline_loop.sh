#!/usr/bin/env bash
# Internal script: poll baseline until in range, then launch OpenEvolve.
# Called by run_until_baseline.sh with RUN_OUT as $1.
set -uo pipefail

TARGET_LOW=10900.0
TARGET_HIGH=11300.0
MAX_ATTEMPTS=20
RUN_OUT="$1"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    echo ""
    echo "=== Attempt $attempt / $MAX_ATTEMPTS : checking baseline ==="

    BASELINE_JSON=$(mktemp /tmp/baseline_XXXXXX.json)

    if ! uv run python trimul_kernel/run_eval.py trimul_kernel/starting_point.py \
            -o "$BASELINE_JSON" --mode leaderboard; then
        echo "  Baseline eval failed — retrying in 20s..."
        rm -f "$BASELINE_JSON"
        sleep 20
        continue
    fi

    PARSE=$(uv run python -c "
import json, re
md = json.load(open('$BASELINE_JSON'))
gm = re.search(r'Geometric mean: ⏱ ([\d.]+)', md)
gpu = re.search(r'GPU: \`([^\`]+)\`', md)
print(gm.group(1) if gm else '0')
print(gpu.group(1) if gpu else 'unknown')
")
    rm -f "$BASELINE_JSON"

    GEOMEAN=$(echo "$PARSE" | sed -n '1p')
    GPU_NAME=$(echo "$PARSE" | sed -n '2p')

    echo "  GPU     : $GPU_NAME"
    echo "  Geomean : ${GEOMEAN} µs   (target: ${TARGET_LOW}–${TARGET_HIGH} µs)"

    IN_RANGE=$(uv run python -c "g=float('$GEOMEAN'); print('yes' if $TARGET_LOW <= g <= $TARGET_HIGH else 'no')")

    if [ "$IN_RANGE" = "yes" ]; then
        echo "  ✅ Baseline accepted — launching OpenEvolve"
        uv run python -m openevolve.cli \
          trimul_kernel/starting_point.py \
          trimul_kernel/openevolve_evaluator.py \
          --config trimul_kernel/openevolve_config.yaml \
          --iterations 25 \
          --output "$RUN_OUT"
        exit 0
    else
        if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
            echo "  ❌ Out of range — waiting 30s..."
            sleep 30
        else
            echo "  ❌ Gave up after $MAX_ATTEMPTS attempts. Last: ${GEOMEAN} µs on $GPU_NAME"
            exit 1
        fi
    fi
done
