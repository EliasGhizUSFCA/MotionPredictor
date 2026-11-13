#!/usr/bin/env bash
# Leave-one-scene-out training + evaluation on all five ETH/UCY scenes, then aggregate results.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
for s in eth hotel univ zara1 zara2; do
  $PY training/train.py --test-scene "$s" --out "models/socialmdn_$s" "$@"
done
$PY training/summarize.py
