#!/usr/bin/env bash
# Download the ETH/UCY pedestrian trajectory files (world coordinates, meters, 2.5 Hz)
# as redistributed by the Trajectron++ repository. The original datasets are released
# for research use by ETH Zurich (BIWI) and the University of Cyprus (UCY Crowds);
# review their terms before using them for anything else. Raw data is NOT committed.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DEST"
B=https://raw.githubusercontent.com/StanfordASL/Trajectron-plus-plus/master/experiments/pedestrians/raw
for f in eth/test/biwi_eth.txt hotel/test/biwi_hotel.txt zara1/test/crowds_zara01.txt \
         zara2/test/crowds_zara02.txt univ/test/students001.txt univ/test/students003.txt; do
  echo "fetching $f"
  curl -fsSL -o "$DEST/$(basename "$f")" "$B/$f"
done
echo "done -> $DEST"
