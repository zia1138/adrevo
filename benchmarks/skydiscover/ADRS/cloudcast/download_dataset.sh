#!/usr/bin/env bash
# Download dataset and config files for the Cloudcast benchmark.
#
# Required files:
#   ~/.cache/adrevo/datasets/cloudcast-v1/profiles/cost.csv
#                               Cloud egress cost per region pair ($/GB)
#   ~/.cache/adrevo/datasets/cloudcast-v1/profiles/throughput.csv
#                               Measured throughput per region pair (bps)
#   ~/.cache/adrevo/datasets/cloudcast-v1/examples/config/*.json
#                               Network configurations for evaluation
#
# Usage:
#   cd benchmarks/ADRS/cloudcast
#   bash download_dataset.sh

set -euo pipefail
BASE_URL="https://huggingface.co/datasets/f20180301/adrs-data/resolve/main/cloudcast"
DATA_DIR="${HOME}/.cache/adrevo/datasets/cloudcast-v1"

echo "Downloading Cloudcast benchmark data..."

# Download profiles
mkdir -p "${DATA_DIR}/profiles"
echo "  Downloading profiles/cost.csv..."
wget -q -O "${DATA_DIR}/profiles/cost.csv" "${BASE_URL}/profiles/cost.csv"
echo "  Downloading profiles/throughput.csv..."
wget -q -O "${DATA_DIR}/profiles/throughput.csv" "${BASE_URL}/profiles/throughput.csv"

# Download example configs
mkdir -p "${DATA_DIR}/examples/config"
for config in intra_aws.json intra_azure.json intra_gcp.json inter_agz.json inter_gaz2.json; do
    echo "  Downloading examples/config/${config}..."
    wget -q -O "${DATA_DIR}/examples/config/${config}" "${BASE_URL}/examples/config/${config}"
done

echo ""
echo "Done. Downloaded files:"
ls -lh "${DATA_DIR}/profiles/"*.csv
ls -lh "${DATA_DIR}/examples/config/"*.json
