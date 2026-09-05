#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="${HOME}/.cache/adrevo/datasets/eplb-v1"
mkdir -p "${DATA_DIR}"
wget -q -O "${DATA_DIR}/expert-load.json" \
    https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json
