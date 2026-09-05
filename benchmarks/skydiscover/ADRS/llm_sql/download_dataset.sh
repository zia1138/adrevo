#!/usr/bin/env bash
# Download CSV datasets for the LLM-SQL benchmark.
#
# Required files (placed in ~/.cache/adrevo/datasets/llm-sql-v1/):
#   movies.csv    - Rotten Tomatoes movie reviews (~9 MB)
#   beer.csv      - Beer review dataset (~2.5 MB)
#   BIRD.csv      - BIRD text-to-SQL dataset (~34 MB)
#   PDMX.csv      - PDMX metadata dataset (~7.4 MB)
#   products.csv  - Amazon product catalog (~16 MB)
#
# Usage:
#   cd benchmarks/ADRS/llm_sql
#   bash download_dataset.sh

set -euo pipefail
BASE_URL="https://huggingface.co/datasets/f20180301/adrs-data/resolve/main/llm_sql"
DATA_DIR="${HOME}/.cache/adrevo/datasets/llm-sql-v1"

echo "Downloading LLM-SQL benchmark datasets..."

mkdir -p "${DATA_DIR}"
for dataset in movies.csv beer.csv BIRD.csv PDMX.csv products.csv; do
    echo "  Downloading ${DATA_DIR}/${dataset}..."
    wget -q --show-progress -O "${DATA_DIR}/${dataset}" "${BASE_URL}/datasets/${dataset}"
done

echo ""
echo "Done. Downloaded files:"
ls -lh "${DATA_DIR}/"*.csv
