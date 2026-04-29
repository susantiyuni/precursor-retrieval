#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Optional: activate virtual environment if needed
# source venv/bin/activate
# Make it executable
# chmod +x run_all_zs.sh

python src-ltr/run_zs.py --out_dir runs_zs --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl
python src-ltr/run_zs_baseline.py --out_dir runs_zs_baseline --inputf data/candidate-pool.jsonl 
