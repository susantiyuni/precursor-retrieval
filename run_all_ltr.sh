#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Optional: activate virtual environment if needed
# source venv/bin/activate
# Make it executable
# chmod +x run_all_ltr.sh

python src-ltr/run_ltr.py --out_dir runs_ltr_ablation --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl --ablation base

python src-ltr/run_ltr.py --out_dir runs_ltr_ablation --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl --ablation base+plus_metadata

python src-ltr/run_ltr.py --out_dir runs_ltr_ablation --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl --ablation base+plus_citation

python src-ltr/run_ltr.py --out_dir runs_ltr_ablation --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl --ablation base+plus_gating

python src-ltr/run_ltr.py --out_dir runs_ltr_ablation --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl --ablation base+plus_metadata+plus_citation

python src-ltr/run_ltr.py --out_dir runs_ltr_ablation --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl --ablation base+plus_metadata+plus_citation+plus_gating
