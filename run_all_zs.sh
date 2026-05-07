#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Optional: activate virtual environment if needed
# source venv/bin/activate
# Make it executable
# chmod +x run_all_zs.sh

# default configurations
python src/run_zs.py --out_dir runs_zs --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl

# running on schema ablations
python src/run_zs.py --out_dir runs_zs_abltrace --trace int_influence --temp gamma --abl base
python src/run_zs.py --out_dir runs_zs_abltrace --trace comm_consensus --temp gamma --abl base
python src/run_zs.py --out_dir runs_zs_abltrace --trace pure_topical_continuity --temp gamma --abl base
python src/run_zs.py --out_dir runs_zs_abltrace --trace hybrid_topical_continuity --temp gamma --abl base
