#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Optional: activate virtual environment if needed
# source venv/bin/activate
# Make it executable
# chmod +x run_all_zs.sh

# fixed configurations, example:
python src/run_zs.py --out_dir runs_zs --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl --temp gamma --abl metadata --trace hybrid_topical_continuity
python src/run_zs.py --out_dir runs_zs --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl --temp decay --abl all --trace int_influence
python src/run_zs.py --out_dir runs_zs --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl --temp lognormal --abl citation --trace comm_consensus

# running all ablation + fix temporal mode
python src/run_zs.py --out_dir runs_zs --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl --temp gamma

# running all ablation + all temporal ablations
python src/run_zs.py --out_dir runs_zs --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl

# running on schema/trace ablations
python src/run_zs.py --out_dir runs_zs_abltrace --trace int_influence --temp gamma --abl base
python src/run_zs.py --out_dir runs_zs_abltrace --trace comm_consensus --temp gamma --abl base
python src/run_zs.py --out_dir runs_zs_abltrace --trace pure_topical_continuity --temp gamma --abl base
python src/run_zs.py --out_dir runs_zs_abltrace --trace hybrid_topical_continuity --temp gamma --abl base

# fixed temporal, on base model, using all traces
python src/run_zs.py --out_dir runs_zs --inputf data/candidate-pool.jsonl --feat data/sparql_feats.jsonl --temp decay --abl base 
