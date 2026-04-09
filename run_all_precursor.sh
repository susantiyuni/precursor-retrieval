#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Optional: activate virtual environment if needed
# source venv/bin/activate
# Make it executable
# chmod +x run_all_precursor.sh

# Run the Python script for different temporal prior functions
python src/run-all-precursor.py --out_dir output-all --temp gamma

# python src/run-all-precursor.py --out_dir output-all --temp beta
# python src/run-all-precursor.py --out_dir output-decay --temp decay
# python src/run-all-precursor.py --out_dir output-gaussian --temp gaussian
# python src/run-all-precursor.py --out_dir output-laplace --temp laplace
# python src/run-all-precursor.py --out_dir output-lognormal --temp lognormal
