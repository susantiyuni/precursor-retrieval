#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Optional: activate virtual environment if needed
# source venv/bin/activate
# Make it executable
# chmod +x run_all_precursor.sh

# Run the Python script
python src/run-all-precursor.py --out_dir output-0902
