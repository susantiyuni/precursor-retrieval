#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Optional: activate virtual environment if needed
# source venv/bin/activate
# Make it executable
# chmod +x run_metrics.sh

# Run the Python script
python src/metrics.py --out_dir output-0902
