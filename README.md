# Historically-Grounded Retrieval of Scholarly Research Beyond Citations

We introduce *historically-grounded* **precursor retrieval**, a novel scholarly information retrieval task aimed at identifying influential prior scientific work beyond explicit citations.

This repository contains the resources for:

- Reproducing the proposed __SchemaPathRank__, a _scholarly lineage-aware_ precursor retrieval framework over heterogeneous knowledge graphs.
- Running all experiments including all baselines and ablation studies in the paper
- Computing evaluation metrics reported in the paper
- (Optionally) generating custom query–candidate pools using the constructed knowledge graph

---
## Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Experiments](#running-the-experiments)
  - [Zero-Shot Variants](#zero-shot-variants)
  - [Learning-to-Rank (LTR) Variants](#learning-to-rank-ltr-variants)
  - [Baselines](#baselines)
- [Evaluation Metrics](#evaluation-metrics)
- [Knowledge Graph Dataset](#knowledge-graph-dataset)
- [Human Validation – LLM Judgment](#human-validation--llm-judgment)
- [Citation](#citation)
- [Contact](#contact)

## Repository Structure

```text
precursor-retrieval/
├── data/                  # Evaluation set (query and candidate pool), KG, and other related data
├── output/                # Shared experiment outputs, evaluation logs, significance test etc.
├── src/                   # Main source code
├── annotation/            # Human and LLM annotation results
├── run_all_zs.sh          # Zero-shot experiment runner
├── run_all_ltr.sh         # Learning-to-rank experiment runner
├── requirements.txt
└── README.md
```

## Installation
Clone the repository:

```
git clone https://github.com/susantiyuni/precursor-retrieval.git
cd precursor-retrieval
```

Install dependencies:

```
pip install -r requirements.txt
```
Some dependencies (e.g., PyTorch, ColBERT) may require additional system setup or GPU support. Please install them according to your supported hardware and environment.

## Running the Experiments
### Zero-Shot Variants
Run all zero-shot variants of SchemaPathRank:
```
./run_all_zs.sh 
```
This script executes the zero-shot experiments and ablations as in the paper. 
See [run_all_zs.sh](./run_all_zs.sh ) for configuration details.

### Learning-to-Rank (LTR) Variants
Run the LTR variants and ablation studies:
```
./run_all_ltr.sh 
```
This will run the training and ablation evaluations for the LTR versions of SchemaPathRank.
See [run_all_ltr.sh](./run_all_ltr.sh ) for configuration details.

### Baselines
Run all baseline methods reported in the paper:
```bash
python src/run_baselines.py \
    --out_dir runs_bl01 \
    --inputf data/candidate-pool.jsonl
```
For all the experiments above, the results (e.g., ranked candidate files per query, experiment logs) will be saved in the specified output directory ``out_dir``. For reference, baseline experiment results are available in [`output/run-baseline-01`](./output/run-baseline-01). 

## Evaluation Metrics
To compute evaluation metrics:
1. Update the output filenames (`METHOD_FILES`) in [`metrics.py`](./src/metrics.py)
2. Specify the desired `out_dir` (the output directory containing experiment results), then run:
```bash
python src/metrics.py --out_dir runs_bl01
```
This generates detailed metric scores (e.g., nDCG, Recall, MAP) and saves the evaluation reports in the output folder, with filenames starting with `eval_`. For reference, our evaluation outputs are available in [`output`](./output/). 

#### Significance Testing
We perform paired t-tests to assess the statistical significance of observed performance gains of SchemaPathRank over baselines:
```
python src/paired_test.py
```
For reference, we share our significance test results: [`significance_test`](./output/significance_test.log). 

## Knowledge Graph Dataset

A sample of the knowledge graph dataset is provided here: [sample-200.ttl](./data/sample-200.tll). **The full knowledge graph dataset will be made available on Zenodo upon acceptance of the paper.** To generate your own query–candidate pools, set up the KG in an RDF triple store (e.g., Apache Jena or Virtuoso) with a SPARQL endpoint. Please refer the documentation of your chosen triple store for setup instructions.

After setting up the RDF triple store, make sure that the SPARQL endpoint URL is correctly configured in [`config.py`](./src/config.py) according to your triple-store setup. Then run:
```
python src/query_papers.py
```
This script generates the query papers (example: [query-papers.json](./data/query-papers.jsonl)). Once the query papers have been generated, run:
```
python src/candidate_pool.py
```
This script generates the corresponding candidate paper pools (example: [candidate-pool.jsonl](./data/candidate-pool.jsonl)).

## Human Validation – LLM Judgment

We provide human annotation guidelines and results for assessing precursor relevance, along with LLM-based judgments, including the model outputs and prompts. See [annotation](./annotation/README.md)
 for details.

## Citation
```bibtex
@inproceedings{precursor2026,
  title={Historically-Grounded Retrieval of Scholarly Research Beyond Citations},
  author={X},
  year={2026}
}
```
## Contact
For questions, please open an issue or contact anon@anonymous.org
