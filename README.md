# Historically-Grounded Retrieval of Scholarly Research Beyond Citations

We introduce *historically-grounded* **precursor retrieval**, a novel scholarly information retrieval task aimed at identifying influential prior scientific work beyond explicit citations.

This repository contains the resources for:

- Reproducing the proposed __SchemaPathRank__, a _scholarly lineage-aware_ precursor retrieval framework over heterogeneous knowledge graphs (details in our paper).
- Running all baseline methods from the paper
- Computing evaluation metrics reported in the paper, including significance testing result
- (Optionally) generating custom query–candidate pools using the constructed zbMATH knowledge graph

---

## Repository Structure

```text
precursor-retrieval/
├── data/                  # Evaluation set (query and candidate pool) and other datasets
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
This script executes the zero-shot experiments described in the paper. 
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

## Significance Testing
We perform _paired t-tests_ to measure statistical significance between SchemaPathRank and baseline methods.
```
python src/paired_test.py
```
For reference, we share our significance test results: [`significance_test`](./output/significance_test.log). 

## Knowledge Graph Setting and Installation
**The constructed zbMATH Open RDF knowledge graph will be made available on Zenodo upon acceptance or after the anonymity period.**
To generate your own query–candidate pools, you must set up the RDF triple store for the zbMATH Open KG yourself.
Please refer to the complete installation and setup guide here:

_(to be updated after anonymous period concludes)_

This includes:
- Downloading zbMATH Open KG data via OAI-PMH.
- Setting up the RDF triple store (with Virtuoso or Apache Fuseki)
- Configuring the SPARQL endpoint

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
For questions or collaborations, please open an issue or contact anon@anonymous.org
