## Historically-Grounded Retrieval of Scholarly Research Beyond Citations

We introduce _historically_-grounded **precursor retrieval** task, a scholarly information retrieval task aimed at identifying influential prior scientific work beyond explicit citation links. This repository contains the [`source code`](./src) and [`dataset`](./data) for running the precursor retrieval pipeline for the proposed approach SchemaPathRank and all baselines in the paper and (optionally) generating your own query–candidate pools using the zbMATH Open Knowledge Graph. 

-----
### Installation
Clone the repository:

```
git clone https://github.com/susantiyuni/precursor-retrieval.git
cd precursor-retrieval
```

Install dependencies:

```
pip install -r requirements.txt
```
Some dependencies (e.g., PyTorch, ColBERT) may require additional system setup or GPU support. Install according to your supported system resources and environment.

### Running the Experiments
Make sure the data JSONL files exist at the specified path. Run the full precursor retrieval pipeline with:
#### Zero-Shot variants
```
./run_all_zs.sh 
```
#### LTR variants
```
./run_all_ltr.sh 
```
#### Baselines
```
./run_all_ltr.sh 
```
This will run both the proposed approach (SchemaPathRank, with ablations) and all baselines as specified in the paper. The results (e.g., ranked candidate files per query, experiment logs) will be saved in the specified directory. As reference, our output is shared in the [`output`](./output/) folder. 

#### Evaluate Results
To compute metric scores for all outputs, first update the output folder in [`run_metrics.sh`](run_metrics.sh), then execute:

```
./run_metrics.sh 
```

This will generate detailed evaluation metrics (e.g., nDCG, Recall, MAP) and save the reports in the output folder, with filenames starting with `eval_`. For reference, our evaluation outputs is shared in the [`output`](./output/) folder. 

#### Significance Test
We run paired t-test to, as follows:
```
python src/paired_test.py
```
For reference, we share our significance test results: [`significance_test`](./output/significance_test.log). 

### zbMATH Open KG Setting and Installation

To generate your own query–candidate pools, you must set up the RDF triple store for the zbMATH Open KG yourself.
Please refer to the complete installation and setup guide here:

_(complete readme to be updated after anonymous period concludes)_

This includes:
- Downloading zbMATH Open KG data via OAI-PMH.
- Setting up the RDF triple store (with Virtuoso or Apache Fuseki)
- Configuring the SPARQL endpoint


## Project Structure

```text
precursor-retrieval/
├── data/               # Datasets and preprocessing outputs
├── output/             # Evaluation outputs
├── src/                # Source code
├── LICENSE/          
├── run_all_ltr.sh/            # Script for SchemaPathRank-LTR variants
├── run_zs.sh/            # Script for SchemaPathRank-ZS variants
├── run_baseline.sh/            # Script for running all baselines
├── requirements.txt
└── README.md
```
