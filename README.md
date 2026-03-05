## Historically-Grounded Retrieval of Scholarly Research Beyond Citations

We introduce _historically_-grounded **precursor retrieval** task, a scholarly information retrieval task aimed at identifying influential prior scientific work beyond explicit citation links. This repository contains code for running the precursor retrieval pipeline for all methods in the paper and (optionally) generating your own query–candidate pools using the zbMATH Open Knowledge Graph. 

-----

### Dependencies
First, install all required dependencies:

```
pip install -r requirements.txt
```
Some dependencies (e.g., PyTorch, ColBERT) may require additional system setup or GPU support. Install according to your supported system resources and environment.

### Running Precursor Retrieval

Update the candidate pool location in the [`src/run-all-precursor.py`](./src/run-all-precursor.py):

```
JSONL_PATH = Path("data/candidate-pool-latest.jsonl") 
```
Make sure the JSONL file exists at the specified path. Run the full precursor retrieval pipeline with:
```
./run_all_precursor.sh 
```
This will run both the proposed approach and all baselines as specified in the paper. The results will be saved in the specified directory. As reference, our output is shared in the [`output`](./output/) folder.

### zbMATH Open KG Setting and Installation

To generate your own query–candidate pools, you must set up the RDF triple store for the zbMATH Open KG yourself.
Please refer to the complete installation and setup guide here:

_(complete readme to be updated after anonymous period concludes)_

This includes:
- Downloading zbMATH Open KG data via OAI-PMH.
- Setting up the RDF triple store (with Virtuoso or Apache Fuseki)
- Configuring the SPARQL endpoint
