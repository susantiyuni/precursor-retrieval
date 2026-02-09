## Historically-Grounded Retrieval of Scholarly Research Beyond Citations

We introduce _historically_-grounded **precursor retrieval** task, a scholarly information retrieval task aimed at identifying influential prior scientific work beyond explicit citation links. This repository contains code for running a precursor retrieval pipeline and (optionally) generating your own query–candidate pools using the zbMATH Open Knowledge Graph. 

P.S. The complete readme will be updated after anonymous period concludes.

-----

### Dependencies
First, install all required dependencies:

```
pip install -r requirements.txt
```
Some dependencies (e.g., PyTorch, ColBERT) may require additional system setup or GPU support. Install according to your supported system resources and environment.

### Running Precursor Retrieval

Update the candidate pool location in the code:

```
JSONL_PATH = Path("data/candidate-pool-latest.jsonl") 
```
Make sure the JSONL file exists at the specified path. Run the full precursor retrieval pipeline with:
```
python src/run-all-precursor.py --out_dir output-0902 
```
This will run both the proposed approach and all baselines as specified in the paper. The results will be saved in the specified output directory.

### zbMATH Open KG Setting and Installation

To generate your own query–candidate pools, you must set up the RDF triple store for the zbMATH Open KG yourself.
Please refer to the complete installation and setup guide here:

_(complete readme to be updated after anonymous period concludes)_

This includes:
- Downloading zbMATH Open KG data
- Setting up the RDF triple store
- Configuring the SPARQL endpoint
