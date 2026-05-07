import json, random, os
import math
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter, defaultdict
import argparse
import pickle
import logging
from datetime import datetime
import networkx as nx
import torch
from rank_bm25 import BM25Okapi

import utils as u
SEED = 66
u.set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="run-zs01/", help="Output directory")
parser.add_argument("--inputf", default="data/candidate-pool.jsonl", help="input file")
parser.add_argument("--feat", default="data/sparql_feats.jsonl", help="sparql feats file")
parser.add_argument("--temp", default=None, help="temp prior")
parser.add_argument("--trace", default=None, help="trace mode")
parser.add_argument("--abl", default=None, help="abl mode")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir) 
OUT_DIR.mkdir(parents=True, exist_ok=True)
# DATA_PATH = Path("data/candidate-pool.jsonl")
# FEATURE_PATH = Path("data/sparql_feats.jsonl")
# DATA_PATH = Path("data/candidate-pool.jsonl")
DATA_PATH = Path(args.inputf) 
# FEATURE_PATH = Path("data/sparql_feats.jsonl")
FEATURE_PATH = Path(args.feat) 
TOP_K = 50
YEAR_GAP = 10 

TEMP_MODEL = args.temp #gaussian, laplace, decay, beta, gamma, lognormal
TRACE_MODE = args.trace #none=all, other=semantic, 
ABLATION_MODE = args.abl

ABLATION_MODES = {
    "base",          
    "metadata",      #  metadata
    "citation",      #  citation 
    "all",            
}

TEMP_MODES = {
    "decay",          
    "beta",      # sim + metadata
    "gamma",      # sim + citation signals
    "gaussian",  
    "lognormal",
    "laplace",         
}

EMBEDDING_MODES = {
    "text",         
    "metadata",   
    "graph",     
    "hybrid",    
    "all"       
}

WEIGHTS = {
    "alpha": 0.40, # semantic
    "beta": 0.15, # msc hierarchy
    "gamma": 0.15, # keywords
    "eta": 0.15, # cite
    "zeta": 0.10, #bc
    "delta": 0.2, #time
    "tau_cross": 0.2
}

WEIGHTS.update({
    "tau_cited": 0.20,    
    "tau_structural": 0.20,  
    "tau_semantic": 0.20, 
    "tau_social": 0.05   
})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = OUT_DIR / f"run_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),  # keep console output
    ],)
config = {
    "DATA_PATH": str(DATA_PATH),
    "FEATURE_PATH": str(FEATURE_PATH),
    "TOP_K": TOP_K,
    "YEAR_GAP": YEAR_GAP,
    "TEMP_MODEL": TEMP_MODEL,
    # "ABLATION_MODES": "all",
    "TRACE_MODE": TRACE_MODE,
    "WEIGHTS": WEIGHTS,
    "OUT_DIR": str(OUT_DIR)}

logging.info("CONFIG:\n%s", json.dumps(config, indent=2))
logger = logging.getLogger(__name__)
logger.info("## START! ##")

# ================= LOAD DATA =================
entries = u.load_data(DATA_PATH, FEATURE_PATH)
logger.info(f"Loaded {len(entries)} queries.")

def schemapathrank(query, pool, kw_idf, top_k, mode="all", TEMP_MODEL="decay", TRACE_MODE=None):
    assert mode in ABLATION_MODES
    assert TEMP_MODEL in TEMP_MODES
    logger.info(f"Sanity: {TEMP_MODEL=} {ABLATION_MODE=} {TRACE_MODE=}")    
    q_emb = u.paper_embedding(query, mode="graph")
    query_refs = set(query.get("references", []))
    ranked = []
    q_trace_emb = u.trace_embedding_2(query, mode=TRACE_MODE)
    for c in pool:
        paper_id = c["paper"]
        c_emb = u.paper_embedding(c, mode="graph")
        sim = u.cosine(q_emb, c_emb)
        # metadata
        S_msc, overlap = u.msc_similarity(query.get("mscs", []), c.get("msc_codes", []))
        S_kw  = u.keyword_similarity(query.get("keywords", []),c.get("keywords", []), kw_idf)
        # citation
        cite_score = 1 if paper_id in query_refs else 0
        bc_score = len(query_refs & set(c.get("references", [])))
        bc_score /= max(1, len(query_refs))  # normalize
        # trace cross interaction
        c_trace_emb = u.trace_embedding_2(c, mode=TRACE_MODE)
        trace_len = len(c.get("traces", []))
        gate = np.log1p(trace_len) / (1 + np.log1p(trace_len))  # in (0,1)
        trace_cross = 0.5 * (
            u.cosine(q_emb, c_trace_emb) +
            u.cosine(q_trace_emb, c_emb) )
        score = sim
        if mode in {"metadata", "all"}:
            score += (
                WEIGHTS["beta"]  * S_msc +
                WEIGHTS["gamma"] * S_kw )
        if mode in {"citation", "all"}:
            score += (
                WEIGHTS["eta"]   * cite_score +
                WEIGHTS["zeta"]  * bc_score)
        # temporal weighting
        time_weight = u.temporal_hist_modeling(query["year"], c["year"], mode=TEMP_MODEL)
        if time_weight == 0.0:
            continue
        score *= (WEIGHTS["delta"] * time_weight + (1 - WEIGHTS["delta"]))
        score += WEIGHTS["tau_cross"] * gate * trace_cross
        ranked.append((paper_id, score))
    ranked.sort(key=lambda x: -x[1])
    return [
        {"paper": pid, "rank": i + 1, "score": float(score)}
        for i, (pid, score) in enumerate(ranked[:top_k])]

def save_run(path, data):
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def output_path(name):
    return OUT_DIR / f"run_{name}_{TRACE_MODE}.jsonl"

RANKERS_1 = {
    "spr_base": lambda q, p, k, kw, G: schemapathrank(q, p, kw, k, mode=ABLATION_MODE, TEMP_MODEL=TEMP_MODEL, TRACE_MODE=TRACE_MODE),
}

RANKERS_ALL_ABL = {
    "spr_base": lambda q, p, k, kw, G: schemapathrank(q, p, kw, k, mode="base", TEMP_MODEL=TEMP_MODEL),
    "spr_metadata": lambda q, p, k, kw, G: schemapathrank(q, p, kw, k, mode="metadata", TEMP_MODEL=TEMP_MODEL),
    "spr_citation": lambda q, p, k, kw, G: schemapathrank(q, p, kw, k, mode="citation", TEMP_MODEL=TEMP_MODEL),
    "spr_all": lambda q, p, k, kw, G: schemapathrank(q, p, kw, k, mode="all", TEMP_MODEL=TEMP_MODEL),
}

def make_ranker(mode, temp_model):
    return lambda q, p, k, kw, G: schemapathrank(
        q, p, kw, k, mode=mode, TEMP_MODEL=temp_model)

RANKERS_ALL = {
    f"spr_{mode}_{temp}": make_ranker(mode, temp)
    for temp in TEMP_MODES
    for mode in ABLATION_MODES
}

if TEMP_MODEL and ABLATION_MODE and TRACE_MODE:
    logger.info(f"Running: {TEMP_MODEL=} {ABLATION_MODE=} {TRACE_MODE=}")    
    RANKERS = RANKERS_1
elif TEMP_MODEL and not ABLATION_MODE:
    logger.info(f"Running: {TEMP_MODEL=}, all ablations")    
    RANKERS = RANKERS_ALL_ABL
else:
    logger.info(f"Running: all temporal priors, all ablations! ") 
    RANKERS = RANKERS_ALL

runs = {name: [] for name in RANKERS}
gold_results = []
for i, entry in enumerate(entries):
    query = entry["query_paper"]
    candidates = entry["candidates"]
    pool = candidates
    logger.info(f"# {i}: {query['paper']} {query['year']}")
    pool_sorted = sorted(pool, key=lambda c: c.get("llm_score", 0), reverse=True)
    gold_results.append({
        "query_paper": query,
        "ranked_candidates": pool_sorted[:TOP_K]})

    cited_size = sum(c.get("is_cited") == 1 for c in pool)
    logger.info(f"Pool size: {len(pool)} / Cited size: {cited_size} / TOP-K size: {TOP_K}")

    logger.info("Building kw-idf...")
    kw_idf = u.build_kw_idf(pool)
    G = None
    
    # ---- RANKERS ----
    for name, rank_fn in RANKERS.items():
        logger.info(f"== {name.upper()} Rank ==")
        ranked = rank_fn(query, pool, TOP_K, kw_idf, G)
        runs[name].append({
            "query_paper": query,
            "ranked_candidates": ranked,
        })

logger.info("Finished retrieval.")
logger.info("Saved runs:")
logger.info(" - %s", OUT_DIR)

save_run(output_path("gold"), gold_results)
for name, data in runs.items():
    save_run(output_path(name), data)
