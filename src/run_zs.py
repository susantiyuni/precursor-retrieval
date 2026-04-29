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
parser.add_argument("--temp", default=None, help="Output directory")
# parser.add_argument("--funcmode", default='tm', help="Output directory")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir) 
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/candidate-pool.jsonl")
FEATURE_PATH = Path("data/sparql_feats.jsonl")
TOP_K = 50
YEAR_GAP = 10 
TEMP_MODEL = args.temp #gaussian, laplace, decay, beta, gamma, lognormal
TRACE_MODE = None #none=all, other=semantic, 

ABLATION_MODES = {
    "base",          
    "explicit",      # sim + metadata
    "citation",      # sim + citation signals
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
# TRACE_TYPES = ["cited", "structural", "semantic", "social"]

# assert TEMP_MODEL in TEMP_MODES

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

CITED_TRACES = {"direct", "reverse", "two_hop"}
STRONG_STRUCTURAL = {"co_citation", "bib_coupling"}
SEMANTIC_TRACES = {"msc_direct", "keyword_direct", "hybrid_msc", "hybrid_keyword"}
SOCIAL_TRACES = {"author", "coauthor", "author_topic_traj", "reviewer", "venue"}
TRACE_TYPES = ["cited", "structural", "semantic", "social"]

def trace_type_filter(trace, mode):
    if mode == "cited":
        return trace["type"] in CITED_TRACES
    elif mode == "structural":
        return trace["type"] in STRONG_STRUCTURAL
    elif mode == "semantic":
        return trace["type"] in SEMANTIC_TRACES
    elif mode == "social":
        return trace["type"] in SOCIAL_TRACES
    return True

def trace_score_fn(q_emb, q_trace_embs, c_emb, c):
    trace_score = 0.0
    for t in TRACE_TYPES:
        c_trace_emb_t = u.trace_embedding(c, mode=t)
        # cross interaction
        trace_cross_t = 0.5 * (
            u.cosine(q_emb, c_trace_emb_t) +
            u.cosine(q_trace_embs[t], c_emb))
        # filter traces by type
        traces_t = [
            tr for tr in c.get("traces", [])
            if trace_type_filter(tr, t)]
        # gating
        trace_len_t = min(len(traces_t), 50)
        gate_t = np.log1p(trace_len_t) / (1 + np.log1p(trace_len_t))
        # accumulate
        trace_score += WEIGHTS[f"tau_{t}"] * gate_t * trace_cross_t
    return trace_score

#v4 cross-interaction
def tmgnrxv4(query, pool, kw_idf, top_k, mode="all", TEMP_MODEL="decay"): #graph emb + temporal + ablation features
    assert mode in ABLATION_MODES
    assert TEMP_MODEL in TEMP_MODES
    q_emb = u.paper_embedding(query, mode="graph")
    query_refs = set(query.get("references", []))
    ranked = []
    q_trace_embs = {t: u.trace_embedding(query, mode=t) for t in TRACE_TYPES}
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
        # by trace type
        trace_score = trace_score_fn(q_emb, q_trace_embs, c_emb, c)
        score = sim
        if mode in {"explicit", "all"}:
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
        score += trace_score
        ranked.append((paper_id, score))
    ranked.sort(key=lambda x: -x[1])
    return [
        {"paper": pid, "rank": i + 1, "score": float(score)}
        for i, (pid, score) in enumerate(ranked[:top_k])]

# aggregated traces
def tmgnrxv3(query, pool, kw_idf, top_k, mode="all", TEMP_MODEL="decay", TRACE_MODE=TRACE_MODE):
    assert mode in ABLATION_MODES
    assert TEMP_MODEL in TEMP_MODES
    q_emb = u.paper_embedding(query, mode="graph")
    query_refs = set(query.get("references", []))
    ranked = []
    q_trace_emb = u.trace_embedding(query, mode=TRACE_MODE)
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
        c_trace_emb = u.trace_embedding(c, mode=TRACE_MODE)
        trace_len = len(c.get("traces", []))
        gate = np.log1p(trace_len) / (1 + np.log1p(trace_len))  # in (0,1)
        trace_cross = 0.5 * (
            u.cosine(q_emb, c_trace_emb) +
            u.cosine(q_trace_emb, c_emb) )
        score = sim
        if mode in {"explicit", "all"}:
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
    return OUT_DIR / f"run_{name}.jsonl"

RANKERS_1 = {
    "tmgnrx_base": lambda q, p, k, kw, G: tmgnrxv4(q, p, kw, k, mode="base", TEMP_MODEL=TEMP_MODEL),
    "tmgnrx_explicit": lambda q, p, k, kw, G: tmgnrxv4(q, p, kw, k, mode="explicit", TEMP_MODEL=TEMP_MODEL),
    "tmgnrx_citation": lambda q, p, k, kw, G: tmgnrxv4(q, p, kw, k, mode="citation", TEMP_MODEL=TEMP_MODEL),
    "tmgnrx_all": lambda q, p, k, kw, G: tmgnrxv4(q, p, kw, k, mode="all", TEMP_MODEL=TEMP_MODEL),
}

def make_ranker(mode, temp_model):
    # return lambda q, p, k, kw, G: tmgnrxv4(
    return lambda q, p, k, kw, G: tmgnrxv3(
        q, p, kw, k, mode=mode, TEMP_MODEL=temp_model)

RANKERS_all = {
    # f"tmgnrxv4_{mode}_{temp}": make_ranker(mode, temp)
    f"tmgnrxv3_{mode}_{temp}": make_ranker(mode, temp)
    for temp in TEMP_MODES
    for mode in ABLATION_MODES
}

if TEMP_MODEL:
    logger.info(f"Temporal priors: {TEMP_MODEL=}")    
    RANKERS = RANKERS_1
else:
    logger.info(f"Run on all temporal priors! ") 
    RANKERS = RANKERS_all

runs = {name: [] for name in RANKERS}
gold_results = []
for i, entry in enumerate(entries):
    query = entry["query_paper"]
    candidates = entry["candidates"]
    pool = candidates
    logger.info(f"# {i}: {query['paper']} {query['year']}")
    if len(pool) < TOP_K:
        continue
    pool_sorted = sorted(pool, key=lambda c: c.get("llm_score", 0), reverse=True)
    gold_results.append({
        "query_paper": query,
        "ranked_candidates": pool_sorted[:TOP_K]})

    cited_size = sum(c.get("is_cited") == 1 for c in pool)
    logger.info(f"Pool size: {len(pool)} / Cited size: {cited_size} / TOP-K size: {TOP_K}")

    logger.info("Building kw-idf...")
    kw_idf = u.build_kw_idf(pool)
    # ---- RANKERS ----
    for name, rank_fn in RANKERS.items():
        logger.info(f"== {name.upper()} Rank ==")
        ranked = rank_fn(query, pool, TOP_K, kw_idf, None)
        runs[name].append({
            "query_paper": query,
            "ranked_candidates": ranked,
        })

logger.info("Finished retrieval.")
logger.info("Saved runs:")
logger.info(" - %s", OUT_DIR)
# ---------------- SAVE ----------------

save_run(output_path("gold"), gold_results)
for name, data in runs.items():
    save_run(output_path(name), data)
