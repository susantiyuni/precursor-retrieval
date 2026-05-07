import json, random, os
import math
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import argparse
import logging
from datetime import datetime
import networkx as nx
import torch
from rank_bm25 import BM25Okapi
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
COLBERT_MODEL = "colbert-ir/colbertv2.0"
colbert_config = ColBERTConfig(doc_maxlen=300, nbits=2)
ckpt = Checkpoint(COLBERT_MODEL, colbert_config=colbert_config)
ckpt.model = ckpt.model.to("cpu")

import utils as u
SEED = 66
u.set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="run-bl01/", help="Output directory")
parser.add_argument("--inputf", default="data/candidate-pool.jsonl", help="input file")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir) 
OUT_DIR.mkdir(parents=True, exist_ok=True)
# DATA_PATH = Path("data/candidate-pool.jsonl")
DATA_PATH = Path(args.inputf) 

TOP_K = 50
YEAR_GAP = 10 

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
    "TOP_K": TOP_K,
    "YEAR_GAP": YEAR_GAP,
    "OUT_DIR": str(OUT_DIR)}

logging.info("CONFIG:\n%s", json.dumps(config, indent=2))
logger = logging.getLogger(__name__)
logger.info("## START! ##")

# ================= LOAD DATA =================
entries = []
with DATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        entries.append(json.loads(line))

logger.info(f"Loaded {len(entries)} queries.")

def citation_rank(query, pool, top_k):
    query_refs = set(query.get("references", []))
    scored = []
    for c in pool:
        paper_id = c["paper"]
        cite_score = 1 if paper_id in query_refs else 0
        bc_score = len(query_refs & set(c.get("references", [])))
        scored.append((paper_id, cite_score, float(bc_score), c.get("year", 9999)))
        # scored.append((paper_id, cite_score, float(bc_score)))
    scored.sort(key=lambda x: (-x[1], -x[2], x[3])) # Sort by citation -> bibliographic coupling -> older year
    # scored.sort(key=lambda x: (-x[1], -x[2]))
    topk_output = [
        {"paper": pid,"rank": i + 1,"cite_score": int(cite_score),"bc_score": bc_score}
        # for i, (pid, cite_score, bc_score) in enumerate(scored[:top_k])]
        for i, (pid, cite_score, bc_score, _) in enumerate(scored[:top_k])]
    return topk_output

# citation graph structure
def build_local_graph(query, pool):
    G = nx.Graph()
    def add_paper(p):
        pid = f"p:{p['paper']}"
        G.add_node(pid, type="paper")
        for ref in p.get("references", []):
            rid = f"p:{ref}"
            G.add_edge(pid, rid, weight=1.0)
    add_paper(query)
    for c in pool:
        add_paper(c)
    return G

def ppr_rank(query, pool, top_k, G):
    query_pid = f"p:{query['paper']}"
    if query_pid not in G:
        G.add_node(query_pid)
    personalization = {n: 0.0 for n in G.nodes()}
    personalization[query_pid] = 1.0
    pr = nx.pagerank(G, alpha=0.85, personalization=personalization)
    ranked = []
    for c in pool:
        pid = f"p:{c['paper']}"
        score = pr.get(pid, 0.0)
        ranked.append((c["paper"], score))
    ranked.sort(key=lambda x: -x[1])
    return [
        {"paper": pid, "rank": i+1, "score": float(score)}
        for i, (pid, score) in enumerate(ranked[:top_k]) ]

def tokenize(paper, mode="text"):
    parts = []
    if paper.get("title"):
        parts.append(paper["title"])
    if mode in ("text", "all") and paper.get("review"):
        parts.append(paper["review"])
    if mode in ("metadata", "all"):
        if paper.get("keywords"):
            parts.append(" ".join({norm_kw(k) for k in paper["keywords"]}))
        if paper.get("mscs"):
            parts.append(" ".join(msc_prefixes(paper["mscs"])))
        if paper.get("msc_codes"):
            parts.append(" ".join(msc_prefixes(paper["msc_codes"])))
    if mode not in ("text", "metadata", "all"):
        raise ValueError("mode must be one of: text, metadata, all")
    return " ".join(parts).split()

def bm25_rank(query, pool, top_k, mode="text"):
    tokenized_docs = [tokenize(c, mode=mode) for c in pool]
    paper_ids = [c["paper"] for c in pool]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenize(query, mode=mode))
    ranked = sorted(zip(paper_ids, scores), key=lambda x: -x[1])
    return [{"paper": pid, "rank": i + 1, "score": float(score)}
            for i, (pid, score) in enumerate(ranked[:top_k])]

def paper_text(p):
    return f"{p['title']} {p['review']}"

def colbert_rank(query, pool, top_k):
    docs = [paper_text(c) for c in pool]
    Q = ckpt.queryFromText([paper_text(query)])
    D = ckpt.docFromText(docs)
    scores = (Q @ D.permute(0, 2, 1)).max(dim=-1).values.sum(dim=-1)
    ranked = sorted(zip(pool, scores.tolist()),key=lambda x: x[1],reverse=True)
    return [
        {"paper": c["paper"], "score": float(score)}
        for c, score in ranked[:top_k] ]

def dual_encoder_rank(query, pool, top_k):
    q_emb = u.paper_embedding(query, mode="text")
    ranked = []
    for c in pool:
        paper_id = c["paper"]
        c_emb = u.paper_embedding(c, mode="text")
        # score = float(np.dot(q_emb, c_emb))
        score = u.cosine(q_emb, c_emb)
        ranked.append((paper_id, score))
    ranked.sort(key=lambda x: -x[1])
    return [
        {"paper": pid, "rank": i + 1, "score": float(score)}
        for i, (pid, score) in enumerate(ranked[:top_k]) ]

def save_run(path, data):
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def output_path(name):
    return OUT_DIR / f"run_{name}.jsonl"

RANKERS = {
    "citation": lambda q, p, k, _, G: citation_rank(q, p, k),
    "ppr": lambda q, p, k, _, G: ppr_rank(q, p, k, G), 
    "bm25": lambda q, p, k, _, G: bm25_rank(q, p, k, mode="text"),
    "dualenc": lambda q, p, k, _, G: dual_encoder_rank(q, p, k),
    "colbert": lambda q, p, k, _, G: colbert_rank(q, p, k),
}

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
    logger.info("Building local graph...")
    G = build_local_graph(query, pool)

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
# ---------------- SAVE ----------------

save_run(output_path("gold"), gold_results)
for name, data in runs.items():
    save_run(output_path(name), data)
