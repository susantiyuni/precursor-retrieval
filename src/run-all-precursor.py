import json, random, os
import math
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi
from collections import Counter, defaultdict
import argparse
import pickle
import logging
from datetime import datetime
import networkx as nx
import torch
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
COLBERT_MODEL = "colbert-ir/colbertv2.0"
colbert_config = ColBERTConfig(doc_maxlen=300, nbits=2)
ckpt = Checkpoint(COLBERT_MODEL, colbert_config=colbert_config)
ckpt.model = ckpt.model.to("cpu")

SEED = 66
np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="run-01/", help="Output directory")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir) 
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= CONFIG =================
JSONL_PATH = Path("data/candidate-pool-latest.jsonl")
TOP_K = 50
YEAR_GAP = 10 
TEMP_MODEL = "decay" #gaussian, laplace

ABLATION_MODES = {
    "base",          
    "explicit",      # sim + metadata
    "citation",      # sim + citation signals
    "all",            
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
    "delta": 0.2 #time
}

MSC_FILE = Path("data/msc_codes.jsonl")
msc_lookup = {}
with open(MSC_FILE, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        msc_lookup[entry["code"]] = entry

# msc_clean = "34B27"
# msc_info = msc_lookup.get(msc_clean)
# label = msc_info.get("short_title") if msc_info and msc_info.get("short_title") else msc_clean

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = OUT_DIR / f"run_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),  # keep console output
    ],
)
config = {
    "JSONL_PATH": str(JSONL_PATH),
    "MSC_FILE": str(MSC_FILE),
    "TOP_K": TOP_K,
    "YEAR_GAP": YEAR_GAP,
    "TEMP_MODEL": TEMP_MODEL,
    "WEIGHTS": WEIGHTS,
    "OUT_DIR": str(OUT_DIR)}

logging.info("CONFIG:\n%s", json.dumps(config, indent=2))

logger = logging.getLogger(__name__)
logger.info("## START! ##")
logger.info(f"Output dir: {OUT_DIR}")
logger.info (f"Processing {JSONL_PATH=} {OUT_DIR=}")

# ================= LOAD DATA =================
entries = []
with JSONL_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        entries.append(json.loads(line))

logger.info(f"Loaded {len(entries)} queries.")

# ================= EMBEDDING MODEL =================
# MODEL = SentenceTransformer("all-mpnet-base-v2")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_embed_cache = {}
def embed(text: str, MODEL: SentenceTransformer) -> np.ndarray:
    key = (text, id(MODEL))  # ensure cache is MODEL-specific
    if key not in _embed_cache:
        _embed_cache[key] = MODEL.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
    return _embed_cache[key]

# ================= HELPERS =================
def time_filtered_pool(query, candidates):
    cutoff = query["year"] - YEAR_GAP
    return [c for c in candidates if c.get("year") and c["year"] <= cutoff]

def build_kw_idf(all_candidates):
    kw_counter = Counter()
    N = len(all_candidates)

    for c in all_candidates:
        for k in c.get("keywords", []):
            kw_counter[norm_kw(k)] += 1

    return {k: math.log((N + 1) / (1 + v)) for k, v in kw_counter.items()}

def split_keywords(s):
    return s.split("_") if "_" in s else [s]

def norm_kw(kw_url: str) -> str:
    """URL -> keyword id (bar_induction)"""
    return kw_url.rstrip("/").split("/")[-1].lower()

def get_msc(msc):
    msc_info = msc_lookup.get(msc)
    label = msc_info.get("short_title") if msc_info and msc_info.get("short_title") else msc
    return label

def norm_msc(msc_url: str) -> str: 
    """URL -> MSC code (03B30)"""
    return msc_url.rstrip("/").split("/")[-1]

def msc_prefixes(mscs):
    """
    Input: list of MSC codes or URLs
    Output: hierarchical prefixes
    Example 03B30 -> {03, 03B, 03B30}
    """
    pref = set()
    for m in mscs:
        c = norm_msc(m)
        if len(c) >= 2:
            pref.add(c[:2])   # broad
        if len(c) >= 3:
            pref.add(c[:3])   # mid
        pref.add(c)          # exact
    return pref

def msc_similarity(msc_q, msc_c):
    """
    Hand-designed hierarchical overlap
    """
    q_msc = msc_prefixes(msc_q)
    c_msc = msc_prefixes(msc_c)
    overlap = q_msc & c_msc
    msc_score = (
        1.0 * len([x for x in overlap if len(x) == 5]) +  # exact code
        0.6 * len([x for x in overlap if len(x) == 3]) +  # mid
        0.3 * len([x for x in overlap if len(x) == 2])    # broad
    )
    return msc_score

def keyword_similarity(kw_q, kw_c, kw_idf):
    if not kw_q or not kw_c:
        return 0.0
    q_kw = {norm_kw(k) for k in kw_q}
    c_kw = {norm_kw(k) for k in kw_c}
    kw_score = sum(kw_idf.get(k, 0.0) for k in (q_kw & c_kw))
    kw_score = kw_score / (sum(kw_idf.values()) + 1e-9)
    return kw_score

# ---------- Citation baseline ----------
def citation_rank(query, pool, top_k):
    """
    Return top_k candidates ranked by:
      1. Whether cited in query (1/0)
      2. Bibliographic coupling score
      3. Year (older first)
    Only return paper id, rank, citation flag, and bc_score
    """
    query_refs = set(query.get("references", []))
    scored = []
    for c in pool:
        paper_id = c["paper"]
        cite_score = 1 if paper_id in query_refs else 0
        bc_score = len(query_refs & set(c.get("references", [])))
        # scored.append((paper_id, cite_score, float(bc_score), c.get("year", 9999)))
        scored.append((paper_id, cite_score, float(bc_score)))
    # scored.sort(key=lambda x: (-x[1], -x[2], x[3])) # Sort by citation -> bibliographic coupling -> older year
    scored.sort(key=lambda x: (-x[1], -x[2]))
    topk_output = [
        {"paper": pid,"rank": i + 1,"cite_score": int(cite_score),"bc_score": bc_score}
        for i, (pid, cite_score, bc_score) in enumerate(scored[:top_k])]
        # for i, (pid, cite_score, bc_score, _) in enumerate(scored[:top_k])]
    return topk_output

def assert_citation_sanity(query, pool):
    query_refs = set(query.get("references", []))
    zeros = sum(
        1 for c in pool
        if len(query_refs & set(c.get("references", []))) == 0 )
    assert zeros > 0, "LEAKAGE: citation pool has no zero-overlap candidates."

# ---------- BM25 baseline ----------
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

def bm25_rank(query, pool, top_k, mode="all"):
    tokenized_docs = [tokenize(c, mode=mode) for c in pool]
    paper_ids = [c["paper"] for c in pool]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenize(query, mode=mode))
    ranked = sorted(zip(paper_ids, scores), key=lambda x: -x[1])
    return [{"paper": pid, "rank": i + 1, "score": float(score)}
            for i, (pid, score) in enumerate(ranked[:top_k])]

def temporal_hist_modeling(year_q, year_c, mode=TEMP_MODEL, mu=12, sigma=6, tau=25):
    """
    Summary
    Function	Peak	Symmetric	Decay type	Use case
    Gaussian	Yes	Yes	Quadratic	Prefer a specific year
    Laplace	Yes	Yes	Linear	Prefer a specific year (wider)
    exp(-dt/τ)	No	No	Exponential	Freshness / recency
    """
    dt = year_q - year_c
    if dt <= 0:
        return 0.0
    if mode == "gaussian":
        return math.exp(-((dt - mu) ** 2) / (2 * sigma ** 2))
    elif mode == "laplace":
        return math.exp(-abs(dt - mu) / sigma)
    elif mode == "decay":
        return math.exp(-dt / tau)
    else:
        raise ValueError("mode must be one of: gaussian, laplace, decay")

def get_msc(msc):
    msc_info = msc_lookup.get(msc)
    label = msc_info.get("short_title") if msc_info and msc_info.get("short_title") else msc
    return label

def graph_to_lm_text(paper):
    parts = []
    if paper.get("title"):
        parts.append(f"title: {paper['title']}")
    # if paper.get("review"):
    #     parts.append(paper["review"])
    if paper.get("keywords"):
        kws = ", ".join(part for k in paper["keywords"] for part in split_keywords(norm_kw(k)))
        parts.append(f"keywords: {kws}")        
        # parts.append(" ".join({norm_kw(k) for k in paper["keywords"]}))
    if paper.get("mscs"):
        # parts.append(" ".join(msc_prefixes(paper["mscs"])))
        mscs = ", ".join(get_msc(msc) for msc in msc_prefixes(paper["mscs"]))
        parts.append(f"subject classification: {mscs}")
    if paper.get("msc_codes"):
        # parts.append(" ".join(msc_prefixes(paper["msc_codes"])))
        mscs = ", ".join(get_msc(msc) for msc in msc_prefixes(paper["msc_codes"]))
        parts.append(f"subject classification: {mscs}")
    return "\n".join(parts)

def paper_embedding(paper, mode="text"):
    if mode == "text":
        parts = []
        if paper.get("title"):
            parts.append(paper["title"])
        if paper.get("review"):
            parts.append(paper["review"])
        return embed(" ".join(parts), MODEL)
    elif mode == "metadata":
        parts = []
        if paper.get("title"):
            parts.append(paper["title"])
        if paper.get("keywords"):
            parts.append(" ".join({norm_kw(k) for k in paper["keywords"]}))
        if paper.get("mscs"):
            parts.append(" ".join(msc_prefixes(paper["mscs"])))
        if paper.get("msc_codes"):
            parts.append(" ".join(msc_prefixes(paper["msc_codes"])))
        return embed(" ".join(parts), MODEL)
    elif mode == "all":
        parts = []
        if paper.get("title"):
            parts.append(paper["title"])
        if paper.get("review"):
            parts.append(paper["review"])
        if paper.get("keywords"):
            parts.append(" ".join({norm_kw(k) for k in paper["keywords"]}))
        if paper.get("mscs"):
            parts.append(" ".join(msc_prefixes(paper["mscs"])))
        if paper.get("msc_codes"):
            parts.append(" ".join(msc_prefixes(paper["msc_codes"])))
        return embed(" ".join(parts), MODEL)
    elif mode == "graph":
        # assert G is not None
        # graph_text = graph_to_lm_text(G, paper)
        graph_text = graph_to_lm_text(paper)
        return embed(graph_text, MODEL)
    elif mode == "hybrid":
        alpha = 0.7
        # assert G is not None
        e_text = paper_embedding(paper, mode="text")
        e_graph = paper_embedding(paper, mode="graph")
        return alpha * e_text + (1 - alpha) * e_graph
    else:
        raise ValueError("mode must be: text, metadata, graph, hybrid")

def build_local_graph(query, pool):
    """
    Build per-query heterogeneous graph for PPR.
    Nodes:
        p:<paper_id>
        k:<keyword>
        m:<msc_prefix>
    Edges:
        paper-keyword
        paper-msc
        keyword co-occurrence
        msc hierarchy
    """
    G = nx.Graph()
    def add_paper(p):
        pid = f"p:{p['paper']}"
        G.add_node(pid, type="paper")
        # ---------- normalize ----------
        kws = [norm_kw(k) for k in p.get("keywords", [])]
        msc_raw = p.get("msc_codes") or p.get("mscs", [])
        mscs = [norm_msc(m) for m in msc_raw]
        # msc_pref = msc_prefixes(mscs)
        msc_pref = list(msc_prefixes(mscs))
        # ---------- paper -> keyword ----------
        for kw in kws:
            kid = f"k:{kw}"
            G.add_edge(pid, kid, weight=1.0)
        # ---------- paper -> msc hierarchy ----------
        for pref in msc_pref:
            mid = f"m:{pref}"
            G.add_edge(pid, mid, weight=1.0)
        # ---------- keyword co-occurrence ----------
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                G.add_edge(f"k:{kws[i]}",f"k:{kws[j]}",weight=0.2 )
        # ---------- MSC hierarchy chain ----------
        for i in range(len(msc_pref) - 1):
            G.add_edge(f"m:{msc_pref[i]}",f"m:{msc_pref[i+1]}", weight=0.3 )
    add_paper(query) # add query + candidates
    for c in pool:
        add_paper(c)
    return G

def ppr_rank(query, pool, top_k, G):
    seeds = {f"p:{query['paper']}": 1.0} # seed query paper strongly
    for kw in query.get("keywords", []): # seed metadata too
        seeds[f"k:{norm_kw(kw)}"] = 1.0
    for m in query.get("mscs", []):
        for p in msc_prefixes([m]):
            seeds[f"m:{p}"] = 1.0
    pr = nx.pagerank(G, alpha=0.85, personalization=seeds)
    ranked = []
    for c in pool:
        pid = f"p:{c['paper']}"
        score = pr.get(pid, 0.0)
        time_weight = temporal_hist_modeling(query["year"], c["year"], mode=TEMP_MODEL)
        if time_weight == 0.0:
            continue
        score = score * (WEIGHTS["delta"] * time_weight + (1 - WEIGHTS["delta"]))
        ranked.append((c["paper"], score))
    ranked.sort(key=lambda x: -x[1])
    return [
        {"paper": pid, "rank": i+1, "score": float(score)}
        for i, (pid, score) in enumerate(ranked[:top_k]) ]

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
    q_emb = paper_embedding(query, mode="text")
    ranked = []
    for c in pool:
        paper_id = c["paper"]
        c_emb = paper_embedding(c, mode="text")
        score = float(np.dot(q_emb, c_emb))
        ranked.append((paper_id, score))
    ranked.sort(key=lambda x: -x[1])
    return [
        {"paper": pid, "rank": i + 1, "score": float(score)}
        for i, (pid, score) in enumerate(ranked[:top_k]) ]

def tmgnrx_rank(query, pool, kw_idf, top_k, mode="all"): #graph emb + temporal + ablation features
    assert mode in ABLATION_MODES
    q_emb = paper_embedding(query, mode="graph")
    query_refs = set(query.get("references", []))
    # compute BM25 scores once
    tokenized_docs = [tokenize(c, mode="metadata") for c in pool]
    paper_ids = [c["paper"] for c in pool]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(tokenize(query, mode="metadata"))
    ranked = []
    for c in pool:
        paper_id = c["paper"]
        c_emb = paper_embedding(c, mode="graph")
        sim = float(np.dot(q_emb, c_emb))
        # 1. BM25 score
        S_bm25 = bm25_scores[i]
        # 2. explicit signals
        S_msc = msc_similarity(query.get("mscs", []), c.get("msc_codes", []))
        S_kw  = keyword_similarity(query.get("keywords", []),c.get("keywords", []), kw_idf)
        # 3. citation signals
        cite_score = 1 if paper_id in query_refs else 0
        bc_score = len(query_refs & set(c.get("references", [])))
        bc_score /= max(1, len(query_refs))  # normalize
        # ---- scoring ----
        score = sim
        if mode in {"explicit", "all"}:
            score += (
                WEIGHTS["beta"]  * S_msc +
                WEIGHTS["gamma"] * S_kw )
        if mode in {"citation", "all", "allbm25"}:
            score += (
                WEIGHTS["eta"]   * cite_score +
                WEIGHTS["zeta"]  * bc_score)
        if mode in {"bm25", "allbm25"}:
            score += (WEIGHTS["beta"]  * S_bm25)
        # temporal weighting (kept identical across ablations)
        time_weight = temporal_hist_modeling(query["year"], c["year"], mode=TEMP_MODEL)
        if time_weight == 0.0:
            continue
        score *= (WEIGHTS["delta"] * time_weight + (1 - WEIGHTS["delta"]))
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

def build_candidate_pool(query, candidates):
    uncited = sorted(
        (c for c in candidates if c.get("is_cited") == 0),
        key=lambda c: c.get("llm_score", 0),
        reverse=True)[:90]
    cited = sorted(
        (c for c in candidates if c.get("is_cited") == 1),
        key=lambda c: c.get("llm_score", 0),
        reverse=True)[:10]
    cited = time_filtered_pool(query, cited)
    random.shuffle(uncited)
    random.shuffle(cited)
    pool = uncited + cited
    random.shuffle(pool)
    return pool

RANKERS = {
    "citation": lambda q, p, k, _, G: citation_rank(q, p, k),
    "bm25": lambda q, p, k, _, G: bm25_rank(q, p, k, mode="text"),
    "dualenc": lambda q, p, k, _, G: dual_encoder_rank(q, p, k),
    "ppr": lambda q, p, k, _, G: ppr_rank(q, p, k, G),
    "colbert": lambda q, p, k, _, G: colbert_rank(q, p, k),
    "tmgnrx_base": lambda q, p, k, kw, G: tmgnrx_rank(q, p, kw, k, mode="base"),
    "tmgnrx_explicit": lambda q, p, k, kw, G: tmgnrx_rank(q, p, kw, k, mode="explicit"),
    "tmgnrx_citation": lambda q, p, k, kw, G: tmgnrx_rank(q, p, kw, k, mode="citation"),
    "tmgnrx_all": lambda q, p, k, kw, G: tmgnrx_rank(q, p, kw, k, mode="all")
}

runs = {name: [] for name in RANKERS}
gold_results = []
for i, entry in enumerate(entries):
    query = entry["query_paper"]
    candidates = entry.get("candidates", [])
    logger.info(f"# {i}: {query['paper']} {query['year']}")
    pool = build_candidate_pool(query, candidates)
    if len(pool) < TOP_K:
        continue
    # ---- GOLD ----
    pool_sorted = sorted(pool, key=lambda c: c.get("llm_score", 0), reverse=True)
    gold_results.append({
        "query_paper": query,
        "ranked_candidates": pool_sorted[:TOP_K]})

    cited_size = sum(c.get("is_cited") == 1 for c in pool)
    logger.info(f"Pool size: {len(pool)} / Cited size: {cited_size} / TOP-K size: {TOP_K}")

    logger.info("Building kw-idf...")
    kw_idf = build_kw_idf(pool)
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
