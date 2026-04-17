import random, json, os
import math
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
from pathlib import Path
import torch

SEED = 66
np.random.seed(SEED)
random.seed(SEED)

MSC_FILE = Path("data/msc_codes.jsonl")
msc_lookup = {}
with open(MSC_FILE, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        msc_lookup[entry["code"]] = entry

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

def extract_feature_vector(sparql_feats):
    return np.array([sparql_feats.get(k, 0.0) for k in FEATURE_KEYS], dtype=float)

def normalize_labels(scores):
    scores = np.array(scores)
    if scores.max() == scores.min():
        return np.ones_like(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores

def quantize_labels(scores, bins=5):
    scores = np.array(scores)
    bins = np.linspace(scores.min(), scores.max(), bins)
    return np.digitize(scores, bins)

def quantize_labels_per_query(scores, n_bins=5): #for lgbm, pairwise
    scores = np.array(scores)
    # rank-based binning (robust)
    ranks = scores.argsort().argsort()
    bins = np.floor(n_bins * ranks / len(scores)).astype(int)
    return bins

def verbalize_trace(trace):
    rel_map = {
        "CITES": "cites",
        "CITED_BY": "is cited by",
        "HAS_TOPIC": "has topic",
        "HAS_SUBJECT": "has subject",
        "SHARED_SUBJECT": "shares subject with",
        "SHARED_TOPIC": "shares topic with",
        "HAS_KEYWORD": "has keyword",
        "SHARED_KEYWORD": "shares keyword with",
        "BIB_COUPLED": "shares references with",
        "CO_CITED_BY": "is co-cited with",
        "IS_PART_OF": "published in",
        "SHARED_VENUE": "same venue as",
        "HAS_AUTHOR": "has author",
        "COAUTHOR_LINK": "co-authored with",
        "REVIEWED_BY": "reviewed by",
        "SHARED_REVIEWER": "same reviewer as"
    }
    parts = []
    for edge in trace.get("graph", []):
        src = edge["src"]
        tgt = edge["tgt"]
        rel = rel_map.get(edge["rel"], edge["rel"].lower())
        parts.append(f"{src} {rel} {tgt}")
    if not parts:
        return "no connection"
    return " ; ".join(parts)

def candidate_text(candidate):
    traces = candidate.get("traces", [])
    if not traces:
        return "no known connection between query and candidate"
    # return " ; ".join(verbalize_trace(t) for t in traces)
    return f"{len(traces)} paths: " + " ; ".join(verbalize_trace(t) for t in traces)

def get_embedding(candidate):
    text = candidate_text(candidate)
    return embed(text, MODEL)

# ================================
# BUILD TEXT INPUT
# ================================
def build_query_text(q):
    return (
        f"Find prior work related to the paper: "
        f"URI: {q['paper']}. "
        f"Title: {q['title']}. "
        f"Publication year: {q['year']}. "
        f"Subjects: {q['mscs'][:20]}. "
        f"Keywords: {q['keywords'][:20]}. "
        f"References: {q['references'][:20]}. "
        # f"Review: {c["review"][:1000]}."
    )
    # return f"Find prior work related to paper with id {qid["paper"]}. Title: {qid["title"]}. Publication year: {qid["year"]}. Topics: {qid["mscs"]}. Keywords: {qid["keywords"]}. References: {qid["references"][:20]}"

def format_scores(scores):
    if not scores:
        return "no scores"
    # keep only non-zero signals
    active = {k: round(v, 3) for k, v in scores.items() if v != 0}
    return ", ".join([f"{k}={v}" for k, v in active.items()])

def build_candidate_text(c):
    traces = c.get("traces", [])
    sparql_scores = c.get("sparql_features", {})
    path_strings = []
    for t in traces:
        v = verbalize_trace(t)
        if isinstance(v, list):
            v = " ".join(map(str, v))
        path_strings.append(v)

    if path_strings:
        path_text = " ; ".join(path_strings)
    else:
        path_text = "no known connection"
    return (
        f"Candidate paper URI: {c['paper']}. "
        f"Title: {c['title']}. "
        f"Publication year: {c['year']}. "
        f"Subjects: {c['msc_codes'][:20]}. "
        f"Keywords: {c['keywords'][:20]}. "
        f"References: {c['references'][:20]}. "
        f"Cited by: {c['is_cited']}. "
        f"Evidence: {path_text}. "
        f"Scores: {format_scores(sparql_scores)}."
        # f"Review: {c["review"][:1000]}."
        )

# ================= HELPERS =================
def load_data_path(data_path):
    entries = []
    with data_path.open() as f:
        for line in f:
            entries.append(json.loads(line))
    return entries

def load_data(data_path, feature_path):
    # ---- load feature data and build lookup ----
    feature_lookup = {}
    with open(feature_path) as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["query_paper"]
            feature_lookup[qid] = {
                c["paper"]: c for c in entry.get("candidates", []) }
    # ---- load main data and merge ----
    merged = []
    with open(data_path) as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["query_paper"]["paper"]
            feature_candidates = feature_lookup.get(qid, {})
            new_candidates = []
            for c in entry.get("candidates", []):
                pid = c["paper"]
                feat = feature_candidates.get(pid, {})
                merged_candidate = {
                    **c,
                    "sparql_features": feat.get("sparql_features"),
                    "traces": feat.get("traces"), }
                new_candidates.append(merged_candidate)
            entry["candidates"] = new_candidates
            merged.append(entry)
    return merged
    
def time_filtered_pool(query, candidates, YEAR_GAP=10):
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

def msc_node_id(msc):
    return f"msc:{norm_msc(msc)}"

def msc_similarity(msc_q, msc_c):
    if not msc_q or not msc_c:
        return 0.0, set()
    q_msc = msc_prefixes(msc_q)
    c_msc = msc_prefixes(msc_c)
    overlap = q_msc & c_msc
    msc_score = (
        1.0 * len([x for x in overlap if len(x) == 5]) +  # exact code
        0.6 * len([x for x in overlap if len(x) == 3]) +  # mid
        0.3 * len([x for x in overlap if len(x) == 2])    # broad
    )
    return msc_score, overlap

def keyword_similarity_(kw_q, kw_c):
    if not kw_q or not kw_c:
        return 0.0, set()
    q_kw = {norm_kw(k) for k in kw_q}
    c_kw = {norm_kw(k) for k in kw_c}
    overlap = q_kw & c_kw
    kw_score = len(overlap) / max(1, len(q_kw | c_kw))
    return kw_score, overlap

def keyword_similarity(kw_q, kw_c, kw_idf):
    if not kw_q or not kw_c:
        return 0.0
    q_kw = {norm_kw(k) for k in kw_q}
    c_kw = {norm_kw(k) for k in kw_c}
    kw_score = sum(kw_idf.get(k, 0.0) for k in (q_kw & c_kw))
    kw_score_idf = kw_score / (sum(kw_idf.values()) + 1e-9)
    return kw_score_idf

def metapath_score(query, candidate):
    f_msc = len(set(query.get("mscs", [])) & set(candidate.get("msc_codes", [])))
    f_kw = len(set(query.get("keywords", [])) & set(candidate.get("keywords", [])))
    f_ref = len(set(query.get("references", [])) & set(candidate.get("references", [])))

    return [f_msc, f_kw, f_ref]

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

def build_candidate_pool(query, candidates, pool_un=150, pool_ci=50):
    uncited = sorted(
        (c for c in candidates if c.get("is_cited") == 0),
        key=lambda c: c.get("llm_score", 0),
        reverse=True)[:pool_un]
    cited = sorted(
        (c for c in candidates if c.get("is_cited") == 1),
        key=lambda c: c.get("llm_score", 0),
        reverse=True)[:pool_ci]
    cited = time_filtered_pool(query, cited)
    random.shuffle(uncited)
    random.shuffle(cited)
    pool = uncited + cited
    random.shuffle(pool)
    return pool

def beta_pdf(x, alpha, beta):
    """Compute Beta PDF manually (no scipy dependency)."""
    if x <= 0 or x >= 1:
        return 0.0
    B = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
    return (x ** (alpha - 1) * (1 - x) ** (beta - 1)) / B

def gamma_pdf(dt, k=2, theta=5):
    if dt <= 0:
        return 0.0
    return (dt ** (k - 1) * math.exp(-dt / theta)) / \
           (math.gamma(k) * theta ** k)

def lognormal(dt, mu=2.5, sigma=0.5):
    if dt <= 0:
        return 0.0
    return (1 / (dt * sigma * math.sqrt(2 * math.pi))) * \
           math.exp(-((math.log(dt) - mu) ** 2) / (2 * sigma ** 2))

def temporal_hist_modeling(year_q, year_c, mode="decay", mu=12, sigma=6, tau=25, alpha=2, beta=5, max_dt=50):
    dt = year_q - year_c
    if dt <= 0:
        return 0.0
    if mode == "gaussian":
        return math.exp(-((dt - mu) ** 2) / (2 * sigma ** 2))
    elif mode == "laplace":
        return math.exp(-abs(dt - mu) / sigma)
    elif mode == "decay":
        return math.exp(-dt / tau)
    elif mode == "beta":
        # Normalize dt to [0,1]
        x = min(dt / max_dt, 1.0)
        return beta_pdf(x, alpha, beta)
    elif mode == "gamma":
        return gamma_pdf(dt)
    elif mode == "lognormal":
        return lognormal(dt)
    else:
        raise ValueError("mode must be one of: gaussian, laplace, decay, beta")

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def set_deterministic():
    torch.cuda.empty_cache()
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
