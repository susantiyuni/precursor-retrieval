import json
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON
from string import Template
from functools import lru_cache
import math
import numpy as np
from config import ENDPOINT_URL

# ---------------- CONFIG ----------------
sparql = SPARQLWrapper(ENDPOINT_URL)
sparql.setReturnFormat(JSON)

INPUT_FILE = Path("data/candidate-pool.jsonl")
OUTPUT_FILE = Path("data/sparql_feats.jsonl")
PATH_LIMIT = 10

def run_sparql(query_str):
    sparql.setQuery(query_str)
    return sparql.query().convert()["results"]["bindings"]

# # ---------------- PATH FEATURE DEFINITIONS ----------------
# PATH_TYPES = {
#     "direct": "Q->C (Q cites C)",
#     "reverse": "C->Q (C cites Q)",

#     "co_citation": "Q<-X->C",
#     "bib_coupling": "Q->X<-C",
#     "two_hop": "Q->M->C",

#     "msc_path": "Q->MSC->X->C",
#     "keyword_path": "Q->KW->X->C",
#     "author_path": "Q->A->X->C",

#     "hybrid_path": "Q->MSC->X->cites->C"
# }

def norm_log(x):
    return math.log(1 + x)

def make_trace(path_type, nodes, edges, weight=1):
    assert len(nodes) == len(edges) + 1, (
        f"Invalid trace: nodes={len(nodes)} edges={len(edges)} "
        f"type={path_type}"
    )
    return {
        "type": path_type,
        "weight": float(weight),
        "graph": [
            {
                "src": nodes[i],
                "rel": edges[i],
                "tgt": nodes[i+1],
                "w": weight
            }
            for i in range(len(edges))
        ],
        "linear": " ".join(
            f"{nodes[i]} -[{edges[i]}]-> {nodes[i+1]}"
            for i in range(len(edges))
        ),
        "tokens": " ".join(
            [nodes[0]] +
            sum([[edges[i], nodes[i+1]] for i in range(len(edges))], [])
        )
    }

def strict_json(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: strict_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict_json(v) for v in obj]
    return obj

def topk(d, k=50):
    if isinstance(d, dict):
        return dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:k])
    if isinstance(d, set):
        return list(sorted(list(d))[:k])
    return d

# (A) Direct citation (explicit precursor)
#@lru_cache(maxsize=10000)
def direct_paths(paper):
    q = f"""
    PREFIX cito: <http://purl.org/spar/cito/>
    SELECT ?cand WHERE {{
        <{paper}> cito:cites ?cand .
    }}
    """
    res = run_sparql(q)
    counts = {}
    paths = {}
    for r in res:
        cand = r["cand"]["value"]
        paths[cand] = []
        counts[cand] = 1
    return counts, paths

# (B) Backward citation
#@lru_cache(maxsize=10000)
def reverse_paths(paper):
    q = f"""
    PREFIX cito: <http://purl.org/spar/cito/>
    SELECT ?cand WHERE {{
        ?cand cito:cites <{paper}> .
    }}
    """
    res = run_sparql(q)
    counts = {}
    paths = {}
    for r in res:
        cand = r["cand"]["value"]
        paths[cand] = []
        counts[cand] = 1
    return counts, paths

# (D) Co-citation 
# paper ← cites ← paper → cites → paper
# x	cand
# https://zbmath.org/7507731	https://zbmath.org/4033738
# https://zbmath.org/7507731	https://zbmath.org/1104447
# https://zbmath.org/7507731	https://zbmath.org/3677903
#@lru_cache(maxsize=10000)
def co_citation_paths(paper):
    q = f"""
    PREFIX cito: <http://purl.org/spar/cito/>
    SELECT ?x ?cand WHERE {{
        ?x cito:cites <{paper}> .
        ?x cito:cites ?cand .
        <{paper}> dcterms:issued ?qy .
        ?cand dcterms:issued ?cy .
        FILTER(?cy < ?qy)
        FILTER(?cand != <{paper}>)
    }}
    """
    res = run_sparql(q)
    counts = {}
    paths = {}
    for r in res:
        x = r["x"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add(x)
    counts = {cand: len(xs) for cand, xs in paths.items()}
    # paths = {cand: list(xs) for cand, xs in paths.items()}
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths

# (C) Shared references (bibliographic coupling precursor)
# ref	cand
# https://zbmath.org/3781163	https://zbmath.org/951945
# https://zbmath.org/3781163	https://zbmath.org/951946
# https://zbmath.org/3781164	https://zbmath.org/1190156
#@lru_cache(maxsize=10000)
def bib_coupling_paths(paper):
    q = f"""
    PREFIX cito: <http://purl.org/spar/cito/>
    SELECT ?ref ?cand WHERE {{
        <{paper}> cito:cites ?ref .
        ?cand cito:cites ?ref .
        <{paper}> dcterms:issued ?qy .
        ?cand dcterms:issued ?cy .
        FILTER(?cy < ?qy)
        FILTER(?cand != <{paper}>)
    }}
    """
    res = run_sparql(q)
    counts = {}
    paths = {}
    for r in res:
        ref = r["ref"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add(ref)
    counts = {cand: len(xs) for cand, xs in paths.items()}
    # paths = {cand: list(xs) for cand, xs in paths.items()}
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths

# (H) 2-hop citation expansion (latent precursor)
# mid	cand
# https://zbmath.org/169663	https://zbmath.org/31344
# https://zbmath.org/169663	https://zbmath.org/4033739
# https://zbmath.org/169663	https://zbmath.org/432707
# https://zbmath.org/169663	https://zbmath.org/3572133
#@lru_cache(maxsize=10000)
def two_hop_paths(paper):
    q = f"""
    PREFIX cito: <http://purl.org/spar/cito/>
    SELECT ?mid ?cand WHERE {{
        <{paper}> cito:cites ?mid .
        ?mid cito:cites ?cand .
        <{paper}> dcterms:issued ?qy .
        ?cand dcterms:issued ?cy .
        FILTER(?cy < ?qy)
        FILTER(?cand != <{paper}>)
    }}
    """
    res = run_sparql(q)
    counts = {}
    paths = {}
    for r in res:
        mid = r["mid"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add(mid)
    counts = {cand: len(xs) for cand, xs in paths.items()}
    # paths = {cand: list(xs) for cand, xs in paths.items()}
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths

def msc_direct_overlap(paper):
    q = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    SELECT ?msc ?cand WHERE {{
        <{paper}> dcterms:subject ?msc .
        ?cand dcterms:subject ?msc .
        FILTER(?cand != <{paper}>)
    }}
    """
    res = run_sparql(q)
    paths = {}
    for r in res:
        msc = r["msc"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add(msc)
    counts = {c: len(v) for c, v in paths.items()}
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths

# (E) Subject overlap 
# msc	x	cand
# http://msc2010.org/resources/MSC/2010/03B30	https://zbmath.org/12855	https://zbmath.org/140573
# http://msc2010.org/resources/MSC/2010/03B30	https://zbmath.org/12855	https://zbmath.org/3085434
# http://msc2010.org/resources/MSC/2010/03B30	https://zbmath.org/12855	https://zbmath.org/3549907
# http://msc2010.org/resources/MSC/2010/03B30	https://zbmath.org/12855	https://zbmath.org/3874308
#@lru_cache(maxsize=10000)
def hybrid_msc_paths(paper):
    q = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX cito: <http://purl.org/spar/cito/>
    SELECT ?msc ?x ?cand WHERE {{
        <{paper}> dcterms:subject ?msc .
        ?x dcterms:subject ?msc .
        ?x cito:cites ?cand .
        ?cand dcterms:issued ?cy .
        <{paper}> dcterms:issued ?qy .
        FILTER(?cy < ?qy)
        FILTER(?cand != <{paper}>)
    }}
    """
    res = run_sparql(q)
    counts = {}
    paths = {}
    for r in res:
        msc = r["msc"]["value"]
        x = r["x"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add((msc, x))
    counts = {cand: len(v) for cand, v in paths.items()}
    # paths = {cand: list(v) for cand, v in paths.items()}
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths

def keyword_direct_overlap(paper):
    q = f"""
    PREFIX schema: <https://schema.org/>
    SELECT ?kw ?cand WHERE {{
        <{paper}> schema:keywords ?kw .
        ?cand schema:keywords ?kw .
        FILTER(?cand != <{paper}>)
    }}
    """
    res = run_sparql(q)
    paths = {}
    for r in res:
        kw = r["kw"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add(kw)
    counts = {c: len(v) for c, v in paths.items()}
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths

# (F) Keyword overlap 
# kw	x	cand
# https://zbmath.org/keyword/derivatives	https://zbmath.org/1045590	https://zbmath.org/168345
# https://zbmath.org/keyword/derivatives	https://zbmath.org/1045590	https://zbmath.org/3351017
# https://zbmath.org/keyword/derivatives	https://zbmath.org/1045590	https://zbmath.org/540410
# https://zbmath.org/keyword/derivatives	https://zbmath.org/1045590	https://zbmath.org/644426
#@lru_cache(maxsize=10000)
def hybrid_keyword_paths(paper):
    q = f"""
    PREFIX schema: <https://schema.org/>
    PREFIX cito: <http://purl.org/spar/cito/>
    SELECT ?kw ?x ?cand WHERE {{
        <{paper}> schema:keywords ?kw .
        ?x schema:keywords ?kw .
        ?x cito:cites ?cand .
        ?cand dcterms:issued ?cy .
        <{paper}> dcterms:issued ?qy .
        FILTER(?cy < ?qy)
        FILTER(?cand != <{paper}>)
    }}
    """
    res = run_sparql(q)
    counts = {}
    paths = {}
    for r in res:
        kw = r["kw"]["value"]
        x = r["x"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add((kw, x))
    counts = {cand: len(v) for cand, v in paths.items()}
    # paths = {cand: list(v) for cand, v in paths.items()}
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths

# (G) Author lineage (intellectual continuity)
# a	cand
# https://zbmath.org/authors/rathjen.michael	https://zbmath.org/4079410
# https://zbmath.org/authors/rathjen.michael	https://zbmath.org/4200195
# https://zbmath.org/authors/rathjen.michael	https://zbmath.org/108324
#@lru_cache(maxsize=10000)
def author_paths(paper):
    q = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    SELECT ?a ?cand WHERE {{
        <{paper}> dcterms:creator ?a .
        ?cand dcterms:creator ?a .
        <{paper}> dcterms:issued ?qy .
        ?cand dcterms:issued ?cy .
        FILTER(?cy < ?qy)
        FILTER(?cand != <{paper}>)
    }}
    """
    res = run_sparql(q)
    paths = {}
    for r in res:
        a = r["a"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add(a)
    counts = {cand: len(authors) for cand, authors in paths.items()}
    # paths = {cand: list(authors) for cand, authors in paths.items()}
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths

# topical similarity (MSC) + citation flow (other → cand) + temporal constraint
# msc	other	cand
# http://msc2010.org/resources/MSC/2010/03B30	https://zbmath.org/6693043	https://zbmath.org/6505017
# http://msc2010.org/resources/MSC/2010/03B30	https://zbmath.org/7179727	https://zbmath.org/2751662
#@lru_cache(maxsize=10000)
# def hybrid_precursor(paper):
#     q = f"""
#     PREFIX dcterms: <http://purl.org/dc/terms/>
#     PREFIX cito: <http://purl.org/spar/cito/>
#     SELECT ?msc ?other ?cand WHERE {{
#       <{paper}> dcterms:subject ?msc .
#       ?other dcterms:subject ?msc .
#       ?other cito:cites ?cand .
#       <{paper}> dcterms:issued ?qy .
#       ?cand dcterms:issued ?cy .
#       FILTER(?cy < ?qy)
#       FILTER(?other != <{paper}>)
#     }}
#     """
#     res = run_sparql(q)
#     counts = {}
#     paths = {}
#     for r in res:
#         msc = r["msc"]["value"]
#         other = r["other"]["value"]
#         cand = r["cand"]["value"]
#         paths.setdefault(cand, set()).add((msc, other))
#         counts[cand] = len(paths[cand])
#     # paths = {cand: list(v) for cand, v in paths.items()}
#     paths = {cand: sorted(v) for cand, v in paths.items()}
#     return counts, paths

#venue lineage
def venue_paths(paper):
    q = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    SELECT ?journal ?cand WHERE {{
        <{paper}> dcterms:isPartOf ?journal .
        ?cand dcterms:isPartOf ?journal .

        <{paper}> dcterms:issued ?qy .
        ?cand dcterms:issued ?cy .
        FILTER(?cy < ?qy)
        FILTER(?cand != <{paper}>)
    }}
    """
    res = run_sparql(q)
    paths = {}
    for r in res:
        a = r["journal"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add(a)
    counts = {cand: len(j) for cand, j in paths.items()}
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths

# co_author_neighborhood
def coauthor_path(paper):
    q = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    SELECT ?a1 ?a2 ?cand WHERE {{
        <{paper}> dcterms:creator ?a1 .
        ?cand dcterms:creator ?a2 .
        FILTER(?a1 != ?a2)
        
        ?mid dcterms:creator ?a1 .
        ?mid dcterms:creator ?a2 .

        FILTER(?cand != <{paper}>)
        ?cand dcterms:issued ?cy .
        <{paper}> dcterms:issued ?qy .        
        FILTER(?cy < ?qy)
        
    }}
    """
    res = run_sparql(q)
    paths = {}
    counts = {}
    for r in res:
        a1, a2 = r["a1"]["value"], r["a2"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add((a1, a2))
        counts[cand] = len(paths[cand])
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths

# author → MSC history → candidate MSC
def author_topic_traj(paper):
    q = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    SELECT ?a ?msc ?cand WHERE {{
        <{paper}> dcterms:creator ?a .
        ?p dcterms:creator ?a .
        ?p dcterms:subject ?msc .
        ?cand dcterms:subject ?msc .

        ?cand dcterms:issued ?cy .
        <{paper}> dcterms:issued ?qy .        
        FILTER(?cy < ?qy)
        FILTER(?cand != <{paper}>)
        
    }}
    """
    res = run_sparql(q)
    paths = {}
    counts = {}
    for r in res:
        a, msc = r["a"]["value"], r["msc"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add((a, msc))
        counts[cand] = len(paths[cand])
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths

# reviewer → keyword → candidate
def reviewer_path(paper):
    q = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX schema: <https://schema.org/>
    SELECT ?reviewer ?cand WHERE {{
        <{paper}> schema:review ?r .
        ?r schema:reviewer ?reviewer .

        ?cand schema:review ?r2 .
        ?r2 schema:reviewer ?reviewer .

        ?cand dcterms:issued ?cy .
        <{paper}> dcterms:issued ?qy .        
        FILTER(?cy < ?qy)
        FILTER(?cand != <{paper}>)
        
    }}
    """
    res = run_sparql(q)
    paths = {}
    for r in res:
        reviewer = r["reviewer"]["value"]
        cand = r["cand"]["value"]
        paths.setdefault(cand, set()).add(reviewer)
    counts = {cand: len(r) for cand, r in paths.items()}
    paths = {cand: sorted(v) for cand, v in paths.items()}
    return counts, paths


# ---------------- LOAD DATA ----------------
data = []
with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded {len(data)} query groups")

output = []

for entry in data:
    query = entry["query_paper"]
    qid = query["paper"]
    qyear = int(query["year"])

    print(f"Processing {qid}")

    # -------- QUERY-LEVEL SPARQL CACHE --------
    direct_cnt, direct_p = direct_paths(qid)
    reverse_cnt, reverse_p = reverse_paths(qid)
    two_cnt, two_p = two_hop_paths(qid)
    cocite_cnt, cocite_p = co_citation_paths(qid)
    bib_cnt, bib_p = bib_coupling_paths(qid)
    author_cnt, author_p = author_paths(qid)
    msc_cnt, msc_p = msc_direct_overlap(qid)
    kw_cnt, kw_p = keyword_direct_overlap(qid)
    hybmsc_cnt, hybmsc_p = hybrid_msc_paths(qid)
    hybkw_cnt, hybkw_p = hybrid_keyword_paths(qid)
    venue_cnt, venue_p = venue_paths(qid)
    reviewer_cnt, reviewer_p = reviewer_path(qid)
    coauthor_cnt, coauthor_p = coauthor_path(qid)
    coauthortop_cnt, coauthortop_p = author_topic_traj(qid)

    new_candidates = []
    for c in entry["candidates"]:
        cid = c["paper"]
        cyear = int(c["year"])

        # binary features → flags
        structure_flags = {
            "direct_path": int(cid in direct_cnt),
            "reverse_path": int(cid in reverse_cnt),
            "two_hop_path": int(cid in two_cnt),
            "venue_path": int(venue_cnt.get(cid, 0) > 0)
        }
        # multi-path features → weights
        structure_weights = {
            "author_path": norm_log(author_cnt.get(cid, 0)),
            "co_citation_path": norm_log(cocite_cnt.get(cid, 0)),
            "bib_coupling_path": norm_log(bib_cnt.get(cid, 0)),
            "reviewer_path": norm_log(reviewer_cnt.get(cid, 0)),
            "coauthor_path": norm_log(coauthor_cnt.get(cid, 0)),
            "coauthortop_path": norm_log(coauthortop_cnt.get(cid, 0))
        }

        semantic = {
            "msc_path": norm_log(msc_cnt.get(cid, 0)),
            "keyword_path": norm_log(kw_cnt.get(cid, 0)),
            "hybrid_msc": norm_log(hybmsc_cnt.get(cid, 0)),
            "hybrid_keyword": norm_log(hybkw_cnt.get(cid, 0))
        }

        traces = []
        # DIRECT PATH
        if cid in direct_cnt:
            traces.append(make_trace(
                "direct",
                [qid, cid],
                ["CITES"],
                1
            ))
        if cid in reverse_cnt:
            traces.append(make_trace(
                "reverse",
                [cid, qid],
                ["CITES"],
                1
            ))
        for mid in two_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "two_hop",
                [qid, mid, cid],
                ["CITES", "CITES"],
                1
            ))

        # CO-CITATION PATH # Q <- X -> C
        for x in cocite_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "co_citation",
                [qid, x, cid],
                # ["CITED_BY", "CITES"],
                ["CO_CITED_BY", "CO_CITED_BY"],
                cocite_cnt[cid]
            ))

        # BIB COUPLING # Q -> X <- C
        for ref in bib_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "bib_coupling",
                [qid, ref, cid],
                # ["CITES", "CITED_BY"],
                ["BIB_COUPLED", "BIB_COUPLED"],
                bib_cnt[cid]
            ))

        # AUTHOR (Q → A → C)
        for a in author_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "author",
                [qid, a, cid],
                ["CREATED_BY", "CREATED_BY"],
                author_cnt[cid]
            ))
        
        for msc in msc_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "msc_direct",
                [qid, msc, cid],
                ["HAS_SUBJECT", "SHARED_SUBJECT"],
                msc_cnt[cid]
            ))
        
        for msc, x in hybmsc_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "hybrid_msc",
                [qid, msc, x, cid],
                ["HAS_SUBJECT", "SHARED_SUBJECT", "CITES"],
                hybmsc_cnt[cid]
            ))
        
        for kw in kw_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "keyword_direct",
                [qid, kw, cid],
                ["HAS_KEYWORD", "SHARED_KEYWORD"],
                kw_cnt[cid]
            ))
        
        for kw, x in hybkw_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "hybrid_keyword",
                [qid, kw, x, cid],
                ["HAS_KEYWORD", "SHARED_KEYWORD", "CITES"],
                hybkw_cnt[cid]
            ))
            
        for journal in venue_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "venue",
                [qid, journal, cid],
                ["IS_PART_OF", "SHARED_VENUE"],
                venue_cnt[cid]
            ))
        
        for a1, a2 in coauthor_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "coauthor",
                [qid, a1, a2, cid],
                ["HAS_AUTHOR", "COAUTHOR_LINK", "HAS_AUTHOR"],
                coauthor_cnt[cid]
            ))
        
        for a, msc in coauthortop_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "author_topic_traj",
                [qid, a, msc, cid],
                ["HAS_AUTHOR", "HAS_PAST_TOPIC", "HAS_TOPIC"],
                coauthortop_cnt[cid]
            ))
        
        for reviewer in reviewer_p.get(cid, [])[:PATH_LIMIT]:
            traces.append(make_trace(
                "reviewer",
                [qid, reviewer, cid],
                ["REVIEWED_BY", "SHARED_REVIEWER"],
                reviewer_cnt[cid]
            ))
            
        feats = {
            **structure_flags,
            **structure_weights,
            **semantic,
            # "total_structure_strength":
            #     sum(structure_weights.values()),
            # "total_semantic_strength":
            #     sum(semantic.values()),
            "graph_semantic_bridge": #“general structural activation weighted by topic relevance”
                sum(structure_flags.values()) * semantic["msc_path"],
            "co_citation_×_keyword":
                structure_weights["co_citation_path"] * semantic["keyword_path"], 
            "co_citation_x_msc":
                structure_weights["co_citation_path"] * semantic["msc_path"],
            "bib_x_keyword":
                structure_weights["bib_coupling_path"] * semantic["keyword_path"],
            "author_x_topic":
                structure_weights["author_path"] * semantic["msc_path"],
            "signal_coverage": (
                structure_flags["direct_path"] +
                structure_flags["reverse_path"] +
                structure_flags["two_hop_path"] +
                structure_flags["venue_path"] +
                int(structure_weights["co_citation_path"] > 0) +
                int(structure_weights["bib_coupling_path"] > 0) +
                int(structure_weights["author_path"] > 0) +
                int(structure_weights["reviewer_path"] > 0) +
                int(structure_weights["coauthor_path"] > 0) +
                int(structure_weights["coauthortop_path"] > 0) +
                int(semantic["msc_path"] > 0) +
                int(semantic["keyword_path"] > 0) +
                int(semantic["hybrid_msc"] > 0) +
                int(semantic["hybrid_keyword"] > 0)
            )
        }


        new_candidates.append({
            "paper": cid,
            "year": cyear,
            "sparql_features": feats,
            "traces": traces, 
        })

    output.append({
        "query_paper": qid,
        "candidates": new_candidates
    })
    
with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for row in output:
        f.write(json.dumps(strict_json(row), ensure_ascii=False) + "\n")

print(f"Saved → {OUTPUT_FILE}")

