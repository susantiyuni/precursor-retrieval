# python src-ltr/metrics.py --out_dir runs_ce/

import json
import math
from pathlib import Path
from collections import defaultdict
import argparse
import logging
from datetime import datetime
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="output/", help="Output directory")
args = parser.parse_args()

# ================= CONFIG =================
OUT_DIR = Path(args.out_dir)
RELEVANCE_PATH = Path("data/candidate-pool.jsonl")

METHOD_FILES = [
    ###zs
    "run_bm25.jsonl",
    "run_citation.jsonl",
    "run_dualenc.jsonl",
    "run_ppr.jsonl",
    "run_colbert.jsonl",

    ### zs temporal ablation
    # "run_tmgnrxv3_base_gamma.jsonl",
    # "run_tmgnrxv3_citation_gamma.jsonl",
    # "run_tmgnrxv3_explicit_gamma.jsonl",
    # "run_tmgnrxv3_all_gamma.jsonl",

    # "run_tmgnrxv3_base_beta.jsonl",
    # "run_tmgnrxv3_citation_beta.jsonl",
    # "run_tmgnrxv3_explicit_beta.jsonl",
    # "run_tmgnrxv3_all_beta.jsonl",

    # "run_tmgnrxv3_base_decay.jsonl",
    # "run_tmgnrxv3_citation_decay.jsonl",
    # "run_tmgnrxv3_explicit_decay.jsonl",
    # "run_tmgnrxv3_all_decay.jsonl",

    # "run_tmgnrxv3_base_gaussian.jsonl",
    # "run_tmgnrxv3_citation_gaussian.jsonl",
    # "run_tmgnrxv3_explicit_gaussian.jsonl",
    # "run_tmgnrxv3_all_gaussian.jsonl",

    # "run_tmgnrxv3_base_laplace.jsonl",
    # "run_tmgnrxv3_citation_laplace.jsonl",
    # "run_tmgnrxv3_explicit_laplace.jsonl",
    # "run_tmgnrxv3_all_laplace.jsonl",

    # "run_tmgnrxv3_base_lognormal.jsonl",
    # "run_tmgnrxv3_citation_lognormal.jsonl",
    # "run_tmgnrxv3_explicit_lognormal.jsonl",
    # "run_tmgnrxv3_all_lognormal.jsonl",

    ### zs schema ablation
    # "run_tmgnrx_base_int_influence.jsonl",
    # "run_tmgnrx_base_comm_consensus.jsonl",
    # "run_tmgnrx_base_pure_topical_continuity.jsonl",
    # "run_tmgnrx_base_hybrid_topical_continuity.jsonl",
    # "run_tmgnrx_base_hybrid.jsonl",

    #ltr ablation
    # "run_ltr_base.jsonl",
    # "run_ltr_base+plus_metadata.jsonl",
    # "run_ltr_base+plus_citation.jsonl",
    # "run_ltr_base+plus_gating.jsonl",
    # "run_ltr_base+plus_metadata+plus_citation.jsonl",
    # "run_ltr_base+plus_metadata+plus_citation+plus_gating.jsonl",
    ]

TOP_K_NDCG = 50
TOP_K_RECALL = 50
STRICT_THRESHOLD = 0.7
NORMAL_THRESHOLD = 0.5

# ================= LOGGING =================
OUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = OUT_DIR / f"eval_{timestamp}.log"
# log_path = OUT_DIR / "evalx.log"
logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s | %(levelname)s | %(message)s",
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),  # keep console output
    ],
)

logger = logging.getLogger(__name__)
logger.info("# Starting evaluation")
logger.info(f"Output dir: {OUT_DIR}")
logger.info(f"Gold file: {RELEVANCE_PATH}")

# LOAD GOLD JUDGMENTS

gold = {}

with RELEVANCE_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        qid = row["query_paper"]["paper"]

        gold[qid] = {
            c["paper"]: {
                "llm_score": c["llm_score"],
                "is_cited": c["is_cited"],
            }
            for c in row.get("candidates", []) or row.get("ranked_candidates")
        }

logger.info(f"Loaded gold for {len(gold)} queries")

def mean_std(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)), float(np.std(xs))

def dcg(rels):
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

def ndcg_at_k(ranked, judged, k, uncited_only=False):
    rels = []
    for c in ranked[:k]:
        pid = c["paper"]
        if pid not in judged:
            rels.append(0)
            continue
        if uncited_only and judged[pid]["is_cited"] == 1:
            rels.append(0)
        else:
            rels.append(judged[pid]["llm_score"])
    dcg_val = dcg(rels)
    ideal = [ v["llm_score"] for v in judged.values()
        if not uncited_only or v["is_cited"] == 0 ]
    ideal = sorted(ideal, reverse=True)[:k]
    idcg_val = dcg(ideal)
    return None if idcg_val == 0 else dcg_val / idcg_val


def recall_at_k(ranked, judged, k, threshold, uncited_only=False):
    def relevant(v): return v["llm_score"] >= threshold
    if uncited_only:
        relevant_ids = {pid for pid, v in judged.items() if relevant(v) and v["is_cited"] == 0 }
    else:
        relevant_ids = {pid for pid, v in judged.items() if relevant(v)}
    if not relevant_ids:
        return None
    hits = sum(
        1 for c in ranked[:k]
        if c["paper"] in relevant_ids
    )
    return hits / len(relevant_ids)

def precision_at_k(ranked, judged, k, threshold, uncited_only=False):
    def relevant(v): return v["llm_score"] >= threshold

    if uncited_only:
        relevant_ids = { pid for pid, v in judged.items() if relevant(v) and v["is_cited"] == 0 }
    else:
        relevant_ids = { pid for pid, v in judged.items() if relevant(v)}
    top_k = ranked[:k]
    if not top_k:
        return None
    hits = sum(
        1 for c in top_k
        if c["paper"] in relevant_ids
    )
    return hits / len(top_k)

def average_precision(ranked, judged, threshold, uncited_only=False):
    def relevant(v): 
        return v["llm_score"] >= threshold

    if uncited_only:
        rel_set = {
            pid for pid, v in judged.items()
            if relevant(v) and v["is_cited"] == 0
        }
    else:
        rel_set = {
            pid for pid, v in judged.items()
            if relevant(v)
        }

    if not rel_set:
        return None

    hits = 0
    ap = 0

    for i, c in enumerate(ranked, start=1):
        pid = c["paper"]

        if pid in rel_set:
            hits += 1
            ap += hits / i

    return ap / len(rel_set)

def mrr(ranked, judged, threshold):
    rel_set = {pid for pid, v in judged.items() if v["llm_score"] >= threshold}
    for i, c in enumerate(ranked, start=1):
        if c["paper"] in rel_set:
            return 1 / i
    return 0


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0


for method_file in METHOD_FILES:
    path = OUT_DIR / method_file
    if not path.exists():
        logger.warning(f"Skip {method_file}")
        continue
    per_query = []
    logger.info(f"\n # Evaluating {method_file}")
    per_query_path = OUT_DIR / f"{method_file.replace('.jsonl','')}_per_query_metrics.jsonl"

    with path.open("r", encoding="utf-8") as f, per_query_path.open("w", encoding="utf-8") as out_f:
        for line in f:
            row = json.loads(line)

            qid = row["query_paper"]["paper"] 
            # qid = row["query_paper"]
            if qid not in gold:
                logger.debug(f"Skipping qid {qid} (not in gold)")
                continue

            judged = gold[qid]

            # support both key names
            ranked = row.get("ranked_candidates") or row.get("retrieved_candidates")
            # pid = c["paper"]["paper"] if isinstance(c["paper"], dict) else c["paper"]

            ranked = [{"paper": c["paper"]} for c in ranked]

            if not ranked:
                logger.debug(f"No ranked candidates for qid {qid}")
                continue

            metrics = {
                "nDCG@": ndcg_at_k(ranked, judged, TOP_K_NDCG),
                "nDCG@_uncited": ndcg_at_k(ranked, judged, TOP_K_NDCG, uncited_only=True),

                "Recall": recall_at_k(ranked, judged, TOP_K_RECALL, NORMAL_THRESHOLD),
                "Recall_uncited": recall_at_k(ranked, judged, TOP_K_RECALL, NORMAL_THRESHOLD, True),

                "RecallS": recall_at_k(ranked, judged, TOP_K_RECALL, STRICT_THRESHOLD),
                "RecallS_uncited": recall_at_k(ranked, judged, TOP_K_RECALL, STRICT_THRESHOLD, True),

                "Precision": precision_at_k(ranked, judged, TOP_K_RECALL, NORMAL_THRESHOLD),
                "Precision_uncited": precision_at_k(ranked, judged, TOP_K_RECALL, NORMAL_THRESHOLD, True),

                "PrecisionS": precision_at_k(ranked, judged, TOP_K_RECALL, STRICT_THRESHOLD),
                "PrecisionS_uncited": precision_at_k(ranked, judged, TOP_K_RECALL, STRICT_THRESHOLD, True),

                "MAP": average_precision(ranked, judged, NORMAL_THRESHOLD),
                "MAP_uncited": average_precision(ranked, judged, NORMAL_THRESHOLD, True),
            }

            per_query.append(metrics)
            out_f.write(json.dumps(metrics) + "\n")
    
    logger.info(
        f"{method_file}: evaluated {len(per_query)} queries "
        f"(per-query metrics -> {per_query_path.name})"
    )

    # ================= REPORT =================
    print(f"===== {method_file} =====")
    for key in [k for k in per_query[0].keys() if k not in ("qid", "method")]:
        # val = mean([q[key] for q in per_query])
        mean_val, std_val = mean_std([q[key] for q in per_query])
        # print(f"{key:18s}: {val:.3f}")
        logger.info(f"{method_file} | {key} = {mean_val:.4f} ± {std_val:.4f}")
    print(f"\n")
