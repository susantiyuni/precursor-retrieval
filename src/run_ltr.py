import json, argparse, logging
from datetime import datetime
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from lightgbm import LGBMRanker

import utils as u
SEED = 66
u.set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="runs_ltr01/", help="output directory")
parser.add_argument("--inputf", default="data/sparql_feats.jsonl", help="sparql feats file")
parser.add_argument("--feat", default="data/sparql_feats.jsonl", help="sparql feats file")
parser.add_argument("--model", default="lgbm", help="ranking model")
parser.add_argument("--ablation", default="all", help="ablation")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir) 
OUT_DIR.mkdir(parents=True, exist_ok=True)
# DATA_PATH = Path("data/candidate-pool.jsonl")
DATA_PATH = Path(args.inputf) 
# FEATURE_PATH = Path("data/sparql_feats.jsonl")
FEATURE_PATH = Path(args.feat) 
RANKING_MODEL = args.model
ABLATION = args.ablation #all, all+minus_citation, all+minus_metadata...
global_path = OUT_DIR / f"run_ltr_{ABLATION}.jsonl"
TOP_K = 50

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
    "ABLATION": str(ABLATION),
    "MODEL": str(RANKING_MODEL),
    "OUT_DIR": str(OUT_DIR),
    "OUT_FILE": str(global_path)
    }
logging.info("CONFIG:\n%s", json.dumps(config, indent=2))
logger = logging.getLogger(__name__)
logger.info("## START! ##")

def l2norm(x):
    return x / (np.linalg.norm(x) + 1e-8)

def parse_ablation_mode(mode): # "all_minus_sparql+minus_trace_social"
    return set(mode.split("+"))

taken = parse_ablation_mode(ABLATION)
logger.info(f"{taken}")

def build_full_feature(c, query_p, sim_threshold=0.6, mode=ABLATION, kw_idf=None):
    disabled = parse_ablation_mode(mode)
    taken = parse_ablation_mode(mode)

    feats = []
    # ===== Base =====
    paper_emb = u.paper_embedding(c, mode="graph")
    qpaper_emb = u.paper_embedding(query_p, mode="graph")
    temp = u.temporal_features(query_p["year"], c["year"])
    embedding_sim, sim_text, sim_graph = u.embedding_similarity(query_p, c)
    feats.extend([
        paper_emb,
        qpaper_emb,
        temp,
    ])

    # ===== Similarity ===== 
    # if "minus_sim" not in disabled:
    feats.append(np.array([embedding_sim, sim_text, sim_graph], dtype=float))

    # ===== Sparql ===== 
    cited_sparql, uncited_sparql = u.split_sparql_features(c["sparql_features"])
    cited_sparql = np.log1p(cited_sparql)
    uncited_sparql = np.log1p(uncited_sparql)
    # if "minus_sparql" not in disabled:
    feats.extend([cited_sparql, uncited_sparql])

    # ===== Trace =====
    trace_sem = l2norm(u.trace_embedding(c, "semantic"))
    trace_cited = l2norm(u.trace_embedding(c, "cited"))
    trace_struct = l2norm(u.trace_embedding(c, "structural"))
    trace_social = l2norm(u.trace_embedding(c, "social"))

    # if "minus_trace_cited" not in disabled: #harmful
    #     feats.append(trace_cited)
    # if "minus_trace_struct" not in disabled:
    feats.append(trace_struct)
    # if "minus_trace_sem" not in disabled:
    feats.append(trace_sem)
    # if "minus_trace_social" not in disabled:
    feats.append(trace_social)

    # ===== Metadata =====
    S_msc, overlap = u.msc_similarity(query_p.get("mscs", []), c.get("msc_codes", []))
    S_kw = u.keyword_similarity(query_p.get("keywords", []), c.get("keywords", []), kw_idf)
    # if "minus_metadata" not in disabled:
    if "plus_metadata" in taken:
        feats.append(np.array([S_msc, S_kw], dtype=float))

    # ===== Citation =====
    citation_strength = (
        2.0 * c["sparql_features"].get("direct_path", 0) +
        1.0 * c["sparql_features"].get("co_citation_path", 0) +
        1.0 * c["sparql_features"].get("bib_coupling_path", 0)
    )
    is_cited = int(c.get("is_cited", 0))
    # if "minus_citation" not in disabled:
    if "plus_citation" in taken:
        feats.append(np.array([citation_strength, is_cited], dtype=float))

    # ===== Gating =====
    # if "minus_gating" not in disabled:
    if "plus_gating" in taken:
        high_sim_no_cite = float((embedding_sim > sim_threshold) and (not is_cited))
        sim_x_citation = embedding_sim * citation_strength
        sim_x_uncited = embedding_sim * (1 - is_cited)
        cited_mass = np.linalg.norm(cited_sparql)
        uncited_mass = np.linalg.norm(uncited_sparql)
        block_ratio = cited_mass / (uncited_mass + 1e-6)
        gating = np.array([
            embedding_sim,
            high_sim_no_cite,
            sim_x_citation,
            sim_x_uncited,
            # cited_mass,
            # uncited_mass,
            block_ratio
        ], dtype=float)
        feats.append(gating)
    return np.concatenate(feats)

def build_ltr_dataset(entries, leave_out_idx, kw_idf):
    X, y, group = [], [], []
    for i, entry in enumerate(entries):
        if i == leave_out_idx:
            continue
        query_p = entry["query_paper"]
        candidates = sorted(
            entry["candidates"],
            key=lambda c: c["llm_score"],
            reverse=True)
        group.append(len(candidates))
        scores = [c["llm_score"] for c in candidates]
        labels = u.quantize_labels_per_query(scores)
        for c, y_i in zip(candidates, labels):
            # X.append(build_full_feature(c, query_p))
            X.append(build_full_feature(c, query_p, kw_idf=kw_idf))
            y.append(int(y_i))
    return np.array(X), np.array(y), group

# ================================
# TRAIN LGBM RANKER
# ================================
def train_lgbm_ranker(X, y, group):
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=10,
        min_gain_to_split=0.01,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
    )
    model.fit(X, y, group=group)
    return model

# ================================
# RANK/INFERENCE WITH LGBM
# ================================
def rank_lgbm(model, candidates, query_p, kw_idf):
    X = np.array([
        # extract_feature_vector(c["sparql_features"])
        build_full_feature(c, query_p, kw_idf=kw_idf)
        for c in candidates
    ])
    scores = model.predict(X)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [{ "paper": c["paper"],
            "score": float(s),
            "rank": i + 1}
        for i, (c, s) in enumerate(ranked[:TOP_K])]

# ================================
# LOO EVALUATION
# ================================
def run_loo():
    entries = u.load_data(DATA_PATH, FEATURE_PATH)
    # logging.info (entries[0]["candidates"][0]) #sanity check merged data
    all_results = []
    for leave_out in range(len(entries)):
        query_p = entries[leave_out]["query_paper"]
        logging.info (f"\n===== LOO {leave_out} {query_p['year']} =====")
        test_entry = entries[leave_out]
        candidates = test_entry["candidates"]
        # candidates = test_entry["candidate_pool"]

        if RANKING_MODEL in ['lgbm']:
          kw_idf = u.build_kw_idf(candidates)
          X, y, group = build_ltr_dataset(entries, leave_out, kw_idf)
          model_lgbm = train_lgbm_ranker(X, y, group)
          ranked = rank_lgbm(model_lgbm, candidates, query_p, kw_idf)

        elif RANKING_MODEL in ['pw']:
          Xp, yp = build_pairwise_dataset(entries, leave_out)
          model_pw, scaler_pw = train_pairwise(Xp, yp)
          ranked = rank_pairwise(model_pw, scaler_pw, candidates, query_p)

        logging.info(f"\nTop Ranked Precursors: {test_entry['query_paper']['paper']}")
        for r in ranked[:2]:
            logging.info (r)

        output = {"query_paper": test_entry["query_paper"],
            "ranked_candidates": ranked,}
        all_results.append(output)

        # save per-query file
        out_path = OUT_DIR / f"run_{leave_out}_{RANKING_MODEL}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    with global_path.open("w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logging.info (f"\n===== Saved: {global_path} =====")
    return all_results

if __name__ == "__main__":
    run_loo()
