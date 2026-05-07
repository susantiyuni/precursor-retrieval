import json
from pathlib import Path

import json
import numpy as np
from scipy.stats import spearmanr, rankdata


# ---------- convert relevance scores -> rank dict ----------
def scores_to_rank_dict(candidates):
    papers = [c["paper"] for c in candidates]
    scores = [c["relevance"] for c in candidates]

    # higher relevance = better rank
    ranks = rankdata([-s for s in scores], method="average")

    return dict(zip(papers, ranks))


# ---------- ordered candidate list -> rank dict ----------
def list_to_rank_dict(candidates):
    return {
        c["paper"]: i
        for i, c in enumerate(candidates, start=1)
    }


# ---------- align rankings ----------
def align(human_rank, system_rank):
    common = set(human_rank) & set(system_rank)

    human = [human_rank[p] for p in common]
    system = [system_rank[p] for p in common]

    return human, system


# ---------- evaluate one query ----------
def evaluate_query(human_candidates, system_candidates):
    human_rank = scores_to_rank_dict(human_candidates)
    system_rank = list_to_rank_dict(system_candidates)
    print (human_rank)
    print (system_rank)
    # sys.exit()

    human, system = align(human_rank, system_rank)

    if len(human) < 2:
        return np.nan

    sp, _ = spearmanr(human, system)

    return sp


# ---------- load proper JSONL ----------
def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


# ---------- save JSONL ----------
def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


# ---------- full evaluation ----------
def evaluate_jsonl(human_file, system_file, output_file=None):
    human_data = load_jsonl(human_file)
    system_data = load_jsonl(system_file)

    print(f"Loaded: {len(human_data)} human / {len(system_data)} system")

    # for a, b in zip(human_data, system_data):
    #   print (a['query_title'])
    #   print (b['query_title'])
    #   sys.exit()
    print (human_data[0]['query_title'])
    print (system_data[0]['query_title'])

    assert len(human_data) == len(system_data), (
        "Files must have same number of entries"
    )

    spearman_scores = []
    per_query = []

    for i, (h, s) in enumerate(zip(human_data, system_data)):
        sp = evaluate_query(
            h["candidates"],
            s["candidates"]
        )

        spearman_scores.append(sp)

        row = {
            "idx": i,
            "spearman": None if np.isnan(sp) else float(sp)
        }

        per_query.append(row)

    # optionally save per-query results
    if output_file:
        save_jsonl(output_file, per_query)

    results = {
        "mean_spearman": float(np.nanmean(spearman_scores)),
        "per_query": per_query
    }

    return results


# ---------- run ----------
if __name__ == "__main__":
    results = evaluate_jsonl(
        "human-cleaned.jsonl",
        "subset-ranked-cleaned.jsonl",
        output_file="corr_results.jsonl"
    )

    print("\n=== Results ===")

    print(
        f"Overall Mean Spearman: "
        f"{results['mean_spearman']:.4f}"
    )

    print("\n=== Per Query ===")

    for q in results["per_query"]:
        score = q["spearman"]

        if score is None:
            score_str = "nan"
        else:
            score_str = f"{score:.4f}"

        print(
            f"Query {q['idx']:02d} | "
            f"Spearman: {score_str}"
        )
