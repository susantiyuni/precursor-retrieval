import json
from pathlib import Path
import numpy as np

import json
from pathlib import Path
from statistics import mean, median

# ---------------- CONFIG ----------------
jsonl_path = Path("candidate-pool-latest.jsonl")  # your file

msc_areas = 6  # number of MSC areas you are considering

# ---------------- LOAD DATA ----------------
data = []
with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# ---------------- INITIALIZE STATISTICS ----------------
num_queries = len(data)
candidate_pool_sizes = []
query_reference_counts = []
publication_gaps = []
cited_counts = []
uncited_counts = []
keywords_per_candidate = []
keywords_per_query = []
msc_counts_query = []
msc_counts_candidate = []
title_lengths_query = []
title_lengths_candidate = []
candidates_missing_review = 0
candidates_missing_title = 0

# ---------------- PROCESS DATA ----------------
for entry in data:
    query = entry["query_paper"]
    candidates = entry["candidates"]

    # Query statistics
    query_reference_counts.append(len(query.get("references", [])))
    msc_counts_query.append(len(query.get("mscs", [])))
    title_lengths_query.append(len(query.get("title", "").split()))
    keywords_per_query.append(len(query.get("keywords", [])))

    # Candidate pool statistics
    pool_size = len(candidates)
    candidate_pool_sizes.append(pool_size)

    for c in candidates:
        # Publication gap
        gap = query["year"] - c["year"]
        publication_gaps.append(gap)

        # Cited / uncited
        if c.get("is_cited", 0) == 1:
            cited_counts.append(1)
        else:
            uncited_counts.append(1)

        # Keywords & MSC
        keywords_per_candidate.append(len(c.get("keywords", [])))
        msc_counts_candidate.append(len(c.get("msc_codes", [])))

        # Title length
        if c.get("title"):
            title_lengths_candidate.append(len(c["title"].split()))
        else:
            candidates_missing_title += 1

        # Missing review
        if not c.get("review"):
            candidates_missing_review += 1

# ---------------- CALCULATE METRICS ----------------
def safe_mean(lst):
    return mean(lst) if lst else 0

def safe_median(lst):
    return median(lst) if lst else 0

stats = {
    "# Query papers": num_queries,
    "# MSC areas": msc_areas,
    "Avg. references per query": round(safe_mean(query_reference_counts), 2),
    # "Median references per query": safe_median(query_reference_counts),
    "Avg. candidate pool size": round(safe_mean(candidate_pool_sizes), 2),
    # "Median candidate pool size": safe_median(candidate_pool_sizes),
    "Avg. publication gap": round(safe_mean(publication_gaps), 2),
    # "Median publication gap": safe_median(publication_gaps),
    "Min / Max publication gap": (min(publication_gaps), max(publication_gaps)) if publication_gaps else (0,0),
    "# cited candidates": len(cited_counts),
    "# uncited candidates": len(uncited_counts),
    "Fraction cited": round(len(cited_counts) / (len(cited_counts) + len(uncited_counts)), 3) if (len(cited_counts) + len(uncited_counts)) > 0 else 0,
    "Fraction uncited": round(len(uncited_counts) / (len(cited_counts) + len(uncited_counts)), 3) if (len(cited_counts) + len(uncited_counts)) > 0 else 0,
    "Avg. keywords per candidate": round(safe_mean(keywords_per_candidate), 2),
    "Avg. keywords per query": round(safe_mean(keywords_per_query), 2),
    "Avg. MSC codes per query": round(safe_mean(msc_counts_query), 2),
    "Avg. MSC codes per candidate": round(safe_mean(msc_counts_candidate), 2),
    "Avg. title length query": round(safe_mean(title_lengths_query), 2),
    "Avg. title length candidate": round(safe_mean(title_lengths_candidate), 2),
    "# candidates missing review": candidates_missing_review,
    "# candidates missing title": candidates_missing_title,
}

# ---------------- PRINT TABLE ----------------
print("Dataset Statistics")
print("="*60)
for k, v in stats.items():
    print(f"{k:30}: {v}")
