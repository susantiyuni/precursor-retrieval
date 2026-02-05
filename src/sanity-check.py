import json
import numpy as np

candidate_counts = []
msc_overlap_rates = []
kw_overlap_rates = []

# Your normalization functions
def norm_msc(msc_url):
    return msc_url.rstrip("/").split("/")[-1]

def msc_prefixes(mscs):
    pref = set()
    for m in mscs:
        c = norm_msc(m)
        if len(c) >= 2:
            pref.add(c[:2])
        if len(c) >= 3:
            pref.add(c[:3])
    return pref

def norm_kw(kw_url):
    return kw_url.rstrip("/").split("/")[-1].lower()

def norm_kw_set(keywords):
    return set(norm_kw(k) for k in keywords or [])

with open("candidate-pool.jsonl") as f:
    for line in f:
        r = json.loads(line)
        q = r["query_paper"]

        # Query MSC prefixes and normalized keywords
        q_msc = msc_prefixes(q.get("mscs", []) or q.get("msc_codes", []))
        q_kw  = norm_kw_set(q.get("keywords", []))

        candidates = r["candidates"]
        candidate_counts.append(len(candidates))

        # Count overlaps
        msc_hits = 0
        kw_hits = 0

        for c in candidates:
            c_msc = msc_prefixes(c.get("msc_codes", []))
            c_kw  = norm_kw_set(c.get("keywords", []))

            if q_msc & c_msc:
                msc_hits += 1
            if q_kw & c_kw:
                kw_hits += 1

        # Fraction of candidates overlapping for this query
        if candidates:
            msc_overlap_rates.append(msc_hits / len(candidates))
            kw_overlap_rates.append(kw_hits / len(candidates))
        else:
            msc_overlap_rates.append(0)
            kw_overlap_rates.append(0)

# Overall averages across all queries
print("Avg candidates per query:", np.mean(candidate_counts))
print("Avg MSC overlap rate:", np.mean(msc_overlap_rates))
print("Avg keyword overlap rate:", np.mean(kw_overlap_rates))
