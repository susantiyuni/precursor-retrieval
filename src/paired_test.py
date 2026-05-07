import json
import numpy as np
from scipy import stats

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def extract_metric(data, metric):
    """
    Extract metric values, skipping None/null entries.
    """
    vals = []
    for d in data:
        if metric in d and d[metric] is not None:
            vals.append(d[metric])
    return np.array(vals)

# our_file = "run-zsv3-alltemp/run_tmgnrxv3_all_gamma_per_query_metrics.jsonl"
our_file = "run-zsv3-alltemp/run_tmgnrxv3_explicit_gamma_per_query_metrics.jsonl"

# baseline_file  = "run-zs-baseline-01/run_bm25_per_query_metrics.jsonl" ##weakest all pool
baseline_file  = "run-zs-baseline-01/run_citation_per_query_metrics.jsonl" ##weakest uncited_only pool
# baseline_file  = "run-zs-baseline-01/run_ppr_per_query_metrics.jsonl" ##strongest all pool
# baseline_file  = "run-zs-baseline-01/run_dualenc_per_query_metrics.jsonl" ##strongest uncited_only pool

baseline_data = read_jsonl(baseline_file)
our_data  = read_jsonl(our_file)

# metrics = [
#     "nDCG", "nDCG_uncited",
#     "Recall", "Recall_uncited",
#     "RecallS", "RecallS_uncited",
#     "Precision", "Precision_uncited",
#     "PrecisionS", "PrecisionS_uncited",
#     "MAP", "MAP_uncited"
# ]

## all candidates
metrics = [
    "nDCG",
    "Recall", 
    "RecallS", 
    "Precision",
    "PrecisionS", 
    "MAP"
]

## uncited_only
metrics = [
    "nDCG_uncited",
    "Recall_uncited",
    "RecallS_uncited",
    "Precision_uncited",
    "PrecisionS_uncited",
    "MAP_uncited"
]

alpha = 0.05  # significance level

for metric in metrics:
    A = extract_metric(baseline_data, metric)
    B = extract_metric(our_data, metric)
    print (f"{metric=}")
    print (f"A:{baseline_file} vs. B:{our_file}")

    mean_A = np.mean(A)
    mean_B = np.mean(B)
    diff_mean = mean_B - mean_A

    print("Mean A:", mean_A)
    print("Mean B:", mean_B)
    print("Difference (B - A):", diff_mean)

    # Must be paired and same length
    if len(A) != len(B):
        print(f"[SKIP] {metric} - length mismatch ({len(A)} vs {len(B)})")
        continue

    # 2. Paired t-test
    t_stat, p_val = stats.ttest_rel(B, A)
    print("Paired t-test:")
    print("t =", t_stat)
    print("p =", p_val)
    if p_val < alpha:
        print("RESULT: Statistically significant (p < 0.05)")
    else:
        print("RESULT: Not statistically significant (p >= 0.05)")
    
    print (f"\n")
