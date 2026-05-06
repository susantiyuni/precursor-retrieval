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

our_file = "run-zsv3-alltemp/run_tmgnrxv3_all_gamma_per_query_metrics.jsonl"
# baseline_file  = "run-zs-baseline-01/run_bm25_per_query_metrics.jsonl" ##lowest baseline for all-pool
baseline_file  = "run-zs-baseline-01/run_citation_per_query_metrics.jsonl" ##lowest baseline for uncited-only pool

baseline_data = read_jsonl(baseline_file)
our_data  = read_jsonl(our_file)

metrics = [
    "nDCG@10", "nDCG@10_uncited",
    "Recall@10", "Recall@10_uncited",
    "Recall@10S", "Recall@10S_uncited",
    "Precision@10", "Precision@10_uncited",
    "Precision@10S", "Precision@10S_uncited",
    "MAP", "MRR"
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
    # # 3. Cohen's d for paired samples
    # diff = B - A
    # cohen_d = np.mean(diff) / np.std(diff, ddof=1)
    # print("\nCohen's d:", cohen_d)
    # if abs(cohen_d) >= 0.8:
    #     print("Effect size: Large")
    # elif abs(cohen_d) >= 0.5:
    #     print("Effect size: Medium")
    # elif abs(cohen_d) >= 0.2:
    #     print("Effect size: Small")
    # else:
    #     print("Effect size: Negligible")

    # # 4. 95% CI for mean difference
    # ci_low, ci_high = stats.t.interval(
    #     0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
    # print("\n95% CI for mean difference:", (ci_low, ci_high))
    print (f"\n")
