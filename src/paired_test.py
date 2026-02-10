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

A = np.array([0.29907196, 0.73702044, 0.61618082, 0.53947103, 0.67387297, 0.66281143,
              0.54413591, 0.68895332, 0.62978117, 0.67217976, 0.77840458, 0.63651409,
              0.65844544, 0.77883119, 0.69796428, 0.65699529, 0.75043723, 0.78377278,
              0.67191863, 0.74389943, 0.77184252, 0.56273971, 0.71713746, 0.72104151,
              0.66292737, 0.71176538, 0.51481396, 0.57552262, 0.30690136, 0.59068256])

B = np.array([0.60355458, 0.79336426, 0.7892031,  0.76680153, 0.7807666,  0.73765851,
              0.66522871, 0.85432541, 0.70672174, 0.72140399, 0.83178142, 0.78544977,
              0.74877359, 0.81962408, 0.78125533, 0.77134351, 0.78956168, 0.84397221,
              0.72599453, 0.82141918, 0.84782643, 0.67095754, 0.79755416, 0.69900802,
              0.81368179, 0.81882218, 0.63892216, 0.80303118, 0.74800216, 0.71337237])

our_file = "output/run_tmgnrx_citation_per_query_metrics.jsonl"
baseline_file  = "output/run_bm25_per_query_metrics.jsonl"
baseline_file  = "output/run_citation_per_query_metrics.jsonl"

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
