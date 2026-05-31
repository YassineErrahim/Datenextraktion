import json
import glob
import os
from collections import Counter

base_dir = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CHANGEPOINT_SET"

cp_categories = Counter()
pr_labels = Counter()
total_prs_with_yes = 0
total_yes_cps = 0

for fpath in glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True):
    with open(fpath) as f:
        pr = json.load(f)

    yes_cps = [
        cp for cp in pr.get("change_points", [])
        if cp.get("llm_verification", {}).get("consensus") == "YES"
    ]
    if not yes_cps:
        continue

    total_prs_with_yes += 1
    total_yes_cps += len(yes_cps)  # count directly from source, not from classification

    clf = pr.get("classification")
    if not clf:
        continue

    pr_labels[clf.get("gt_quality_label", "Unknown")] += 1
    for point in clf.get("per_point_classification", []):
        cp_categories[point.get("category", "Unknown")] += 1

print(f"PRs with at least one YES consensus change point: {total_prs_with_yes}")
print(f"Total YES change points: {total_yes_cps}")

print("\n=== Changepoint-Ebene (nur YES-Punkte) ===")
for cat, count in cp_categories.most_common():
    print(f"  {cat}: {count}")
print(f"  Gesamt (klassifiziert): {sum(cp_categories.values())}")
print(f"  Gesamt (YES gesamt):    {total_yes_cps}")
print(f"  Fehlend:                {total_yes_cps - sum(cp_categories.values())}")

print("\n=== PR-Ebene (gt_quality_label) ===")
for label, count in pr_labels.most_common():
    print(f"  {label}: {count}")
print(f"  Gesamt: {sum(pr_labels.values())}")