import json
from pathlib import Path
from collections import defaultdict

base = Path("/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CHANGEPOINT_SET")

total_yes, total_not_yes, total_all_yes, total_mixed = 0, 0, 0, 0
model_no_counts = defaultdict(int)
prs_with_at_least_one_yes = 0
prs_with_no_yes = 0

for json_file in base.glob("*/*.json"):
    change_points = json.load(open(json_file)).get("change_points", [])
    pr_has_yes = False
    for cp in change_points:
        lv = cp["llm_verification"]
        if lv["consensus"] == "YES":
            pr_has_yes = True
            total_yes += 1
            if all(v == "YES" for v in lv["votes"]):
                total_all_yes += 1
            else:
                total_mixed += 1
                for vote, model in zip(lv["votes"], lv["models"]):
                    if vote == "NO":
                        model_no_counts[model] += 1
        else:
            total_not_yes += 1

    if pr_has_yes:
        prs_with_at_least_one_yes += 1
    else:
        prs_with_no_yes += 1

total_no = sum(model_no_counts.values())

print(f"PRs with at least one confirmed CP: {prs_with_at_least_one_yes} | PRs with no confirmed CP: {prs_with_no_yes}")
print(f"Consensus YES: {total_yes} | NOT YES: {total_not_yes}")
print(f"  All YES: {total_all_yes} | Mixed (has NO): {total_mixed}")
if total_no:
    print("  NO votes by model (in mixed group):")
    for model, count in model_no_counts.items():
        print(f"    {model}: {count/total_no*100:.1f}%")