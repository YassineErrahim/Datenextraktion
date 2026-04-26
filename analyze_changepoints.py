from __future__ import annotations
import json
import pathlib
from collections import defaultdict

CHANGEPOINT_SET = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CHANGEPOINT_SET"

def main():
    total = yes = no = c100 = c67 = c33 = c0 = 0
    total_prs = prs_clean = prs_change = 0
    split_67_dissenter: dict[str, int] = defaultdict(int)
    split_33_dissenter: dict[str, int] = defaultdict(int)

    for f in pathlib.Path(CHANGEPOINT_SET).rglob("*.json"):
        pr = json.load(open(f, encoding="utf-8"))
        cps = pr.get("change_points", [])
        if not cps:
            continue

        total_prs += 1
        consensuses = [cp["llm_verification"]["consensus"] for cp in cps]
        if any(c == "YES" for c in consensuses):
            prs_change += 1
        else:
            prs_clean += 1

        for cp in cps:
            total += 1
            c = cp["llm_verification"]["confidence"]
            consensus = cp["llm_verification"]["consensus"]
            votes = cp["llm_verification"]["votes"]
            models = cp["llm_verification"]["models"]

            if consensus == "YES":
                yes += 1
            else:
                no += 1

            if c == 1.0:
                c100 += 1
            elif c >= 0.66:
                c67 += 1
                for m, v in zip(models, votes):
                    if v == "NO":
                        split_67_dissenter[m] += 1
            elif c >= 0.32:
                c33 += 1
                for m, v in zip(models, votes):
                    if v == "YES":
                        split_33_dissenter[m] += 1
            else:
                c0 += 1

    avg_cps = total / total_prs if total_prs else 0

    print("PR-EBENE:")
    print(f"PRs processed: {total_prs}")
    print(f"Avg CPs per PR: {avg_cps:.2f}")
    print(f"Clean PRs: {prs_clean}")
    print(f"Change PRs: {prs_change}")

    print("\n")

    print("KOMMENTAR-EBENE")
    print(f"Total Kommentare: {total}")
    print(f"Consensus YES: {yes}")
    print(f"Consensus NO: {no}")
    print(f"\n")
    print(f"Confidence 1.0  (3/3 YES): {c100}")
    print(f"Confidence 0.67 (2/3 YES): {c67}")
    print(f"Confidence 0.33 (1/3 YES): {c33}")
    print(f"Confidence 0.0  (0/3 YES): {c0}")

    print("\n")

    print("SPLIT-VOTE ANALYSE")
    print(f"Split 0.67 (2/3 YES): Welches LLM hat NO gesagt?")
    for m, count in sorted(split_67_dissenter.items(), key=lambda x: -x[1]):
        print(f"  {m}: {count}")

    print()
    print(f"Split 0.33 (1/3 YES): Welches LLM hat YES gesagt?")
    for m, count in sorted(split_33_dissenter.items(), key=lambda x: -x[1]):
        print(f"  {m}: {count}")


if __name__ == "__main__":
    main()