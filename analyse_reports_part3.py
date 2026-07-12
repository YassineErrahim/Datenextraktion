import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

REPORTS_DIR = Path("/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CODE_REVIEW_REPORTS")
CP_DIR      = Path("/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CHANGEPOINT_SET")

COMMERCIAL_MODELS = {"claude-sonnet-4-6", "gpt-5.4", "deepseek-v4-pro"}
GPU_COST_PER_SECOND = {
    "google_gemma-4-31b-it": 2.98 / 3600,
    "mistralai_Codestral-22B-v0.1": 1.49 / 3600,
    "ibnzterrell_Meta-Llama-3.3-70B-Instruct-AWQ-INT4": 2.98 / 3600,
    "neuralmagic_DeepSeek-R1-Distill-Llama-70B-quantized.w8a8": 2.98 / 3600,
    "Qwen_Qwen2.5-72B-Instruct-AWQ": 2.98 / 3600,
}

gt = {}
for f in CP_DIR.glob("*/*.json"):
    try:
        d = json.load(open(f))
        cl = d.get("classification", {})
        gt[f.stem] = {
            "gt_quality_label": cl.get("gt_quality_label"),
            "gt_substantive_count": cl.get("category_counts", {}).get("Substantive", 0),
        }
    except: pass

model_files = {}
for model_dir in REPORTS_DIR.iterdir():
    if not model_dir.is_dir(): continue
    model_files[model_dir.name] = set(f.stem for f in model_dir.glob("*/*.json"))

common_files = set.intersection(*model_files.values()) if model_files else set()
print(f"Common PRs: {len(common_files)}\n")

# --- overview per model ---
print("=" * 60)
print("OVERVIEW PER MODEL")
print("=" * 60)

for model_dir in sorted(REPORTS_DIR.iterdir()):
    if not model_dir.is_dir(): continue
    model = model_dir.name
    is_commercial = model in COMMERCIAL_MODELS

    approve = request_changes = 0
    findings, severities, latencies, tokens_list, costs, tool_counts = [], [], [], [], [], []
    tool_freq = defaultdict(int)

    for json_file in model_dir.glob("*/*.json"):
        if json_file.stem not in common_files: continue
        try:
            d = json.load(open(json_file))
            if "error" in d: continue
            meta = d.get("_meta", {})
            tok = meta.get("tokens", {})
            f_list = d.get("findings", [])
            latency = meta.get("latency_s") or 0

            if d.get("verdict") == "approve": approve += 1
            else: request_changes += 1

            findings.append(len(f_list))
            if f_list: severities.append(mean(x["severity_score"] for x in f_list))
            latencies.append(latency)
            tokens_list.append(tok.get("total_tokens") or 0)
            tool_counts.append(meta.get("tool_count") or 0)
            for t in meta.get("tool_calls", []): tool_freq[t["tool"]] += 1

            cost = tok.get("estimated_cost_usd") if is_commercial else latency * GPU_COST_PER_SECOND.get(model, 0)
            if cost: costs.append(cost)
        except: pass

    total = approve + request_changes
    top3 = sorted(tool_freq.items(), key=lambda x: -x[1])[:3]
    avg_sev = f"{mean(severities):.2f}" if severities else "-"
    avg_cost = f"${mean(costs):.5f}" if costs else "-"

    print(f"\nModel          : {model}")
    print(f"  Type         : {'commercial' if is_commercial else 'open-source'}")
    print(f"  PRs reviewed : {total}")
    print(f"  Approve      : {approve} ({approve/total*100:.1f}%)")
    print(f"  Req. changes : {request_changes} ({request_changes/total*100:.1f}%)")
    print(f"  Avg findings : {mean(findings):.2f}")
    print(f"  Avg severity : {avg_sev}")
    print(f"  Avg latency  : {mean(latencies):.1f}s")
    print(f"  Avg tokens   : {int(mean(tokens_list))}")
    print(f"  Avg cost     : {avg_cost}")
    print(f"  Avg tools    : {mean(tool_counts):.1f}")
    print(f"  Top tools    : {', '.join(f'{t}({c})' for t,c in top3) if top3 else '-'}")

# --- breakdown by gt_quality_label ---
for label in ["Substantive", "Shallow", "Noisy", "Documentation"]:
    print(f"\n{'=' * 60}")
    print(f"GT LABEL: {label}")
    print("=" * 60)

    for model_dir in sorted(REPORTS_DIR.iterdir()):
        if not model_dir.is_dir(): continue
        model = model_dir.name
        is_commercial = model in COMMERCIAL_MODELS

        approve = request_changes = 0
        findings, severities, latencies, tokens_list, costs, tool_counts = [], [], [], [], [], []
        tool_freq = defaultdict(int)

        for json_file in model_dir.glob("*/*.json"):
            if json_file.stem not in common_files: continue
            if gt.get(json_file.stem, {}).get("gt_quality_label") != label: continue
            try:
                d = json.load(open(json_file))
                if "error" in d: continue
                meta = d.get("_meta", {})
                tok = meta.get("tokens", {})
                f_list = d.get("findings", [])
                latency = meta.get("latency_s") or 0

                if d.get("verdict") == "approve": approve += 1
                else: request_changes += 1

                findings.append(len(f_list))
                if f_list: severities.append(mean(x["severity_score"] for x in f_list))
                latencies.append(latency)
                tokens_list.append(tok.get("total_tokens") or 0)
                tool_counts.append(meta.get("tool_count") or 0)
                for t in meta.get("tool_calls", []): tool_freq[t["tool"]] += 1

                cost = tok.get("estimated_cost_usd") if is_commercial else latency * GPU_COST_PER_SECOND.get(model, 0)
                if cost: costs.append(cost)
            except: pass

        total = approve + request_changes
        if total == 0: continue
        top3 = sorted(tool_freq.items(), key=lambda x: -x[1])[:3]
        avg_sev = f"{mean(severities):.2f}" if severities else "-"
        avg_cost = f"${mean(costs):.5f}" if costs else "-"

        print(f"\n  Model          : {model}")
        print(f"    Type         : {'commercial' if is_commercial else 'open-source'}")
        print(f"    PRs          : {total}")
        print(f"    Approve      : {approve} ({approve/total*100:.1f}%)")
        print(f"    Req. changes : {request_changes} ({request_changes/total*100:.1f}%)")
        print(f"    Avg findings : {mean(findings):.2f}")
        print(f"    Avg severity : {avg_sev}")
        print(f"    Avg latency  : {mean(latencies):.1f}s")
        print(f"    Avg tokens   : {int(mean(tokens_list))}")
        print(f"    Avg cost     : {avg_cost}")
        print(f"    Avg tools    : {mean(tool_counts):.1f}")
        print(f"    Top tools    : {', '.join(f'{t}({c})' for t,c in top3) if top3 else '-'}")