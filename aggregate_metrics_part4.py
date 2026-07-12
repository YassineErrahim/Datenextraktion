import json
import os
from collections import defaultdict

JUDGE_BASE = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CODE_REVIEW_JUDGE"
CHANGEPOINT_BASE = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CHANGEPOINT_SET"

GPU_COST_PER_SECOND = {
    "google_gemma-4-31b-it": 2.98 / 3600,
    "mistralai_Codestral-22B-v0.1": 1.49 / 3600,
    "ibnzterrell_Meta-Llama-3.3-70B-Instruct-AWQ-INT4": 2.98 / 3600,
    "neuralmagic_DeepSeek-R1-Distill-Llama-70B-quantized.w8a8": 2.98 / 3600,
    "Qwen_Qwen2.5-72B-Instruct-AWQ": 2.98 / 3600,
}

METRICS = ["tp", "fp", "fn", "precision", "recall", "f1", "time_to_feedback_s", "reduction_pct", "cost_usd"]


def get_cost(model, data):
    if model in GPU_COST_PER_SECOND:
        return GPU_COST_PER_SECOND[model] * data["time_to_feedback_s"]
    return data["cost_usd"]


def load_judge_data():
    result = {}
    for model in os.listdir(JUDGE_BASE):
        model_path = os.path.join(JUDGE_BASE, model)
        if not os.path.isdir(model_path):
            continue
        result[model] = {}
        for category in os.listdir(model_path):
            cat_path = os.path.join(model_path, category)
            if not os.path.isdir(cat_path):
                continue
            for fname in os.listdir(cat_path):
                if fname.endswith(".json"):
                    with open(os.path.join(cat_path, fname)) as f:
                        data = json.load(f)
                    result[model][fname] = data
    return result


def load_quality_labels():
    result = {}
    for category in os.listdir(CHANGEPOINT_BASE):
        cat_path = os.path.join(CHANGEPOINT_BASE, category)
        if not os.path.isdir(cat_path):
            continue
        for fname in os.listdir(cat_path):
            if fname.endswith(".json"):
                with open(os.path.join(cat_path, fname)) as f:
                    data = json.load(f)
                label = data.get("classification", {}).get("gt_quality_label")
                if label:
                    result[fname] = label
    return result


def average_metrics(records):
    n = len(records)
    if n == 0:
        return {}
    result = {"n_prs": n}
    for m in METRICS:
        values = [r[m] for r in records if r.get(m) is not None]
        result[m] = sum(values) / len(values) if values else None
    return result


def extract_record(model, fname, data):
    return {
        "tp": data.get("tp"),
        "fp": data.get("fp"),
        "fn": data.get("fn"),
        "precision": data.get("precision"),
        "recall": data.get("recall"),
        "f1": data.get("f1"),
        "time_to_feedback_s": data.get("time_to_feedback_s"),
        "reduction_pct": data.get("reduction_pct"),
        "cost_usd": get_cost(model, data),
    }


def main():
    judge_data = load_judge_data()
    quality_labels = load_quality_labels()

    models = list(judge_data.keys())
    print(f"Models found: {models}\n")

    common_fnames = set(judge_data[models[0]].keys())
    for model in models[1:]:
        common_fnames &= set(judge_data[model].keys())
    print(f"Common PRs across all models: {len(common_fnames)}\n")

    print("=" * 60)
    print("LEVEL — PR LEVEL (all common PRs)")
    print("=" * 60)
    for model in models:
        records = [extract_record(model, fname, judge_data[model][fname]) for fname in common_fnames]
        avg = average_metrics(records)
        print(f"\nModel: {model}")
        for k, v in avg.items():
            print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")

    print("\n" + "=" * 60)
    print("LEVEL — QUALITY LABEL LEVEL")
    print("=" * 60)

    all_labels = ["Substantive", "Shallow", "Noisy", "Documentation"]

    for label in all_labels:
        label_fnames = {fname for fname in quality_labels if quality_labels[fname] == label}
        common_label_fnames = label_fnames.copy()
        for model in models:
            common_label_fnames &= set(judge_data[model].keys())

        print(f"\n--- Label: {label} ({len(common_label_fnames)} common PRs) ---")
        for model in models:
            records = [extract_record(model, fname, judge_data[model][fname]) for fname in common_label_fnames]
            avg = average_metrics(records)
            print(f"\n  Model: {model}")
            for k, v in avg.items():
                print(f"    {k}: {round(v, 4) if isinstance(v, float) else v}")


if __name__ == "__main__":
    main()
