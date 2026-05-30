import json
import os
from datetime import datetime, timezone
from openai import OpenAI
import re
import requests

OPENAI_API_KEY = ""
GITHUB_TOKEN = ""

MODELS = [
    "claude-sonnet-4-6",
    "gpt-5.4",
    "deepseek-v4-pro",
    "google_gemma-4-31b-it",
    "neuralmagic_DeepSeek-R1-Distill-Llama-70B-quantized.w8a8",
    "Qwen_Qwen2.5-72B-Instruct-AWQ",
    "mistralai_Codestral-22B-v0.1",
    "ibnzterrell_Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
]

BASE_REPORTS     = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CODE_REVIEW_REPORTS"
BASE_CHANGEPOINTS = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CHANGEPOINT_SET"
BASE_OUTPUT      = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CODE_REVIEW_JUDGE"

MAX_FILES = 0 # 0 for all files to be processed

JUDGE_MODEL = "o3-mini"

SYSTEM_PROMPT = """You are an objective code review evaluator following the SWR-Bench evaluation methodology.

Your task is a fact-based matching task: determine whether each ground truth change-point
was identified by at least one agent finding.

Rules:
- A ground truth change-point is "hit" if ANY agent finding covers the same underlying issue,
  even if described differently or with different wording.
- Do NOT require exact wording match — semantic equivalence is sufficient.
- Return ONLY valid JSON, no explanation, no markdown fences.
"""


def build_judge_prompt(gt_change_points: list, agent_findings: list) -> str:
    gt_block = ""
    for i, cp in enumerate(gt_change_points):
        rc = cp["reviewer_comment"]
        gt_block += f"""
GT[{i}]
  id: {cp["id"]}
  file: {rc["path"].split("/")[-1]}
  reviewer_comment: {rc["body"].strip()}
  diff_hunk: {rc.get("diff_hunk", "")[:300]}
"""

    findings_block = ""
    for i, finding in enumerate(agent_findings):
        findings_block += f"""
Finding[{i}]
  description: {finding["description"]}
  severity_score: {finding.get("severity_score", "N/A")}
"""

    return f"""You are given:

=== GROUND TRUTH CHANGE-POINTS (real issues identified by human reviewers) ===
{gt_block}

=== AGENT FINDINGS (issues identified by the AI code review agent) ===
{findings_block}

Task:
For each ground truth change-point, determine if it was "hit" by at least one agent finding.
Also determine which finding indices (0-based) matched each ground truth change-point.
Then determine which agent findings are false positives (hit no ground truth change-point).

Return ONLY this JSON structure:
{{
  "hits": [
    {{
      "gt_id": "<id of the ground truth change-point>",
      "hit": true or false,
      "matched_finding_indices": [<list of 0-based finding indices that hit this GT, empty if none>],
      "reasoning": "<one sentence explaining why hit or not>"
    }}
  ],
  "false_positive_finding_indices": [<list of 0-based finding indices that hit no GT change-point>]
}}
"""

def run_judge(gt_change_points: list, agent_findings: list, client: OpenAI) -> dict:
    prompt = build_judge_prompt(gt_change_points, agent_findings)

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        reasoning_effort="medium",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def compute_metrics(judge_result: dict) -> dict:
    hits = judge_result["hits"]
    fp_indices = judge_result.get("false_positive_finding_indices", [])

    tp = sum(1 for h in hits if h["hit"])
    fn = sum(1 for h in hits if not h["hit"])
    fp = len(fp_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def fetch_pr_times(pr_url: str) -> str | None:
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
    if not match:
        return None, None
    owner, repo, pr_number = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    headers = {"Accept": "application/vnd.github+json"}
    headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("merged_at")
    except Exception as e:
        print(f"\n      [WARN] GitHub API error for {pr_url}: {e}")
        return None

from datetime import datetime, timezone

def calculate_human_review_time(pr_data: dict, merged_at: str) -> dict:
    fmt = "%Y-%m-%dT%H:%M:%SZ"

    commit_timestamps = [
        datetime.strptime(node['commit']['committedDate'], fmt).replace(tzinfo=timezone.utc)
        for node in pr_data['commits']['nodes']
    ]

    comment_timestamps = [
        datetime.strptime(comment['createdAt'], fmt).replace(tzinfo=timezone.utc)
        for thread in pr_data['reviewThreads']['nodes']
        for comment in thread['comments']['nodes']
    ]

    dt_merged = datetime.strptime(merged_at, fmt).replace(tzinfo=timezone.utc)

    events = sorted(
        [('commit', dt) for dt in commit_timestamps] +
        [('comment', dt) for dt in comment_timestamps],
        key=lambda x: x[1]
    )

    total_seconds = 0.0
    i = 0

    while i < len(events):
        if events[i][0] == 'commit':
            last_commit = events[i][1]
            while i < len(events) and events[i][0] == 'commit':
                last_commit = events[i][1]
                i += 1
            if i < len(events) and events[i][0] == 'comment':
                last_comment = events[i][1]
                while i < len(events) and events[i][0] == 'comment':
                    last_comment = events[i][1]
                    i += 1

                total_seconds += (last_comment - last_commit).total_seconds()
        else:
            i += 1

    last_event = events[-1][1] if events else dt_merged
    total_seconds += (dt_merged - last_event).total_seconds()

    return {
        'human_review_time_s': total_seconds,
        'human_review_time_h': total_seconds / 3600
    }

def compute_time_metrics(cp_data: dict, meta: dict, f1: float) -> dict:
    merged_at = fetch_pr_times(cp_data.get("url", ""))
    result = calculate_human_review_time(cp_data, merged_at)
    human_review_time_s = result['human_review_time_s']
    human_review_time_h = result['human_review_time_h']
    agent_time_s  = meta.get("latency_s")
    theoretical_saved_s = round((human_review_time_s - agent_time_s) * f1, 2)
    reduction_pct = round((1 - agent_time_s / human_review_time_s) * f1 * 100, 2)

    return {
        "time_to_feedback_s": agent_time_s,
        "human_review_time_s": human_review_time_s,
        "human_review_time_h": human_review_time_h,
        "theoretical_saved_s": theoretical_saved_s,
        "theoretical_saved_h": round(theoretical_saved_s / 3600, 2),
        "reduction_pct": reduction_pct,
    }


def evaluate_file(report_path: str, cp_path: str, out_path: str, model: str, category: str, client: OpenAI):
    with open(report_path) as f:
        report = json.load(f)
    with open(cp_path) as f:
        cp_data = json.load(f)

    gt_change_points = [
        cp for cp in cp_data.get("change_points", [])
        if cp["llm_verification"]["consensus"] == "YES"
    ]

    agent_findings = report.get("findings", [])
    meta = report.get("_meta", {})
    pr_number = cp_data.get("number", "unknown")

    url_parts = cp_data.get("url", "").rstrip("/").split("/")
    repo_name = url_parts[-3] if len(url_parts) >= 3 else "unknown"

    filename = os.path.basename(report_path)
    print(f"    [{model}] {category}/{filename} | GT={len(gt_change_points)} findings={len(agent_findings)}", end="", flush=True)

    judge_result = run_judge(gt_change_points, agent_findings, client)
    metrics = compute_metrics(judge_result)
    time_metrics = compute_time_metrics(cp_data, meta, metrics["f1"])

    output = {
        "pr_number": pr_number,
        "repo": repo_name,
        "model": model,
        "category": category,
        "pr_url": cp_data.get("url", ""),
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "time_to_feedback_s": time_metrics["time_to_feedback_s"],
        "human_review_time_s":time_metrics["human_review_time_s"],
        "human_review_time_h":time_metrics["human_review_time_h"],
        "theoretical_saved_s":time_metrics["theoretical_saved_s"],
        "theoretical_saved_h":time_metrics["theoretical_saved_h"],
        "reduction_pct": time_metrics["reduction_pct"],
        "cost_usd": meta.get("tokens", {}).get("estimated_cost_usd"),
        "total_tokens": meta.get("tokens", {}).get("total_tokens"),
        "judge_hits": judge_result["hits"],
        "false_positive_finding_indices": judge_result.get("false_positive_finding_indices", []),
        "gt_change_points_total": len(cp_data.get("change_points", [])),
        "gt_change_points_yes": len(gt_change_points),
        "agent_findings_count": len(agent_findings),
        "_meta": meta,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": JUDGE_MODEL,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f" → P={metrics['precision']:.2%} R={metrics['recall']:.2%} F1={metrics['f1']:.2%} | saved {time_metrics['reduction_pct']}% | ${meta.get('tokens', {}).get('estimated_cost_usd', '?')}")
    return output


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    total = 0
    skipped = 0
    errors = 0

    for model in MODELS:
        model_reports_base = os.path.join(BASE_REPORTS, model)

        if not os.path.isdir(model_reports_base):
            print(f"[WARN] Model folder not found, skipping: {model_reports_base}")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        for category in sorted(os.listdir(model_reports_base)):
            category_path = os.path.join(model_reports_base, category)
            if not os.path.isdir(category_path):
                continue
            print(f"\n  Category: {category}")

            files_processed = 0
            for filename in sorted(os.listdir(category_path)):
                if not filename.endswith(".json"):
                    continue

                if MAX_FILES != 0 and files_processed >= MAX_FILES:
                    print(f"    [{model}] {category} → MAX_FILES={MAX_FILES} reached, stopping.")
                    break

                report_path = os.path.join(category_path, filename)
                cp_path = os.path.join(BASE_CHANGEPOINTS, category, filename)
                out_path = os.path.join(BASE_OUTPUT, model, category, filename)

                if os.path.exists(out_path):
                    print(f"    [{model}] {category}/{filename} → already exists, skipping")
                    skipped += 1
                    continue
                
                if not os.path.exists(cp_path):
                    print(f"    [{model}] {category}/{filename} → WARN: no matching changepoint file at {cp_path}")
                    skipped += 1
                    continue

                try:
                    result = evaluate_file(report_path, cp_path, out_path, model, category, client)
                    if result:
                        total += 1
                        files_processed += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"    [{model}] {category}/{filename} → ERROR: {e}")
                    errors += 1
                    files_processed += 1

    print(f"\n{'='*60}")
    print(f"Done. Evaluated: {total} | Skipped: {skipped} | Errors: {errors}")
    print(f"Results saved under: {BASE_OUTPUT}")


if __name__ == "__main__":
    main()
