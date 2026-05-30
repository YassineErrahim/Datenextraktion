import argparse
import json
import os
import glob
import time
from google import genai
from google.genai import types

DEFAULT_BASE_DIR = (
    "/Users/yassine/Downloads/Master_Arbeit/Experiment/"
    "DataExtraction/CHANGEPOINT_SET"
)

GEMINI_API_KEY = "AIzaSyAonnmDeU8PvlYY_cycSWHT6D2dpVoco0k"

base_dir = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CHANGEPOINT_SET"
sleep = 10.0
force = False
model_id = "gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)
config = types.GenerateContentConfig(
    response_mime_type="application/json",
)

PROMPT_TEMPLATE = """You are a software engineering researcher classifying the 
ground truth quality of a pull request based on its reviewer comments.

A YES-consensus change point means the comment led to a real code change.
Your only job is to judge WHETHER THAT COMMENT WAS SHALLOW OR MEANINGFUL.

## Category definitions

- Substantive: The comment addresses a real bug, correctness issue, security 
  problem, or non-trivial design/architectural decision. It requires genuine 
  understanding of the code to write and act on.
  Examples: "this will cause infinite HTTP calls on every CD cycle", 
            "this condition will never return the value", 
            "use takeUntilDestroyed instead of manual subscription management"

- Shallow: The comment is a trivial request with no depth — access modifiers 
  (private→protected), import ordering, minor naming, whitespace, style formatting,
  or small cosmetic refactors with no behavioral impact.
  Examples: "mark as protected", "add blank line here", "rename this variable"

- Noisy: The comment is an author reply (not a reviewer request), 
  bot-generated, or a near-duplicate of 
  another comment in the same PR.
  Examples: "The latter.", "Ok done.", "Data updates once at midnight so should not be too hurtful"

- Documentation: The comment concerns only documentation, docstrings, README, 
  or text formatting. No code logic involved.
  Examples: "combine these paragraphs", "add blank line in RST file"

## Aggregation rules

1. Classify ONLY the YES-consensus change points below.
2. Near-duplicate comments (same request repeated) count as ONE signal.
3. Apply majority rule: the category with >= 50% of YES points wins.
4. Tie-break priority: Noisy > Shallow > Documentation > Substantive.

## Pull Request

Title: {title}

## YES-consensus change points (these are your only input)

{yes_change_points}

## Output

Return ONLY this JSON:
{{
  "yes_count": <number of YES change points>,
  "per_point_classification": [
    {{
      "id": "<id>",
      "category": "<Substantive|Shallow|Noisy|Documentation>",
      "reason": "<one sentence>"
    }}
  ],
  "category_counts": {{
    "Substantive": <n>,
    "Shallow": <n>,
    "Noisy": <n>,
    "Documentation": <n>
  }},
  "gt_quality_label": "<Substantive|Shallow|Noisy|Documentation>",
  "reason": "<2-3 sentences explaining the majority rule result>"
}}"""


def format_yes_change_points(change_points):
    yes_points = [
        cp for cp in change_points
        if cp["llm_verification"]["consensus"] == "YES"
    ]
    if not yes_points:
        return "None."
    lines = []
    for cp in yes_points:
        lv = cp["llm_verification"]
        rc = cp["reviewer_comment"]
        lines.append(
            f"[{cp['id']}] conf={lv['confidence']}\n"
            f"  comment: {rc['body'][:600]}"
        )
    return "\n\n".join(lines)


def classify_pr(pr_data):
    title = pr_data.get("title", "")

    yes_points = [
        cp for cp in pr_data.get("change_points", [])
        if cp["llm_verification"]["consensus"] == "YES"
    ]

    if not yes_points:
        return {
            "yes_count": 0,
            "per_point_classification": [],
            "category_counts": {
                "Substantive": 0, "Shallow": 0,
                "Noisy": 0, "Documentation": 0
            },
            "gt_quality_label": "Noisy",
            "reason": "No YES-consensus change points exist. GT is unreliable.",
        }

    prompt = PROMPT_TEMPLATE.format(
        title=title,
        yes_change_points=format_yes_change_points(pr_data.get("change_points", [])),
    )
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=config
    )
    raw = response.text.strip()

    # strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def collect_pr_files(base_dir):
    pattern = os.path.join(base_dir, "**", "*.json")
    return sorted(glob.glob(pattern, recursive=True))


def main():
    pr_files = collect_pr_files(base_dir)
    print(f"Found {len(pr_files)} PR JSON file(s) under {base_dir}")

    done = skipped = errors = 0

    for fpath in pr_files:
        with open(fpath) as f:
            pr_data = json.load(f)

        pr_num = pr_data.get("number", os.path.basename(fpath))
        if "classification" in pr_data and not force:
            print(f"  Skip PR {pr_num} (already has classification)")
            skipped += 1
            continue

        yes_count = sum(
            1 for cp in pr_data.get("change_points", [])
            if cp["llm_verification"]["consensus"] == "YES"
        )
        title_short = pr_data.get("title", "")[:55]
        print(f"  PR {pr_num} ({yes_count} YES): {title_short}...")

        try:
            classification = classify_pr(pr_data)
            pr_data["classification"] = classification
            with open(fpath, "w") as f:
                json.dump(pr_data, f, indent=2, ensure_ascii=False)

            label = classification.get("gt_quality_label", "?")
            counts = classification.get("category_counts", {})
            print(f"    → {label} | {counts}")
            done += 1

        except Exception as e:
            print(f"    ERROR on PR {pr_num}: {e}")
            errors += 1

        time.sleep(sleep)

    print(f"\n=== Done ===")
    print(f"  Classified : {done}")
    print(f"  Skipped    : {skipped}")
    print(f"  Errors     : {errors}")


if __name__ == "__main__":
    main()
