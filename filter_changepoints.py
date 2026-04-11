import json
import os
import shutil

SRC_DIR = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CHANGEPOINT_SET"
DST_DIR = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CHANGEPOINT_FILTREDSET"
CATEGORIES = ["Performance", "Quality", "Refactor", "Security", "Testing"]

def is_not_clean(pr: dict) -> bool:
    for cp in pr.get("change_points", []):
        if cp.get("llm_verification", {}).get("votes", []).count("YES") >= 2:
            return True
    return False

total = 0
saved = 0

for category in CATEGORIES:
    src_category_dir = os.path.join(SRC_DIR, category)
    dst_category_dir = os.path.join(DST_DIR, category)
    if not os.path.isdir(src_category_dir):
        continue
    os.makedirs(dst_category_dir, exist_ok=True)
    for filename in os.listdir(src_category_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(src_category_dir, filename)
        total += 1
        with open(filepath, "r", encoding="utf-8") as f:
            pr = json.load(f)
        if is_not_clean(pr):
            shutil.copy2(filepath, os.path.join(dst_category_dir, filename))
            saved += 1

print(f"Processed: {total} | Saved: {saved}")
