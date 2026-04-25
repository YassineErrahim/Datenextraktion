import json
import os
import shutil
import requests
import re

GITHUB_TOKEN = ""

SOURCE_DIR = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/Master_Arbeit_Data"
DEST_DIR   = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/FINAL_GOLDEN_SET"


MIN_ADDED_LINES   = 5
MIN_REMOVED_LINES = 1
MIN_TOTAL_CHANGES = 10

MAX_DIFF_LINES = 400

CATEGORIES = ["Quality", "Security", "Performance", "Refactor", "Testing"]

MIN_DIFF_CHARS    = 50
MAX_DIFF_CHARS = 64_000 * 4 # mmm for the max tokens context window, 4 charcters in average for a token

def has_at_least_one_comment(pr):
    for thread in pr.get('reviewThreads', {}).get('nodes', []):
        for comment in thread.get('comments', {}).get('nodes', []):
            if comment.get('body') and comment.get('body').strip():
                return True

    return False


def analyse_diff(raw_diff: str) -> dict:
    added   = 0
    removed = 0
    for line in raw_diff.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            added += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed += 1
    return {
        'added':   added,
        'removed': removed,
        'total':   added + removed,
    }


def fetch_full_diff(owner: str, repo: str, pr_number: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.text

def extract_pr_url_parts(pr: dict) -> tuple[str, str, str]:
    url = pr.get("url", "")
    match = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    return match.group(1), match.group(2), match.group(3)

def has_sufficient_changes(pr) -> tuple[bool, str, str]:
    try:
        owner, repo, pr_number = extract_pr_url_parts(pr)
        full_diff = fetch_full_diff(owner, repo, pr_number)
    except Exception as e:
        return False, 'diff_too_large', ''
    
    if len(full_diff) > MAX_DIFF_CHARS:
        return False, 'diff_too_large', ''
    
    if len(full_diff) < MIN_DIFF_CHARS:
        return False, 'diff_too_short', ''

    stats = analyse_diff(full_diff)

    if stats['added'] < MIN_ADDED_LINES:
        return False, 'too_few_added', ''

    if stats['removed'] < MIN_REMOVED_LINES:
        return False, 'too_few_removed', ''

    if stats['total'] < MIN_TOTAL_CHANGES:
        return False, 'diff_too_small', ''
    
    if stats['total'] > MAX_DIFF_LINES:
        return False, 'diff_too_many_lines', ''

    return True, '', full_diff


def validate_prereview_prs(pr) -> tuple[bool, str]:
    threads = pr.get("reviewThreads", {}).get("nodes", [])
    if not threads:
        return False, 'no_prereview_commits'
    first_comment_dates = []
    for thread in threads:
        comment = thread["comments"]["nodes"][0]
        first_comment_dates.append(comment["createdAt"])
    if not first_comment_dates:
        return False, 'no_prereview_commits'
    first_review_date = min(first_comment_dates)
    commit_nodes = pr.get("commits", {}).get("nodes", [])
    pre_review_commits = [
        n for n in commit_nodes
        if not n["commit"]["message"].startswith("Merge")
        and n["commit"]["committedDate"] <= first_review_date
    ]
    if not pre_review_commits:
        return False, 'no_prereview_commits'
    return True, ''

def check_duplication(unique_filename: str) -> bool:
    for category in CATEGORIES:
        path = os.path.join(DEST_DIR, category, unique_filename)
        if os.path.exists(path):
            return True
    return False


def validate_and_fix_collisions():
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR)

    SKIP_DIRS = {"_clones", ".DS_Store"}
    for repo in sorted(os.listdir(SOURCE_DIR)):
        if repo in SKIP_DIRS:
            continue
        repo_path = os.path.join(SOURCE_DIR, repo)
        print(f"\nRepo: {repo} \n")
        rejections = {
            'is_revert':0,
            'no_comment':0,
            'diff_too_short':0,   
            'too_few_added':0,   
            'too_few_removed':0,   
            'diff_too_small':0,   
            'diff_too_large':0, 
            'diff_too_many_lines': 0,
            'no_prereview_commits':0
        }
        for category in CATEGORIES:
            cat_path = os.path.join(repo_path, category)
            total_removed_by_category = 0
            for file in sorted(os.listdir(cat_path)):
                filepath = os.path.join(cat_path, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        pr = json.load(f)
                    except json.JSONDecodeError:
                        print(f"  Error JSONDecodeError {filepath}")
                        continue

                if "Revert" in pr.get('title', ''):
                    rejections['is_revert'] += 1
                    total_removed_by_category+=1
                    continue

                if not has_at_least_one_comment(pr):
                    rejections['no_comment'] += 1
                    total_removed_by_category+=1
                    continue

                diff_ok, diff_reason, full_diff = has_sufficient_changes(pr)
                if not diff_ok:
                    rejections[diff_reason] += 1
                    total_removed_by_category+=1
                    continue

                prereview_ok, prereview_reason = validate_prereview_prs(pr)
                if not prereview_ok:
                    rejections[prereview_reason] += 1
                    total_removed_by_category+=1
                    continue

                unique_filename = f"{repo}_{file}"
                if check_duplication(unique_filename):
                    continue

                pr['full_diff'] = full_diff

                dest_folder = os.path.join(DEST_DIR, category)
                os.makedirs(dest_folder, exist_ok=True)

                unique_filename = f"{repo}_{file}"
                dest_path = os.path.join(dest_folder, unique_filename)

                with open(dest_path, 'w', encoding='utf-8') as f:
                    json.dump(pr, f, indent=2)
            print(f" - Removed in {category} : {total_removed_by_category}")
        print(f" - Rejections: {rejections} \n\n")

if __name__ == "__main__":
    validate_and_fix_collisions()
