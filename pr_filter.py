import json
import os
import shutil


SOURCE_DIR = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/Master_Arbeit_Data"
DEST_DIR   = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/FINAL_GOLDEN_SET"



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

MIN_ADDED_LINES   = 5
MIN_REMOVED_LINES = 1
MIN_TOTAL_CHANGES = 10
MAX_TOTAL_CHANGES = 300
MIN_DIFF_CHARS    = 50

def has_sufficient_changes(pr) -> tuple[bool, str]:
    head_sha = pr.get('headRefOid', '')
    diff = pr.get('individual_commit_diffs', {}).get(head_sha, '')
    if len(diff) < MIN_DIFF_CHARS:
        return False, 'diff_too_short'

    stats = analyse_diff(diff)

    if stats['added'] < MIN_ADDED_LINES:
        return False, 'too_few_added'

    if stats['removed'] < MIN_REMOVED_LINES:
        return False, 'too_few_removed'

    if stats['total'] < MIN_TOTAL_CHANGES:
        return False, 'diff_too_small'

    if stats['total'] > MAX_TOTAL_CHANGES:
        return False, 'diff_too_large'

    return True, ''

def validate_and_fix_collisions():
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR)
    categories = ["Quality", "Security", "Performance", "Refactor", "Testing"]

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
        }
        for category in categories:
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

                diff_ok, diff_reason = has_sufficient_changes(pr)
                if not diff_ok:
                    rejections[diff_reason] += 1
                    total_removed_by_category+=1
                    continue

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
