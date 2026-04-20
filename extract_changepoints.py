from __future__ import annotations
import json
import re
import time
import requests
from dataclasses import dataclass, field
from pathlib import Path
import anthropic
from openai import OpenAI
from google import genai
from google.genai import types

GEMINI_API_KEY = ""
OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
GITHUB_TOKEN = ""


GOLDEN_DIR = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/FINAL_GOLDEN_SET"
OUTPUT_DIR = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CHANGEPOINT_SET"

MAX_PRS = 0
CATEGORY = ""
GEMINI_MODEL = "gemini-2.5-pro"
OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-sonnet-4-5"

VOTES_REQUIRED = 2
TOTAL_VOTES = 3


@dataclass
class ReviewerComment:
    body: str
    path: str
    created_at: str
    diff_hunk: str

@dataclass
class LLMVerification:
    votes: list[str]
    models: list[str]
    consensus: str
    confidence: float

@dataclass
class ChangePoint:
    id: str
    reviewer_comment: ReviewerComment
    llm_verification: LLMVerification

@dataclass
class PRResult:
    pr_number: int
    repo: str
    category: str
    filepath:str
    change_points: list[ChangePoint] = field(default_factory=list)


SYSTEM_PROMPT = """You are a senior software engineer analysing GitHub Pull Request review conversations.

Your task: given a reviewer comment and the FULL diff of the PR (all changes made in this pull request),
decide whether the developer addressed this comment somewhere in the full diff. 
The comment's diff hunk and file path are provided only as additional context to help locate where the comment was posted, but use mainly the full diff.

You must answer in EXACTLY this format and nothing else:
VERDICT: YES

OR:

VERDICT: NO

Rules:
- YES = the comment was fully addressed in the final PR diff.
- NO = the final diff shows no change related to this comment."""


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


def build_prompt(comment: ReviewerComment) -> str:
    return (
        "=== REVIEWER COMMENT ===\n"
        f"{comment.body}\n\n"

        "=== COMMENT LOCATION (diff hunk where comment was posted, for context only) ===\n"
        "```diff\n"
        f"{comment.diff_hunk}\n"
        "```\n\n"

        "=== FILE PATH ===\n"
        f"{comment.path}\n\n"

        "=== INSTRUCTIONS ===\n"
        "1. First look in the diff hunk above for evidence the comment was addressed\n"
        f"2. If not found in the diff hunk, search the file '{comment.path}' in the full PR diff\n"
        "3. If still not found, search ALL other files in the full PR diff\n"
        "4. The fix may be anywhere in the full diff — different location, different file, different section\n"
        "5. Look for '+' lines (additions) or '-' lines (deletions) that relate to the reviewer comment\n"
        "6. If you find evidence anywhere in the full diff → VERDICT: YES\n"
        "7. Only if absolutely no related change exists anywhere in the entire full diff → VERDICT: NO\n"
    )

def call_gemini(comments: list[ReviewerComment], full_diff: str) -> list[str]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        cache = client.caches.create(
            model=GEMINI_MODEL,
            config=types.CreateCachedContentConfig(
                display_name="pr_diff_cache",
                system_instruction=SYSTEM_PROMPT,
                contents=[f"=== FULL PR DIFF (all changes in this PR) ===\n{full_diff}"],
                ttl="3600s",
            ),
        )
        use_cache = True
    except Exception as e:
        if "too small" in str(e).lower() or "2048" in str(e) or "INVALID_ARGUMENT" in str(e):
            use_cache = False
        else:
            raise

    def call(prompt: str) -> str:
        if use_cache:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    cached_content=cache.name,
                    temperature=0.0,
                    max_output_tokens=8192,
                ),
            )
        else:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=8192,
                ),
            )
        return (response.text or "").strip()

    results = []
    try:
        for comment in comments:
            prompt = build_prompt(comment) if use_cache else (
                f"=== FULL PR DIFF (all changes in this PR) ===\n{full_diff}\n\n{build_prompt(comment)}"
            )
            try:
                results.append(call(prompt))
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    time.sleep(61)
                    results.append(call(prompt))
                else:
                    raise RuntimeError(f"Gemini call failed: {e}") from e
    finally:
        if use_cache:
            client.caches.delete(name=cache.name)

    return results


def call_openai(comments: list[ReviewerComment], full_diff: str) -> list[str]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"=== FULL PR DIFF (all changes in this PR) ===\n{full_diff}"},
    ]
    results = []
    for comment in comments:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=base_messages + [{"role": "user", "content": build_prompt(comment)}],
                temperature=0.0,
                max_tokens=8192,
            )
            results.append(resp.choices[0].message.content.strip() or "")
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(61)
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=base_messages + [{"role": "user", "content": build_prompt(comment)}],
                    temperature=0.0,
                    max_tokens=8192,
                )
                results.append(resp.choices[0].message.content.strip() or "")
            else:
                raise RuntimeError(f"OpenAI call failed: {e}") from e

    return results


def call_anthropic(comments: list[ReviewerComment], full_diff: str) -> list[str]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    cached_system = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": f"=== FULL PR DIFF (all changes in this PR) ===\n{full_diff}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    results = []
    for comment in comments:
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=8192,
                system=cached_system,
                messages=[{"role": "user", "content": build_prompt(comment)}],
            )
            results.append(response.content[0].text.strip())
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(61)
                response = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=8192,
                    system=cached_system,
                    messages=[{"role": "user", "content": build_prompt(comment)}],
                )
                results.append(response.content[0].text.strip())
            else:
                raise RuntimeError(f"Anthropic call failed: {e}") from e

    return results

def parse_response(text: str) -> str:
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("VERDICT:"):
            v = stripped.split(":", 1)[1].strip().upper()
            if "YES" in v:
                return "YES"
    return "NO"


def majority_vote(comments: list[ReviewerComment], full_diff: str) -> list[LLMVerification]:
    gemini_votes = call_gemini(comments, full_diff)
    openai_votes = call_openai(comments, full_diff)
    anthropic_votes = call_anthropic(comments, full_diff)
    results = []
    for i in range(len(comments)):
        votes = [gemini_votes[i], openai_votes[i], anthropic_votes[i]]
        parsed = [parse_response(v) for v in votes]
        yes_count = sum(1 for v in parsed if v == "YES")
        consensus = "YES" if yes_count >= VOTES_REQUIRED else "NO"
        confidence = round(yes_count / TOTAL_VOTES, 2)
        results.append(LLMVerification(
            votes = parsed,
            models = [GEMINI_MODEL, OPENAI_MODEL, ANTHROPIC_MODEL],
            consensus = consensus,
            confidence = confidence,
        ))
    return results



def extract_all_comments(pr: dict) -> list[ReviewerComment]:
    comments = []
    threads = pr.get("reviewThreads", {}).get("nodes", []) or []
    for thread in threads:
        nodes = thread.get("comments", {}).get("nodes", []) or []
        for cm in nodes:
            if not (cm.get("body") or "").strip():
                continue
            comments.append(ReviewerComment(
                body = cm.get("body", "").strip(),
                path = cm.get("path") or "unknown",
                created_at = cm.get("createdAt"),
                diff_hunk = cm.get("diffHunk") or "",
            ))
    return comments


def extract_pr_changepoints(pr: dict, pr_result: PRResult) -> None:
    owner, repo, pr_number = extract_pr_url_parts(pr)
    full_diff = fetch_full_diff(owner, repo, pr_number)
    comments = extract_all_comments(pr)
    verifications = majority_vote(comments, full_diff)
    print(f"\n\ncomments: {len(comments)}, verfications: {len(verifications)}")
    for c_idx, (comment, verification) in enumerate(zip(comments, verifications)):
        cp = ChangePoint(
            id = f"{pr_result.pr_number}_comment_{c_idx}",
            reviewer_comment = comment,
            llm_verification = verification,
        )
        pr_result.change_points.append(cp)
    return full_diff    


def to_dict(cp: ChangePoint) -> dict:
    return {
        "id": cp.id,
        "reviewer_comment": {
            "body": cp.reviewer_comment.body,
            "path": cp.reviewer_comment.path,
            "created_at": cp.reviewer_comment.created_at,
            "diff_hunk": cp.reviewer_comment.diff_hunk,
        },
        "llm_verification": {
            "votes": cp.llm_verification.votes,
            "models": cp.llm_verification.models,
            "consensus": cp.llm_verification.consensus,
            "confidence": cp.llm_verification.confidence,
        },
    }


def save_pr(pr: dict, pr_result: PRResult, output_dir: Path, full_diff: str) -> None:
    dest_dir = output_dir / pr_result.category
    dest_dir.mkdir(parents=True, exist_ok=True)
    copy_pr = dict(pr)
    copy_pr["change_points"] = [to_dict(cp) for cp in pr_result.change_points]
    copy_pr["full_diff"] = full_diff
    dest_path = dest_dir / Path(pr_result.filepath).name
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(copy_pr, f, indent=2)

def main():
    golden = Path(GOLDEN_DIR)
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    all_files = sorted([
        str(f) for f in golden.rglob("*.json")
        if not f.name.startswith(".")
    ])
    if CATEGORY:
        all_files = [f for f in all_files if Path(f).parent.name == CATEGORY]
    if MAX_PRS:
        all_files = all_files[:MAX_PRS]

    total = len(all_files)
    print(f"Starting extraction: {total} PRs", flush=True)

    for i, filepath in enumerate(all_files, 1):
        fname = Path(filepath).stem
        parts = fname.rsplit("_pr_", 1)
        repo = parts[0] if len(parts) == 2 else fname
        category = Path(filepath).parent.name

        dest = output / category / Path(filepath).name
        if dest.exists():
            print(f"already done {fname}", flush=True)
            continue

        try:
            with open(filepath, encoding="utf-8") as f:
                pr = json.load(f)
        except Exception as e:
            print(f"SKIP {fname}: {e}", flush=True)
            continue

        print(f"{fname}", flush=True)

        pr_number = pr.get("number", 0)
        pr_result = PRResult(
            pr_number = pr_number,
            repo = repo,
            category = category,
            filepath = filepath,
        )

        try:
            full_diff = extract_pr_changepoints(pr, pr_result)
            save_pr(pr, pr_result, output, full_diff)
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)

    print(f"\n\n\nDone — {total} PRs processed → {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
