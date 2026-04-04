from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import anthropic
from openai import OpenAI


GEMINI_API_KEY = ""
OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""

GOLDEN_DIR = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/FINAL_GOLDEN_SET"
OUTPUT_DIR = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CHANGEPOINT_SET"

MAX_PRS = 0 #i want to specify prs counts for testing, 0 for all 
CATEGORY = "" # same principe, empty for all
GEMINI_MODEL = "gemini-2.5-pro"
OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-sonnet-4-5"

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
VOTES_REQUIRED = 2
TOTAL_VOTES = 3
CALL_DELAY = 2

@dataclass
class ReviewerComment:
    bodies: list[str]            
    paths: list[str]            
    lines: list[Optional[int]]
    created_ats: list[Optional[str]] 
    diff_hunks: list[Optional[str]]

@dataclass
class CodeChange:
    commit_sha: str
    committed_at: Optional[str]
    diff_excerpt: str
    files_changed: list[str]

@dataclass
class LLMVerification:
    votes: list[str]        
    models: list[str]        
    consensus: str              
    confidence: float            
    trigger: str # majority trigger: "comment"|"body"|"both"|"none"
    triggering_comments: list[list[int]]

@dataclass
class ChangePoint:
    id: str
    reviewer_comment: ReviewerComment
    code_change: Optional[CodeChange]
    llm_verification: LLMVerification

@dataclass
class PRResult:
    pr_number: int
    repo: str
    category: str
    filepath: str
    change_points: list[ChangePoint] = field(default_factory=list)



SYSTEM_PROMPT = """You are a senior software engineer analysing GitHub Pull Request review conversations.
Your task: decide whether a developer made a code change in response to EITHER the reviewer's comment OR the PR description.

You must answer in EXACTLY this format and please nothing else:
VERDICT: YES
TRIGGER: comment
TRIGGERING_COMMENTS: <comma-separated list of comment indices, e.g., 0,2>

OR:

VERDICT: PARTIAL
TRIGGER: body
TRIGGERING_COMMENTS: 1

OR:

VERDICT: NO
TRIGGER: none
TRIGGERING_COMMENTS:

Rules:
- VERDICT must be YES, PARTIAL, or NO.
  * YES = the change was fully triggered by the comment or PR description.
  * PARTIAL = the change was partially triggered.
  * NO = the commit shows no change related to either the comment or PR description.
- TRIGGER must be one of: comment | body | both | none
  * comment = change was triggered by the reviewer comment
  * body = change was triggered by the PR description and not the comment
  * both = both the comment and PR description point to this change
  * none = change has no connection to comment or PR description
- TRIGGERING_COMMENTS must be a comma separated list of 0-based indices of the thread
  comments (shown as [0], [1], [2]...) that directly triggered the code change.
  Only include comments that are actual reviewer requests and not developer replies like
  "done", "fixed", "you are correct". and please leave empty when VERDICT is NO.
- Focus on the specific file and lines the reviewer commented on.
- Even if the reviewer used uncertain language ("maybe", "consider", "could"),
  if the developer made the suggested change → answer YES or PARTIAL based on the changes made.
- Do not add any other text before or after this block."""


def build_thread_section(comment: ReviewerComment) -> str:
    parts = []
    for i, (bodyComments, pathsFilesChanged, linesSpecifedChanges, codeHasBeenSeen) in enumerate(zip(
        comment.bodies, comment.paths, comment.lines, comment.diff_hunks
    )):
        hunk      = codeHasBeenSeen or ""
        line_info = linesSpecifedChanges or "unknown"
        entry     = f"[{i}] (file: {pathsFilesChanged}, line: {line_info})\n{bodyComments}"
        if hunk and i == 0:
            entry += f"\n\nCODE THE REVIEWER SAW:\n{hunk}"
        parts.append(entry)

    if len(parts) == 1:
        return f"REVIEW THREAD (1 comment exists):\n\n{parts[0]}"
    return (
        f"REVIEW THREAD ({len(parts)} comments — full conversation):\n\n"
        + "\n\n".join(parts)
    )

def build_prompt(comment: ReviewerComment, diff_excerpt: str, pr_body: str = "") -> str:
    pr_context = ""
    if pr_body := (pr_body or "").strip():
        pr_context = f"PR DESCRIPTION:\n{pr_body}\n\n"

    thread_section = build_thread_section(comment)
    return f"""{pr_context}{thread_section} 
            CODE CHANGES AFTER THESE COMMENTS (commit diff — last commit in window):
            {diff_excerpt if diff_excerpt else '[no diff available]'}

            IMPORTANT: Consider BOTH the reviewer comment and the PR description as possible
            triggers. A change-point exists if the commit addresses any of them.
            Focus on the specific files and lines the reviewer commented on.

            Question: Was this code change triggered by the reviewer comment(s), the PR description, or both?"""


def call_gemini(client: OpenAI, prompt: str) -> str:
    for i in range(4):
        try:
            resp = client.chat.completions.create( model=GEMINI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ], temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"Gemini call failed: {e}") from e
    raise RuntimeError("Unexpected failure in call_gemini")


def call_openai(client: OpenAI, prompt: str) -> str:
    for i in range(4):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL, messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}], temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {e}") from e
    raise RuntimeError("Unexpected failure in call_openai")


def call_anthropic_model(client: anthropic.Anthropic, prompt: str) -> str:
    for i in range(4):
        try:
            resp = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=4096, system=SYSTEM_PROMPT, messages=[ {"role": "user", "content": prompt}],
            )
            return resp.content[0].text if resp.content else ""
        except Exception as e:
            raise RuntimeError(f"Anthropic call failed: {e}") from e
    raise RuntimeError("Unexpected failure in call_anthropic_model")


def parse_response(text: str) -> tuple[str, str, list[int]]:
    verdict = "NO"
    trigger = "none"
    triggering_comments = []

    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("VERDICT:"):
            v = stripped.split(":", 1)[1].strip().upper()
            verdict = "YES" if ("YES" in v or "PARTIAL" in v) else "NO"
        elif stripped.startswith("TRIGGER:"):
            t = stripped.split(":", 1)[1].strip().lower()
            if t in ("comment", "body", "both", "none"):
                trigger = t
        elif stripped.startswith("TRIGGERING_COMMENTS:"):
            raw = stripped.split(":", 1)[1].strip()
            for part in raw.split(","):
                part = part.strip()
                if part.isdigit():
                    triggering_comments.append(int(part))

    return verdict, trigger, triggering_comments

def majority_vote(gemini_client: OpenAI, openai_client: OpenAI, anthropic_client: anthropic.Anthropic, prompt: str) -> LLMVerification:
    model_names = [GEMINI_MODEL, OPENAI_MODEL, ANTHROPIC_MODEL]
    raw_responses = [
        call_gemini(gemini_client, prompt),
        call_openai(openai_client, prompt),
        call_anthropic_model(anthropic_client, prompt)
    ]
    votes = []
    all_triggers = []
    all_triggering_comments= []

    for raw in raw_responses:
        verdict, trig, trig_comments = parse_response(raw)
        votes.append(verdict)
        all_triggers.append(trig)
        all_triggering_comments.append(trig_comments)
        time.sleep(CALL_DELAY)

    yes_count = votes.count("YES")
    consensus = "YES" if yes_count >= VOTES_REQUIRED else "NO"
    confidence = round(yes_count / TOTAL_VOTES, 2)

    yes_idx = [i for i, v in enumerate(votes) if v == "YES"] or list(range(TOTAL_VOTES))
    trigger_votes = [all_triggers[i] for i in yes_idx]
    final_trigger = max(set(trigger_votes), key=trigger_votes.count) if trigger_votes else "none"
    verification = LLMVerification(
        votes = votes,
        models = model_names,
        consensus = consensus,
        confidence = confidence,
        trigger = final_trigger,
        triggering_comments = all_triggering_comments,
    )
    return verification


def parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


REBASE_PATTERN = re.compile(
    r"^(Merge (branch|pull request|remote-tracking branch|tag)|"
    r"Rebase|squash!|fixup!|Apply suggestions|"
    r"Co-authored-by:|Signed-off-by:)",
    re.IGNORECASE,
)

def is_rebase_commit(message: str) -> bool:
    return bool(REBASE_PATTERN.match((message or "").strip()))

def build_timeline(pr: dict) -> list[dict]:
    icd = pr.get("individual_commit_diffs", {}) or {}
    timeline = []
    threads = pr.get("reviewThreads", {}).get("nodes", []) or []
    for thread in threads:
        nodes = thread.get("comments", {}).get("nodes", []) or []
        if not nodes:
            continue
        thread_comments = [cm for cm in nodes if (cm.get("body") or "").strip()]
        if not thread_comments:
            continue
        first_cm = thread_comments[0]
        date = parse_date(first_cm.get("createdAt"))
        timeline.append({
            "type": "comment",
            "date": date,
            "thread_comments": thread_comments,
        })

    for node in pr.get("commits", {}).get("nodes", []) or []:
        commit = node.get("commit", {})
        sha = commit.get("oid", "")
        message = commit.get("message", "")
        date = parse_date(commit.get("committedDate"))
        if is_rebase_commit(message):
            continue

        diff = icd.get(sha, "") or ""
        if not diff or diff == "Diff unavailable":
            continue

        timeline.append({
            "type": "commit",
            "date": date,
            "committed_at": commit.get("committedDate", ""),
            "sha": sha,
            "message": message,
            "diff": diff,
        })
        
    def sort_key(e):
        d = e["date"]
        if d is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        return d

    timeline.sort(key=sort_key)
    return timeline


def group_timeline(timeline: list[dict]) -> list[dict]:
    comment_buf = []
    last_commit = None
    groups = []
    for entry in timeline:
        if entry["type"] == "comment":
            if last_commit is not None and comment_buf:
                last_comment_date = comment_buf[-1]["date"]
                commit_date = last_commit["date"]
                valid = (
                    last_comment_date is None
                    or commit_date is None
                    or commit_date > last_comment_date
                )
                groups.append({
                    "comments": comment_buf[:],
                    "commit":last_commit if valid else None,
                })
                comment_buf = []
                last_commit = None
            comment_buf.append(entry)
        else:
            last_commit = entry

    if comment_buf:
        last_comment_date = comment_buf[-1]["date"]
        commit_date = last_commit["date"] if last_commit else None
        valid = (last_commit is not None
            and (last_comment_date is None or commit_date is None
                 or commit_date > last_comment_date))
        groups.append({
            "comments": comment_buf,
            "commit":   last_commit if valid else None,
        })

    return groups


def extract_pr_changepoints(pr: dict, pr_result: PRResult, gemini_client: OpenAI, openai_client: OpenAI, anthropic_client: anthropic.Anthropic) -> None:
    pr_body = pr.get("body", "") or ""
    timeline = build_timeline(pr)
    groups = group_timeline(timeline)
    for g_id_index, group in enumerate(groups):
        comments_raw = group["comments"]
        commit_raw = group["commit"]
        all_thread_comments = []
        for thread_entry in comments_raw:
            all_thread_comments.extend(thread_entry["thread_comments"])

        reviewer_comment = ReviewerComment(
            bodies = [c.get("body", "").strip() for c in all_thread_comments],
            paths = [c.get("path") or "unknown" for c in all_thread_comments],
            lines = [c.get("line") for c in all_thread_comments],
            created_ats = [c.get("createdAt") for c in all_thread_comments],
            diff_hunks = [c.get("diffHunk") for c in all_thread_comments],
        )
        if not commit_raw:
            verification = LLMVerification(
                votes = ["NO", "NO", "NO"],
                models = [GEMINI_MODEL, OPENAI_MODEL, ANTHROPIC_MODEL],
                consensus = "NO",
                confidence = 0.0,
                trigger = "none",
                triggering_comments = [[], [], []],
            )
            cp = ChangePoint(
                id = f"{pr_result.pr_number}_group_{g_id_index}",
                reviewer_comment = reviewer_comment,
                code_change = None,
                llm_verification = verification,
            )
            pr_result.change_points.append(cp)
            continue

        diff_full = commit_raw["diff"]
        sha = commit_raw["sha"]
        committed_at = commit_raw["committed_at"]
        files_changed = re.findall(r"diff --git a/(\S+)", diff_full)

        code_change = CodeChange(
            commit_sha = sha,
            committed_at = committed_at,
            diff_excerpt = diff_full,
            files_changed = files_changed,
        )

        prompt = build_prompt(reviewer_comment, diff_full, pr_body)
        verification = majority_vote(
            gemini_client, openai_client, anthropic_client, prompt
        )
        cp = ChangePoint(
            id = f"{pr_result.pr_number}_group_{g_id_index}",
            reviewer_comment = reviewer_comment,
            code_change = code_change,
            llm_verification = verification,
        )
        pr_result.change_points.append(cp)


def to_dict(cp: ChangePoint) -> dict:
    return {
        "id": cp.id,
        "reviewer_comment": {
            "thread": [
                {
                    "body": body,
                    "path": paths,
                    "line": lines,
                    "created_at": creation_time,
                    "diff_hunk": differences_make,
                }
                for body, paths, lines, creation_time, differences_make in zip(
                    cp.reviewer_comment.bodies,
                    cp.reviewer_comment.paths,
                    cp.reviewer_comment.lines,
                    cp.reviewer_comment.created_ats,
                    cp.reviewer_comment.diff_hunks,
                )
            ],
        },
        "code_change": {
            "commit_sha": cp.code_change.commit_sha,
            "committed_at": cp.code_change.committed_at,
            "diff_excerpt": cp.code_change.diff_excerpt,
            "files_changed": cp.code_change.files_changed,
        } if cp.code_change else None,
        "llm_verification": {
            "votes": cp.llm_verification.votes,
            "models": cp.llm_verification.models,
            "consensus": cp.llm_verification.consensus,
            "confidence": cp.llm_verification.confidence,
            "trigger": cp.llm_verification.trigger,
            "triggering_comments": cp.llm_verification.triggering_comments,
        }
    }


def save_pr(pr: dict, pr_result: PRResult, output_dir: Path) -> None:
    dest_dir = output_dir / pr_result.category
    dest_dir.mkdir(parents=True, exist_ok=True)
    copy_pr = dict(pr)
    copy_pr["change_points"] = [to_dict(cp) for cp in pr_result.change_points]
    dest_path = dest_dir / Path(pr_result.filepath).name
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(copy_pr, f, indent=2)


def main():
    golden = Path(GOLDEN_DIR)
    if not golden.exists():
        raise FileNotFoundError(f"Golden dir not found: {golden}")

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

    gemini_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_API_BASE)
    openai_client= OpenAI(api_key=OPENAI_API_KEY)
    anthropic_client= anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    total = len(all_files)
    print(f"Starting extraction: {total} PRs", flush=True)

    for i, filepath in enumerate(all_files, 1):
        fname = Path(filepath).stem
        parts = fname.rsplit("_pr_", 1)
        repo = parts[0] if len(parts) == 2 else fname
        category = Path(filepath).parent.name

        dest = output / category / Path(filepath).name
        if dest.exists():
            print(f"[{i}/{total}] already done {fname}", flush=True)
            continue

        try:
            with open(filepath, encoding="utf-8") as f:
                pr = json.load(f)
        except Exception as e:
            print(f"[{i}/{total}] SKIP {fname}: {e}", flush=True)
            continue

        pr_number = pr.get("number", 0)
        pr_result = PRResult(
            pr_number = pr_number,
            repo = repo,
            category = category,
            filepath = filepath,
        )

        try:
            extract_pr_changepoints(pr, pr_result, gemini_client, openai_client, anthropic_client)
            save_pr(pr, pr_result, output)
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)

    print(f"\nDone — {total} PRs processed → {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()