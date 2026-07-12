import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from threading import Timer

from flask import Flask, jsonify, request

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None



REPO_CLONE_DIR = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/Master_Arbeit_Data/_clones"
REPORTS_DIR = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/CODE_REVIEW_REPORTS"

MESSAGES_DEBUG_DIR = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/MESSAGES_DEBUG"

ANTHROPIC_API_KEY = ""
COMMERCIAL_MODELS = [
    {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    {"provider": "openai", "model": "gpt-5.4"},
    {"provider": "deepseek", "model": "deepseek-v4-pro"}
]

ANTHROPIC_PRICE_INPUT_PER_M = 3.00   
ANTHROPIC_PRICE_CACHE_CREATION_PER_M = 3.75
ANTHROPIC_PRICE_CACHE_READ_PER_M = 0.30
ANTHROPIC_PRICE_OUTPUT_PER_M = 15.00

OPENAI_API_KEY = ""
OPENAI_BASE_URL = "https://api.openai.com/v1"         
OPENAI_PRICE_INPUT_PER_M = 2.50
OPENAI_PRICE_CACHE_READ_PER_M = 0.25
OPENAI_PRICE_OUTPUT_PER_M = 15.00

DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_PRICE_INPUT_PER_M   = 0.435
DEEPSEEK_PRICE_CACHE_HIT_PER_M = 0.003625
DEEPSEEK_PRICE_OUTPUT_PER_M  = 0.87


# SSH tunnel: ssh -L 11434:localhost:11434 root@YOUR_IP -p YOUR_PORT -i /Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/runpodkey -N
VLLM_BASE_URL = "http://64.247.206.204:49003/v1"
VLLM_API_KEY  = "token-abc123"

RUNPOD_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct-AWQ", #VLLM_CONTEXT_LIMIT = 131072
    "neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8", #VLLM_CONTEXT_LIMIT = 131072
    "mistralai/Codestral-22B-v0.1", #VLLM_CONTEXT_LIMIT = 81920 (nativ context 32k)
    "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", #VLLM_CONTEXT_LIMIT = 131072,
    "google/gemma-4-31b-it", # VLLM_CONTEXT_LIMIT = 163840
]
VLLM_CONTEXT_LIMIT = 131072
RUNPOD_MODEL_CONTEXT_LIMITS = {
    "Qwen/Qwen2.5-72B-Instruct-AWQ": 131072,
    "neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8": 131072,
    "mistralai/Codestral-22B-v0.1": 81920,
    "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4": 131072,
    "google/gemma-4-31b-it": 163840,
}
VLLM_MAX_OUTPUT = 8000

MAX_TOOL_CALLS = 60
REVIEW_TIMEOUT_S = 1800 # per-model hard timeout in seconds (30 min)
SLEEP_AFTER_HOW_MUCH_ITERATIONS = 2



SYSTEM_PROMPT = """You are a senior software engineer performing a pull request code review.

You have three tools to look up additional context from the repository when needed:
  get_file(path) — read any file's full content
  search_symbol(symbol, file_extension)  — find all usages of a symbol across the repo
  get_file_tree(path, extension) — list files under a directory

When to use tools:
  - A symbol, function, or variable is removed or renamed → search_symbol to check impact
  - A function body changed significantly → get_file to see the full class or module
  - A new file is introduced → get_file_tree to understand where it fits
  - You need surrounding code to judge whether a change is safe and needed → get_file

When NOT to use tools:
  - Pure formatting or whitespace changes that are self-evident from the diff
  - When the diff already gives you enough context to judge confidently

######## OUTPUT FORMAT ########
No markdown fences. No text before or after. No extra keys.
Any deviation from this schema will be treated as a failure — the response will be rejected and nothing will be saved.

{
  "findings": [
    {
      "description":    "<what the problem is, which file, which line if known>",
      "severity_score": <float 0.0–10.0>
    }
  ],
  "summary": "<overall review in 2-4 sentences>",
  "verdict": "approve | request_changes"
}

######## RULES ########
  - Keep tool calls minimal — use at absolutly maximum 60 tool calls only if needed to understand the context.
  - When you have gathered enough information, stop calling tools and write your final report immediately. Do not call the same tool twice with the same arguments.
  - "findings" must be a list. Use [] if there is nothing that needs to change.
  - Only report a finding if it represents something that genuinely needs to change — do NOT report observations, praise, or notes about correct code.
  - Each finding must have exactly these two keys: "description" and "severity_score".
  - "description" must be a non-empty string explaining what needs to change and why.
  - "severity_score" must be a float between 0.0 and 10.0 using these anchors:
      8–10  — must fix before merge
      4–7   — should fix before merge
      1–3   — minor but worth addressing
  - "verdict" must be exactly "approve" or "request_changes" — no other value is accepted.
  - "verdict" is "request_changes" if at least one finding requires a code change, otherwise "approve".
  - "summary" must be a non-empty string covering: what the PR does, overall quality, and the most important concern if any.
  - Do NOT merge multiple distinct problems into one finding.
"""

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "get_file",
            "description": (
                "Fetch the full content of a file from the repository "
                "(at the state before this PR). Use when you need to understand "
                "the full context around a changed function or class."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path from repo root, e.g. 'src/Umbraco.Core/Security/FileStreamSecurityValidator.cs'",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_symbol",
            "description": (
                "Search the entire repository for usages of a symbol "
                "(function name, variable, class, etc.). Use when something "
                "is removed or renamed in the diff and you need to check impact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Exact symbol or pattern to search for",
                    },
                    "file_extension": {
                        "type": "string",
                        "description": "Optional: limit to this extension, e.g. 'cs', 'ts', 'py'",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_tree",
            "description": (
                "List files under a directory in the repository. "
                "Use to understand project structure or locate test files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path, e.g. 'src/Umbraco.Core' or '' for root",
                    },
                    "extension": {
                        "type": "string",
                        "description": "Optional: filter by extension, e.g. 'cs'",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

TOOLS_ANTHROPIC = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOLS_OPENAI
]

def prepare_repo(repo_name: str, base_ref_oid: str):
    repo_path = os.path.join(REPO_CLONE_DIR, repo_name)
    if not os.path.exists(repo_path):
        raise ValueError(
            f"Repo '{repo_name}' not found in {REPO_CLONE_DIR}. "
            f"Run git_clone_or_update first."
        )
    result = subprocess.run(
        ["git", "-C", repo_path, "checkout", "--force", base_ref_oid],
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode().strip()
        print(f"[prepare_repo] skipping — git checkout failed for {repo_name}@{base_ref_oid}: {stderr}")
        return None
    return repo_path

def save_report(model_name: str, file_name: str, report: dict, category: str = "") -> str:
    if "error" in report:
        print(f"[save_report] skipping save for {file_name} — report has error: {report['error']}\n")
        print(f"[save_report] full report: {json.dumps(report, indent=2)}\n")
        return None
    safe_model = re.sub(r"[:/\\]", "_", model_name)
    report_dir = os.path.join(REPORTS_DIR, safe_model, category)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{file_name}.json")
    report["_meta"]["ground_truth_category"] = category
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report_path

def save_messages_debug(file_name: str, model: str, messages: list):
    safe_model = re.sub(r"[:/\\]", "_", model)
    debug_dir = os.path.join(MESSAGES_DEBUG_DIR, safe_model)
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"{file_name}.json")
    with open(path, "w") as f:
        json.dump(messages, f, indent=2, default=str)

class RepoTools:
    def __init__(self, repo_path: str):
        self.root = Path(repo_path).resolve()
        if not self.root.exists():
            raise ValueError(f"Check path of the repository, it was not found or model to Masterarbeit Data: {self.root}")
    def _safe_resolve(self, relative: str) -> Path | None:
        target = (self.root / relative.lstrip("/")).resolve()
        try:
            common = os.path.commonpath([str(self.root), str(target)])
        except ValueError:
            return None
        return target if common == str(self.root) else None

    def get_file(self, path: str) -> dict:
        target = self._safe_resolve(path)
        if target is None:
            return {"error": "Path traversal not allowed"}
        if not target.exists():
            return {"error": f"File not found: {path}"}
        if not target.is_file():
            return {"error": f"Not a file: {path}"}
        try:
            content   = target.read_text(errors="replace")
            return {"path": path, "content": content}
        except Exception as e:
            return {"error": str(e)}

    def search_symbol(self, symbol: str, file_extension: str = None) -> dict:
        if not symbol or len(symbol.strip()) < 2:
            return {"error": "Symbol must be at least 2 characters"}
        cmd = ["grep", "-rn"]
        if file_extension:
            cmd += [f"--include=*.{file_extension.lstrip('.')}"]
        cmd += [symbol, str(self.root)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=100)
            output = result.stdout
            if not output.strip():
                return {"symbol": symbol, "matches": "No usages found in repository."}
            output = output.replace(str(self.root) + "/", "")
            lines = output.strip().splitlines()
            return {
                "symbol": symbol,
                "matches": "\n".join(lines),
                "count": len(lines)
            }
        except subprocess.TimeoutExpired:
            return {"error": "Search timed out"}
        except Exception as e:
            return {"error": str(e)}

    def get_file_tree(self, path: str = "", extension: str = None) -> dict:
        target = self._safe_resolve(path) if path else self.root
        if target is None:
            return {"error": "Path traversal not allowed"}
        if not target.exists():
            return {"error": f"Path not found: {path or '.'}"}
        cmd = ["find", str(target), "-type", "f"]
        if extension:
            cmd += ["-name", f"*.{extension.lstrip('.')}"]
        for noise in ["node_modules", ".git", "__pycache__", "dist", ".next", "build", "bin", "obj"]:
            cmd += ["!", "-path", f"*/{noise}/*"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=100)
            output = result.stdout.replace(str(self.root) + "/", "")
            lines = sorted(output.strip().splitlines())
            return {
                "path": path or ".",
                "files": "\n".join(lines),
                "count":len(lines),
            }
        except subprocess.TimeoutExpired:
            return {"error": "File tree timed out"}
        except Exception as e:
            return {"error": str(e)}

    def dispatch(self, tool_name: str, tool_args: dict) -> str:
        if tool_name == "get_file":
            result = self.get_file(tool_args.get("path", ""))
        elif tool_name == "search_symbol":
            result = self.search_symbol(tool_args.get("symbol", ""), tool_args.get("file_extension"))
        elif tool_name == "get_file_tree":
            result = self.get_file_tree(tool_args.get("path", ""), tool_args.get("extension"))
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        return json.dumps(result)


def build_user_prompt(pr: dict) -> str:
    body = re.sub(
        r"https://github\.com/user-attachments/assets/\S+",
        "[media attachment — not accessible]",
        pr.get("body", "") or "",
    ).strip()
    diff = (pr.get("diff", "") or "").strip()
    sections = [f"######## PR: {pr.get('title', 'Untitled')} ########\n"]
    if body:
        sections.append(f"######## PR Description ########\n{body}")
    sections.append(f"######## PR Diff ########\n```\n{diff}\n```")
    return "\n\n".join(sections)


def parse_final_report(text: str) -> dict:
    result = parse_json_response(text)
    if _is_tool_call(result):
        raise json.JSONDecodeError(f"LLM returned a tool call instead of a report: tool={result.get('tool')}", text, 0)
    return result


def validate_report(report: dict) -> dict:
    if "findings" not in report:
        raise ValueError("Schema violation: missing 'findings' key in LLM response")

    if not isinstance(report["findings"], list):
        raise ValueError("Schema violation: 'findings' must be a list")

    for i, f in enumerate(report["findings"]):
        desc = (f.get("description") or "").strip()
        if not desc:
            raise ValueError(f"Schema violation: finding[{i}] missing 'description'")
        raw_score = f.get("severity_score")
        if raw_score is None:
            raise ValueError(f"Schema violation: finding[{i}] missing 'severity_score'")
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            raise ValueError(f"Schema violation: finding[{i}] invalid severity_score '{raw_score}' — must be a number")
        if not (0.0 <= score <= 10.0):
            raise ValueError(f"Schema violation: finding[{i}] severity_score {score} out of range 0.0-10.0")

    if report.get("verdict") not in {"approve", "request_changes"}:
        raise ValueError(f"Schema violation: invalid verdict '{report.get('verdict')}' — must be 'approve' or 'request_changes'")

    if not report.get("summary"):
        raise ValueError("Schema violation: missing or empty 'summary'")

    return report


def _is_tool_call(parsed: dict) -> bool:
    return (
        "tool" in parsed
        and "args" in parsed
        and "findings" not in parsed
        and "verdict" not in parsed
    )

def _force_final_review(vllm_client, model, messages, tool_log, total_input, total_output, file_name, reason: str):
    print(f"[{model}] {reason} — forcing final review now ({len(tool_log)} tool calls so far)")
    save_messages_debug(file_name, model, messages)
    final_response = vllm_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=VLLM_MAX_OUTPUT,
        temperature=0.0,
        timeout=480,
    )
    if final_response.usage:
        total_input  += final_response.usage.prompt_tokens or 0
        total_output += final_response.usage.completion_tokens or 0
    content = (final_response.choices[0].message.content or "").strip()
    report = parse_final_report(content)
    return report, tool_log, total_input, total_output

def run_opensource_loop(vllm_client, model, messages, repo_tools, file_name):
    tool_log = []
    total_input = 0
    total_output = 0
    last_tool_call = None
    tools_recently_added = 0
    token_budget = int(RUNPOD_MODEL_CONTEXT_LIMITS.get(model, VLLM_CONTEXT_LIMIT) * 0.9)

    for iteration in range(MAX_TOOL_CALLS + 1):
        response = None
        print(f"[{model}] iteration {iteration}")
        if total_input >= token_budget:
            if tools_recently_added > 0:
                del messages[-tools_recently_added:]
            messages.append({
                "role": "user",
                "content": "Based on the information you have gathered so far, write your code review report now based on diff gived and results of tool calls. Output ONLY the JSON object following the schema in the system prompt."
            })
            return _force_final_review(
                vllm_client, model, messages, tool_log,
                total_input, total_output, file_name,
                f"token budget reached estimated tokens"
            )

        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "tools": TOOLS_OPENAI,
                "tool_choice": "auto",
                "max_tokens": 8000,
                "temperature": 0.0,
                "timeout": 480,
            }
            response = vllm_client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"exception trigered here: {e}\n\n")
            if "maximum context length" in str(e):
                print(f"[{model}] context exceeded on API call — removing last message and forcing review")
                if tools_recently_added > 0:
                    del messages[-tools_recently_added:]
                messages.append({
                    "role": "user",
                    "content": "Based on the information you have gathered so far, write your code review report now. Output ONLY the JSON object following the schema in the system prompt."
                })
                return _force_final_review(
                    vllm_client, model, messages, tool_log,
                    total_input, total_output, file_name,
                    "context length exceeded — last calltools called removed"
                )
            else:
                raise

        if response.usage:
            total_input  += response.usage.prompt_tokens or 0
            total_output += response.usage.completion_tokens or 0

        choice  = response.choices[0]
        message = choice.message

        messages.append(message.model_dump(exclude_unset=True))
        if choice.finish_reason == "tool_calls" or (choice.finish_reason == "stop" and message.tool_calls):    
            for tc in message.tool_calls:
                name   = tc.function.name
                args   = json.loads(tc.function.arguments or "{}")
                current_call = (name, json.dumps(args, sort_keys=True))
                if current_call == last_tool_call:
                    print(f"[{model}] duplicate tool call detected ({name})")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "You already called this tool with the same arguments."}),
                    })
                    continue                    
                
                last_tool_call = current_call
                print(f"[{model}] tool: {name}({args})")
                tool_log.append({"tool": name, "args": args, "iteration": iteration})
                result = repo_tools.dispatch(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            tools_recently_added = len(message.tool_calls)
            continue

        content = (message.content or "").strip()
        print(f"[{model}] done — {iteration} iterations, {len(tool_log)} tool calls | input={total_input} output={total_output} tokens")
        save_messages_debug(file_name, model, messages)
        try:
            report = parse_final_report(content)
            return report, tool_log, total_input, total_output
        except json.JSONDecodeError:
            print(f"[{model}] response was not valid JSON report, sending correction message")
            messages.append({
                "role": "user",
                "content": "Your response must be a valid JSON object only. No markdown, no explanation. Output ONLY the JSON following the schema provided in the system prompt with keys if you finisch review and tool calls.",
            })
            continue

    raise RuntimeError(f"[{model}] exceeded {MAX_TOOL_CALLS} tool calls")

def _extract_report_json(text: str) -> dict | None:
    text = text.replace('\u0120', ' ').replace('\u010a', '\n').replace('\u010A', '\n')
    text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")

    start_match = re.search(r'\{\s*"findings"', text)
    if not start_match:
        return None

    end_match = re.search(r'"verdict"\s*:\s*"[^"]*"\s*\}', text[start_match.start():])
    if not end_match:
        return None

    end = start_match.start() + end_match.end()
    chunk = text[start_match.start():end]

    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        pass

    def fix_string_values(s):
        result = []
        i = 0
        while i < len(s):
            if s[i] == '"':
                result.append('"')
                i += 1
                while i < len(s):
                    if s[i] == '\\' and i + 1 < len(s):
                        next_char = s[i+1]
                        if next_char in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'):
                            result.append(s[i])
                            result.append(next_char)
                            i += 2
                        else:
                            result.append('\\\\')
                            i += 1
                    elif s[i] == '"':
                        rest = s[i+1:].lstrip()
                        if rest and rest[0] in ':,]}\n\r ':
                            result.append('"')
                            i += 1
                            break
                        else:
                            result.append('\\"')
                            i += 1
                    else:
                        result.append(s[i])
                        i += 1
            else:
                result.append(s[i])
                i += 1
        return ''.join(result)

    try:
        return json.loads(fix_string_values(chunk))
    except json.JSONDecodeError:
        return None
    
def parse_json_response(text: str) -> dict:
    text = (text or "").strip()

    result = _extract_report_json(text)
    if result is not None:
        return result
    print(f"[extract_report_json] not working check the report: \n{text}\n")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # handles: ```json\n{...}``` or ``` json\n{...}```
    fence_match = re.search(r"```\s*(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError as jserr:
            print(f"[parse_json_response] - fence_match: json decode error - {jserr}\n -- \n{text}\n")
            pass

    # handles: narrative text before/after JSON by scanning from the last } valid object, then comming back to first {, that matches json format.
    last_brace = text.rfind("}")
    if last_brace != -1:
        first_brace = text.rfind("{", 0, last_brace)
        while first_brace != -1:
            candidate = text[first_brace:last_brace + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                first_brace = text.rfind("{", 0, first_brace)

    raise json.JSONDecodeError("No JSON found in model response", text, 0)


def run_openai_loop(client, model, messages, repo_tools, file_name):
    tool_log = []
    total_input  = 0
    total_cache_read = 0
    total_output = 0
    sleep_s = 0 

    for iteration in range(MAX_TOOL_CALLS + 1):
        print(f"[{model}] iteration {iteration}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_OPENAI,
            max_completion_tokens=4000,
            timeout=480,
        )

        if response.usage:
            cached = (response.usage.prompt_tokens_details.cached_tokens or 0) if response.usage.prompt_tokens_details else 0
            total_input += (response.usage.prompt_tokens or 0) - cached
            total_cache_read += cached
            total_output += response.usage.completion_tokens or 0

        choice  = response.choices[0]
        message = choice.message

        messages.append(message.model_dump(exclude_unset=True))

        if choice.finish_reason == "tool_calls":
            for tc in message.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                print(f"[{model}] tool: {name}({args})")
                tool_log.append({"tool": name, "args": args, "iteration": iteration})
                result = repo_tools.dispatch(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
                if iteration > 0 and iteration % SLEEP_AFTER_HOW_MUCH_ITERATIONS == 0:
                    time.sleep(60)
                    sleep_s += 60
            continue

        text = (message.content or "").strip()
        print(f"[{model}] done — {iteration} iterations, {len(tool_log)} tool calls | input={total_input} output={total_output} tokens")
        save_messages_debug(file_name, model, messages)
        return parse_final_report(text), tool_log, total_input, total_output, total_cache_read, sleep_s

    raise RuntimeError(f"[{model}] exceeded {MAX_TOOL_CALLS} tool calls")


def run_deepseek_loop(client, model, messages, repo_tools, file_name):
    tool_log = []
    total_input = 0
    total_cache_read = 0
    total_output = 0
    sleep_s = 0 

    for iteration in range(MAX_TOOL_CALLS + 1):
        print(f"[{model}] iteration {iteration}")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_OPENAI,
            max_tokens=4000,
            timeout=480)

        if response.usage:
            total_cache_read += response.usage.prompt_cache_hit_tokens or 0
            total_input += response.usage.prompt_cache_miss_tokens or 0
            total_output += response.usage.completion_tokens or 0

        choice  = response.choices[0]
        message = choice.message

        messages.append(message.model_dump(exclude_unset=True))

        if choice.finish_reason == "tool_calls":
            for tc in message.tool_calls:
                name   = tc.function.name
                args   = json.loads(tc.function.arguments or "{}")
                print(f"[{model}] tool: {name}({args})")
                tool_log.append({"tool": name, "args": args, "iteration": iteration})
                result = repo_tools.dispatch(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
                if iteration > 0 and iteration % SLEEP_AFTER_HOW_MUCH_ITERATIONS == 0:
                    time.sleep(60)
                    sleep_s += 60
            continue

        text = (message.content or "").strip()
        if not text:
            text = (getattr(message, "reasoning_content", None) or "").strip()
        if not text:
            raise RuntimeError(f"[{model}] model returned empty content and empty reasoning_content")
        try:
            result = parse_final_report(text)
            save_messages_debug(file_name, model, messages)
            return result, tool_log, total_input, total_output, total_cache_read, sleep_s
        except json.JSONDecodeError:
            print(f"[{model}] response was not JSON, sending correction message to try to output correct json answer")
            messages.append({
                "role": "user",
                "content": "Your response must be a valid JSON object only. No markdown, no explanation. Output ONLY the JSON following the schema provided. If you still need to call tools, do so. Only output the final JSON when you are completely done with tool calling.",
            })
            continue
    raise RuntimeError(f"[{model}] exceeded {MAX_TOOL_CALLS} tool calls")


def run_anthropic_loop(client, model, messages, repo_tools, file_name):
    tool_log = []
    total_input = 0   
    total_output = 0 
    total_cache_read = 0   
    total_cache_creation = 0
    sleep_s = 0 
    system_with_cache = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages[0] = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": messages[0]["content"],
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }

    for iteration in range(MAX_TOOL_CALLS + 1):
        print(f"[{model}] iteration {iteration}")
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=system_with_cache,
            tools=TOOLS_ANTHROPIC,
            messages=messages,
            temperature=0,
            service_tier="auto",
        )

        if response.usage:
            total_input += response.usage.input_tokens or 0
            total_output += response.usage.output_tokens or 0
            total_cache_read += getattr(response.usage, "cache_read_input_tokens",    0) or 0
            total_cache_creation += getattr(response.usage, "cache_creation_input_tokens", 0) or 0

        if response.stop_reason == "end_turn":
            text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            if not text_blocks:
                raise RuntimeError(f"[{model}] model returned end_turn with no text block")
            text = "\n".join(text_blocks)
            save_messages_debug(file_name, model, messages)
            return parse_final_report(text), tool_log, total_input, total_output, total_cache_read, total_cache_creation, sleep_s

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                name = block.name
                args = block.input or {}
                print(f"[{model}] tool: {name}({args})")
                tool_log.append({"tool": name, "args": args, "iteration": iteration})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": repo_tools.dispatch(name, args),
                })
            messages.append({"role": "user", "content": tool_results})
            if iteration > 0 and iteration % SLEEP_AFTER_HOW_MUCH_ITERATIONS == 0:
                time.sleep(60)
                sleep_s += 60
            continue

        raise RuntimeError(f"[{model}] unexpected stop_reason: {response.stop_reason}")

    raise RuntimeError(f"[{model}] exceeded {MAX_TOOL_CALLS} tool calls")


def review_with_model(provider, llm_client, model, user_prompt, repo_tools, file_name) -> dict:
    t0 = time.time()
    timed_out = [False]
    timer = Timer(REVIEW_TIMEOUT_S, lambda: timed_out.__setitem__(0, True))
    timer.start()

    try:
        if provider == "anthropic":
            messages = [{"role": "user", "content": user_prompt}]
            report, tool_log, total_input, total_output, cache_read, cache_creation, sleep_s = run_anthropic_loop(llm_client, model, messages, repo_tools, file_name)
        elif provider == "openai":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
            report, tool_log, total_input, total_output, cache_read, sleep_s = run_openai_loop(llm_client, model, messages, repo_tools, file_name)
            cache_creation = 0
        elif provider == "deepseek":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
            report, tool_log, total_input, total_output, cache_read, sleep_s = run_deepseek_loop(llm_client, model, messages, repo_tools, file_name)
            cache_creation = 0
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}]
            report, tool_log, total_input, total_output = run_opensource_loop(llm_client, model, messages, repo_tools, file_name)
            cache_read = 0
            cache_creation = 0
            sleep_s = 0
    except json.JSONDecodeError as e:
        return {"error": f"LLM returned invalid JSON: {e}"}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}
    finally:
        timer.cancel()

    if timed_out[0]:
        return {"error": f"Review timed out after {REVIEW_TIMEOUT_S}s"}

    wall_time_s = round(time.time() - t0, 2)
    latency = round(wall_time_s - sleep_s, 2)

    try:
        report = validate_report(report)
    except ValueError as e:
        return {"error": str(e)}
    
    true_input = total_input + cache_creation + cache_read

    if provider == "anthropic":
        price_input = ANTHROPIC_PRICE_INPUT_PER_M
        price_cache_read = ANTHROPIC_PRICE_CACHE_READ_PER_M
        price_cache_write= ANTHROPIC_PRICE_CACHE_CREATION_PER_M
        price_output = ANTHROPIC_PRICE_OUTPUT_PER_M
    elif provider == "openai":
        price_input = OPENAI_PRICE_INPUT_PER_M
        price_cache_read = OPENAI_PRICE_CACHE_READ_PER_M 
        price_cache_write= 0
        price_output = OPENAI_PRICE_OUTPUT_PER_M
    elif provider == "deepseek":
        price_input = DEEPSEEK_PRICE_INPUT_PER_M
        price_cache_read = DEEPSEEK_PRICE_CACHE_HIT_PER_M
        price_cache_write = 0
        price_output = DEEPSEEK_PRICE_OUTPUT_PER_M
    else:
        price_input = 0
        price_cache_read = 0
        price_cache_write= 0
        price_output = 0

    estimated_cost_usd = round(
        (total_input / 1_000_000) * price_input +
        (cache_creation / 1_000_000) * price_cache_write +
        (cache_read / 1_000_000) * price_cache_read +
        (total_output / 1_000_000) * price_output,
        6)

    token_info = {
        "input_tokens": total_input,
        "cache_creation_tokens": cache_creation,
        "cache_read_tokens": cache_read,
        "output_tokens": total_output,
        "true_input_tokens": true_input,
        "total_tokens": true_input + total_output,
        "estimated_cost_usd": estimated_cost_usd,
        "pricing": {
            "input_per_m": price_input,
            "cache_creation_per_m": price_cache_write,
            "cache_read_per_m": price_cache_read,
            "output_per_m": price_output,
            "model": model,
        },
    }

    report["_meta"] = {
        "latency_s": latency,
        "sleep_s": sleep_s,
        "wall_time_s": wall_time_s,
        "tool_calls": tool_log,
        "tool_count": len(tool_log),
        "tokens": token_info,
    }
    return report

def create_app(provider: str) -> Flask:
    app = Flask(__name__)

    if provider == "commercial":
        if _anthropic is None:
            raise RuntimeError("pip install anthropic")
        if OpenAI is None:
            raise RuntimeError("pip install openai")

        anthropic_client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        openai_client    = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL) if OPENAI_API_KEY else None
        deepseek_client  = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL) if DEEPSEEK_API_KEY else None

        clients = []
        for entry in COMMERCIAL_MODELS:
            p, m = entry["provider"], entry["model"]
            if p == "anthropic":
                if anthropic_client is None:
                    raise RuntimeError("ANTHROPIC_API_KEY is not set")
                clients.append({"provider": p, "model": m, "client": anthropic_client})
            elif p == "openai":
                if openai_client is None:
                    raise RuntimeError("OPENAI_API_KEY is not set")
                clients.append({"provider": p, "model": m, "client": openai_client})
            elif p == "deepseek":
                if deepseek_client is None:
                    raise RuntimeError("DEEPSEEK_API_KEY is not set")
                clients.append({"provider": p, "model": m, "client": deepseek_client})
            else:
                raise ValueError(f"Unknown provider in COMMERCIAL_MODELS: {p!r}")

        if not clients:
            raise RuntimeError("COMMERCIAL_MODELS is empty — uncomment at least one model")

    elif provider == "opensource":
        vllm_client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_BASE_URL)
        clients = [
            {"provider": "opensource", "model": model, "client": vllm_client}
            for model in RUNPOD_MODELS
        ]
        if not clients:
            raise RuntimeError("RUNPOD_MODELS is empty — add at least one model name")
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose 'commercial' or 'opensource'.")


    @app.post("/review")
    def review():
        pr = request.get_json(force=True, silent=True)
        if not pr:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        for field in ("title", "diff", "baseRefOid", "fileName", "category"):
            if not pr.get(field):
                return jsonify({"error": f"Missing required field: '{field}'"}), 400

        file_name = pr["fileName"]
        repo_name = file_name.rsplit("_pr_", 1)[0]

        def _report_path_for(model_name: str) -> str:
            safe = re.sub(r"[:/\\]", "_", model_name)
            return os.path.join(REPORTS_DIR, safe, pr["category"], f"{file_name}.json")

        clients_to_run = []
        for entry in clients:
            p = _report_path_for(entry["model"])
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        existing = json.load(f)
                    if "error" not in existing:
                        print(f"[{entry['model']}] already reviewed, skipping: {p}")
                        continue
                except Exception:
                    pass
            clients_to_run.append(entry)

        if not clients_to_run:
            print(f"All models already reviewed for {file_name}, skipping")
            return jsonify({"skipped": True, "fileName": file_name})

        try:
            repo_path = prepare_repo(repo_name, pr["baseRefOid"])
            if repo_path is None:
                return jsonify({"skipped": True, "reason": "git checkout failed", "fileName": file_name}), 200
            repo_tools = RepoTools(repo_path)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode().strip() if e.stderr else ""
            return jsonify({"error": f"git checkout failed: {stderr}"}), 500

        user_prompt = build_user_prompt(pr)

        all_reports = {}
        saved_paths = {}
        save_errors = {}

        for entry in clients_to_run:
            model = entry["model"]
            llm_client = entry["client"]
            entry_provider = entry.get("provider", provider)
            print(f"Starting review — {entry_provider}/{model} | repo={repo_name}/{file_name} | commit={pr['baseRefOid'][:8]}")
            report = review_with_model(entry_provider, llm_client, model, user_prompt, repo_tools, file_name)
            all_reports[model] = report
            try:
                path = save_report(model, file_name, report, pr["category"])
                saved_paths[model] = path
                print(f"Report saved → {path}")
            except Exception as e:
                print(f"Failed to save report for {model}: {e}")
                save_errors[model] = str(e)

        response_body = {"reports": all_reports, "saved_to": saved_paths}
        if save_errors:
            response_body["save_errors"] = save_errors
        return jsonify(response_body)
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code review agent service")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["commercial", "opensource"],
        help="'commercial' → commercial models on port 5001 | 'opensource' → OSS models on port 5002",
    )
    parser.add_argument("--port", default=5001, type=int)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    app = create_app(args.provider)
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
