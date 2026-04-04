import requests
import json
import os
import time
import subprocess

GITHUB_TOKEN = "" 
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.diff"
}

REPOS = [
    {"owner": "appwrite",       "name": "appwrite"},
    {"owner": "dotnet",         "name": "aspnetcore"},
    {"owner": "umbraco",        "name": "Umbraco-CMS"},
    {"owner": "home-assistant", "name": "core"},
    {"owner": "django",         "name": "django"},
    {"owner": "elastic",        "name": "elasticsearch"},
    {"owner": "cockroachdb",    "name": "cockroach"},
    {"owner": "abpframework",   "name": "abp"},
    {"owner": "odoo",           "name": "odoo"},
    {"owner": "go-gitea",       "name": "gitea"},
]

BASE_DATA_DIR  = "/Users/yassine/Downloads/Master_Arbeit/Experiment/DataExtraction/Master_Arbeit_Data"
REPO_CLONE_DIR = os.path.join(BASE_DATA_DIR, "_clones")


GRAPHQL_QUERY_PRS = """
query($searchQuery: String!) {
  search(query: $searchQuery, type: ISSUE, first: 100) {
    nodes {
      ... on PullRequest {
        number
        title
        url
        baseRefOid
        headRefOid
        body
      }
    }
  }
}
"""

GRAPHQL_QUERY_COMMITS = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      commits(first: 100) {
        totalCount
        nodes {
          commit {
            oid
            message
            committedDate
          }
        }
      }
    }
  }
}
"""

GRAPHQL_QUERY_THREADS = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100) {
        totalCount
        nodes {
          isResolved
          comments(first: 100) {
            totalCount
            nodes {
              body
              path
              line
              createdAt
              diffHunk
              originalCommit { oid }
              commit { oid }
            }
          }
        }
      }
    }
  }
}
"""


FILTERS = "-author:app/dependabot -author:app/github-actions -label:dependencies -label:documentation"
def get_queries(owner, name):
    return {
        "Quality": [
            f'repo:{owner}/{name} is:pr is:merged label:bug {FILTERS}',
            f'repo:{owner}/{name} is:pr is:merged "fix crash" {FILTERS}',
            f'repo:{owner}/{name} is:pr is:merged "null pointer" {FILTERS}'
        ],
        "Security": [
            f'repo:{owner}/{name} is:pr is:merged label:security {FILTERS}',
            f'repo:{owner}/{name} is:pr is:merged "vulnerability" {FILTERS}',
            f'repo:{owner}/{name} is:pr is:merged "XSS" {FILTERS}'
        ],
        "Performance": [
            f'repo:{owner}/{name} is:pr is:merged label:performance {FILTERS}',
            f'repo:{owner}/{name} is:pr is:merged "optimize" {FILTERS}'
        ],
        "Refactor": [
            f'repo:{owner}/{name} is:pr is:merged label:refactor -label:bug -label:fix {FILTERS}',
            f'repo:{owner}/{name} is:pr is:merged "cleanup" -label:bug -label:fix {FILTERS}'
        ],
        "Testing": [
            f'repo:{owner}/{name} is:pr is:merged label:test {FILTERS}',
            f'repo:{owner}/{name} is:pr is:merged "flaky" {FILTERS}'
        ]
    }


def git_clone_or_update(owner, name):
    repo_url    = f"https://github.com/{owner}/{name}.git"
    target_path = os.path.join(REPO_CLONE_DIR, name)
    if not os.path.exists(target_path):
        subprocess.run(["git", "clone", repo_url, target_path], check=True, capture_output=True)
    else:
        subprocess.run(["git", "-C", target_path, "fetch"], check=True, capture_output=True)
    return target_path


def get_commit_diff(owner, name, sha):
    url = f"https://api.github.com/repos/{owner}/{name}/commits/{sha}"
    res = requests.get(url, headers=HEADERS)
    return res.text if res.status_code == 200 else "Diff unavailable"


def graphql_post(payload):
    res = requests.post(
        "https://api.github.com/graphql",
        json    = payload,
        headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"},
    )
    if res.status_code != 200:
        print(f"  HTTP {res.status_code}: {res.text[:200]}")
        return None
    body = res.json()
    if "errors" in body:
        for e in body["errors"]:
            print(f"  GraphQL error: {e.get('message')}")
        return None
    return body


def fetch_commits(owner, name, pr_number):
    body = graphql_post({
        "query":     GRAPHQL_QUERY_COMMITS,
        "variables": {"owner": owner, "name": name, "number": pr_number},
    })
    if not body:
        return None, 0
    obj = body.get("data", {}).get("repository", {}).get("pullRequest", {}).get("commits", {})
    return obj.get("nodes", []), obj.get("totalCount", 0)


def fetch_review_threads(owner, name, pr_number):
    body = graphql_post({
        "query":     GRAPHQL_QUERY_THREADS,
        "variables": {"owner": owner, "name": name, "number": pr_number},
    })
    if not body:
        return None, 0
    obj = body.get("data", {}).get("repository", {}).get("pullRequest", {}).get("reviewThreads", {})
    return obj.get("nodes", []), obj.get("totalCount", 0)



def run_swr_extraction():
    os.makedirs(REPO_CLONE_DIR, exist_ok=True)
    summary = {}
    for repo in REPOS:
        owner, name = repo['owner'], repo['name']
        print(f"\n Starting Extraction: {owner}/{name}")
        summary[name] = {}
        local_repo_path  = git_clone_or_update(owner, name)
        repo_output_path = os.path.join(BASE_DATA_DIR, name)
        queries          = get_queries(owner, name)
        for category, query_list in queries.items():
            category_dir = os.path.join(repo_output_path, category)
            os.makedirs(category_dir, exist_ok=True)
            seen_in_category = set()

            for q in query_list:
                body = graphql_post({"query": GRAPHQL_QUERY_PRS, "variables": {"searchQuery": q}})
                if not body:
                    continue

                nodes = body.get("data", {}).get("search", {}).get("nodes", [])
                if not nodes:
                    continue

                for node in nodes:
                    if not node or node['number'] in seen_in_category:
                        continue

                    pr_number = node['number']

                    commit_nodes, commit_total = fetch_commits(owner, name, pr_number)
                    time.sleep(0.3)

                    if commit_nodes is None:
                        continue
                    if commit_total > 100:
                        continue

                    thread_nodes, thread_total = fetch_review_threads(owner, name, pr_number)
                    time.sleep(0.3)

                    if thread_nodes is None:
                        continue
                    if thread_total > 100:
                        continue

                    node['commits']       = {"totalCount": commit_total, "nodes": commit_nodes}
                    node['reviewThreads'] = {"totalCount": thread_total, "nodes": thread_nodes}

                    commit_diffs = {}
                    for c_node in commit_nodes:
                        sha = c_node['commit']['oid']
                        commit_diffs[sha] = get_commit_diff(owner, name, sha)
                        time.sleep(0.2)
                    node['individual_commit_diffs']   = commit_diffs
                    node['snapshot_checkout_command'] = (
                        f"git -C {local_repo_path} checkout {node['baseRefOid']}"
                    )

                    file_path = os.path.join(category_dir, f"pr_{pr_number}.json")
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(node, f, indent=4)

                    seen_in_category.add(pr_number)
                    time.sleep(0.5)

            count = len(seen_in_category)
            summary[name][category] = count
            print(f"  {name} - {category} {count} PRs saved")


if __name__ == "__main__":
    run_swr_extraction()
    print("\n DATA COLLECTION COMPLETE ")
