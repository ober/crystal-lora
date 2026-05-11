#!/usr/bin/env python3
"""
Build CPT corpus v2: existing local sources PLUS top Crystal repos from GitHub.

The original build_cpt_corpus.py only walks ~/mine/crystal and ~/mine/crystal-mcp,
yielding ~3 MB. To meaningfully fine-tune a 30B model we need ~50M+ tokens
(~200 MB of source). This script:

  1. Walks local sources (same as v1)
  2. `gh search repos --language=crystal --sort=stars --limit=N` for top GitHub repos
  3. Shallow-clones each, walks for .cr files
  4. SHA-dedups against existing corpus and across repos
  5. Filters: skip vendored, generated, tiny (<100B), huge (>256KB)
  6. Writes JSONL with {text: <file content>}

Usage:
  python3 build_cpt_corpus_v2.py --out cpt_corpus_v2.jsonl --limit 200
  python3 build_cpt_corpus_v2.py --skip-github   # local sources only
  python3 build_cpt_corpus_v2.py --keep-clones   # keep cloned repos for re-runs
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
CRYSTAL_SRC = Path.home() / "mine" / "crystal"
CRYSTAL_MCP = Path.home() / "mine" / "crystal-mcp"

CODE_EXT = ".cr"
DOC_EXT = ".md"

SKIP_DIRS_LOCAL = {
    ".git", ".github", ".circleci", "node_modules", "build", "dist",
    ".cache", "target", "lib", "compiler", "llvm", ".well-known",
}
# For GitHub repos: also skip vendored shards but keep specs (Crystal spec
# files teach the spec-framework idioms we want).
SKIP_DIRS_GITHUB = {
    ".git", ".github", "node_modules", "build", "dist", ".cache",
    "target", "lib", "vendor",
}

MIN_BYTES = 100
MAX_BYTES = 256 * 1024


def iter_local_files(root: Path, skip: set):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext == CODE_EXT or ext == DOC_EXT:
                yield p, ext


def fetch_repo_list(limit: int) -> list[str]:
    print(f"Fetching top {limit} Crystal repos via gh search...")
    r = subprocess.run(
        ["gh", "search", "repos", "--language=crystal", "--sort=stars",
         f"--limit={limit}", "--json", "fullName,stargazersCount"],
        capture_output=True, text=True, check=True,
    )
    repos = json.loads(r.stdout)
    repos.sort(key=lambda r: -r["stargazersCount"])
    print(f"  got {len(repos)} repos (top: {repos[0]['fullName']} {repos[0]['stargazersCount']}*)")
    return [r["fullName"] for r in repos]


def clone_repo(full_name: str, dest_root: Path) -> Path | None:
    safe = full_name.replace("/", "__")
    dest = dest_root / safe
    if dest.exists():
        return dest
    url = f"https://github.com/{full_name}.git"
    r = subprocess.run(
        ["git", "clone", "--depth=1", "--quiet", url, str(dest)],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        return None
    return dest


def write_file_to_corpus(path: Path, rel_label: str, ext: str,
                         seen_sha: set, fout, stats: dict) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    size = len(text.encode("utf-8"))
    if size < MIN_BYTES or size > MAX_BYTES or not text.strip():
        return False
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if sha in seen_sha:
        stats["dedup"] += 1
        return False
    seen_sha.add(sha)
    if ext == CODE_EXT:
        header = f"# FILE: {rel_label}\n"
        stats["code"] += 1
    else:
        header = f"<!-- FILE: {rel_label} -->\n"
        stats["doc"] += 1
    fout.write(json.dumps({"text": header + text}) + "\n")
    stats["bytes"] += size
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpt_corpus_v2.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--clones-dir", default="crystal_corpus_raw")
    ap.add_argument("--skip-github", action="store_true")
    ap.add_argument("--skip-local", action="store_true")
    ap.add_argument("--keep-clones", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="List repos that would be cloned and exit")
    args = ap.parse_args()

    out_path = REPO / args.out
    seen_sha = set()
    stats = {"code": 0, "doc": 0, "bytes": 0, "dedup": 0,
             "repos_ok": 0, "repos_fail": 0}

    if args.dry_run:
        repos = fetch_repo_list(args.limit)
        print("\n".join(repos))
        return

    with open(out_path, "w") as fout:
        # Local sources first
        if not args.skip_local:
            for root, label in [(CRYSTAL_SRC, "crystal"), (CRYSTAL_MCP, "crystal-mcp")]:
                if not root.exists():
                    print(f"  skipping {root} (not present)", file=sys.stderr)
                    continue
                print(f"Walking local: {root}")
                for path, ext in iter_local_files(root, SKIP_DIRS_LOCAL):
                    rel_label = f"{label}/{path.relative_to(root)}"
                    write_file_to_corpus(path, rel_label, ext, seen_sha, fout, stats)
                print(f"  cumulative: {stats['code']} code, {stats['doc']} doc, "
                      f"{stats['bytes']//1024} KB")

        # GitHub
        if not args.skip_github:
            clones_dir = REPO / args.clones_dir
            clones_dir.mkdir(exist_ok=True)
            try:
                repos = fetch_repo_list(args.limit)
            except subprocess.CalledProcessError as e:
                print(f"gh search failed: {e.stderr}", file=sys.stderr)
                sys.exit(1)

            for i, repo_name in enumerate(repos, 1):
                print(f"[{i:3d}/{len(repos)}] {repo_name}", flush=True)
                try:
                    local = clone_repo(repo_name, clones_dir)
                except subprocess.TimeoutExpired:
                    print("    TIMEOUT")
                    stats["repos_fail"] += 1
                    continue
                if local is None:
                    print("    clone failed")
                    stats["repos_fail"] += 1
                    continue
                stats["repos_ok"] += 1

                kept_in_repo = 0
                for dirpath, dirnames, filenames in os.walk(local):
                    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS_GITHUB]
                    for fn in filenames:
                        p = Path(dirpath) / fn
                        ext = p.suffix.lower()
                        if ext != CODE_EXT:
                            continue
                        rel_label = f"{repo_name}/{p.relative_to(local)}"
                        if write_file_to_corpus(p, rel_label, ext, seen_sha, fout, stats):
                            kept_in_repo += 1
                if kept_in_repo:
                    print(f"    +{kept_in_repo} files (cumulative {stats['bytes']//1024} KB)")

            if not args.keep_clones:
                print(f"\nCleaning up {clones_dir}...")
                shutil.rmtree(clones_dir, ignore_errors=True)

    print(f"\n=== Summary ===")
    print(f"  Code files:        {stats['code']}")
    print(f"  Doc files:         {stats['doc']}")
    print(f"  Total bytes:       {stats['bytes']:,} ({stats['bytes']//(1024*1024)} MB)")
    print(f"  Approx tokens:     {stats['bytes']//4:,}  (heuristic: 4 bytes/token)")
    print(f"  Dedup-skipped:     {stats['dedup']}")
    print(f"  GitHub repos OK:   {stats['repos_ok']}")
    print(f"  GitHub repos fail: {stats['repos_fail']}")
    print(f"  Output:            {out_path}")


if __name__ == "__main__":
    main()
