#!/usr/bin/env python3
"""Harvest official Crystal docs into JSONL for CPT.

Sources (must already be cloned to /tmp/<repo_underscored>):
  - crystal-lang/crystal-book      (the Crystal book — markdown)
  - crystal-lang/crystal-website   (release notes, blog posts, RFC excerpts)
  - crystal-lang/rfcs              (language proposals)
  - veelenga/awesome-crystal       (curated list)

Also walks ~/mine/crystal/src/ for the stdlib (.cr with doc comments).
SHA-dedup within this run; the merge step dedups against the GitHub corpus.
"""

import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
CRYSTAL_STDLIB = Path.home() / "mine" / "crystal" / "src"

DOC_SOURCES = [
    ("/tmp/crystal-book",                    "crystal-book",   {".md"}),
    ("/tmp/crystal-lang_crystal-website",    "crystal-website",{".md"}),
    ("/tmp/crystal-lang_rfcs",               "crystal-rfcs",   {".md"}),
    ("/tmp/veelenga_awesome-crystal",        "awesome-crystal",{".md"}),
]

SKIP_DIRS = {".git", ".github", "node_modules", "build", "dist",
             "_site", "vendor", "assets", "_sass", "_layouts"}

MIN_BYTES = 100
MAX_BYTES = 256 * 1024


def harvest(root: Path, label: str, exts: set, seen: set, fout, stats: dict):
    if not root.exists():
        print(f"  SKIP {root} (not cloned)")
        return
    kept = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() not in exts:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            size = len(text.encode("utf-8"))
            if size < MIN_BYTES or size > MAX_BYTES or not text.strip():
                continue
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if sha in seen:
                stats["dedup"] += 1
                continue
            seen.add(sha)
            ext = p.suffix.lower()
            rel = p.relative_to(root)
            if ext == ".cr":
                header = f"# FILE: {label}/{rel}\n"
                stats["code"] += 1
            else:
                header = f"<!-- FILE: {label}/{rel} -->\n"
                stats["doc"] += 1
            fout.write(json.dumps({"text": header + text}) + "\n")
            stats["bytes"] += size
            kept += 1
    print(f"  {label}: +{kept} files (cumulative {stats['bytes']//1024} KB)")


def main():
    out = REPO / "cpt_docs.jsonl"
    seen = set()
    stats = {"code": 0, "doc": 0, "bytes": 0, "dedup": 0}

    with open(out, "w") as fout:
        for src, label, exts in DOC_SOURCES:
            print(f"Harvesting {label} from {src}")
            harvest(Path(src), label, exts, seen, fout, stats)

        if CRYSTAL_STDLIB.exists():
            print(f"Harvesting stdlib from {CRYSTAL_STDLIB}")
            harvest(CRYSTAL_STDLIB, "crystal-stdlib", {".cr"}, seen, fout, stats)

    print(f"\n=== Summary ===")
    print(f"  Code files: {stats['code']}")
    print(f"  Doc files:  {stats['doc']}")
    print(f"  Bytes:      {stats['bytes']:,} ({stats['bytes']//1024} KB)")
    print(f"  Approx tok: {stats['bytes']//4:,}")
    print(f"  Dedup:      {stats['dedup']}")
    print(f"  Output:     {out}")


if __name__ == "__main__":
    main()
