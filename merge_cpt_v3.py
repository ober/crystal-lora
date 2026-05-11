#!/usr/bin/env python3
"""Merge cpt_corpus_v3.jsonl + cpt_docs.jsonl into a single deduped file.

The v3 GitHub scrape clones crystal-lang/crystal which includes its src/
stdlib; the docs harvest also walks the local stdlib. Cross-file SHA
dedup prevents the stdlib appearing twice in training.
"""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
INPUTS = ["cpt_corpus_v3.jsonl", "cpt_docs.jsonl"]
OUTPUT = "cpt_corpus_v3_merged.jsonl"


def main():
    seen = set()
    n_kept = n_dup = 0
    out_path = REPO / OUTPUT
    with open(out_path, "w") as fout:
        for src in INPUTS:
            src_path = REPO / src
            kept = dup = 0
            with open(src_path) as f:
                for line in f:
                    obj = json.loads(line)
                    text = obj["text"]
                    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if sha in seen:
                        dup += 1
                        continue
                    seen.add(sha)
                    fout.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
            print(f"  {src}: kept {kept}, deduped {dup}")
            n_kept += kept
            n_dup += dup

    size = out_path.stat().st_size
    print(f"\n{OUTPUT}: {n_kept} examples, {n_dup} cross-file dedups")
    print(f"  size: {size:,} bytes ({size//(1024*1024)} MB)")
    print(f"  approx tokens (4 B/tok): {size//4:,}")


if __name__ == "__main__":
    main()
