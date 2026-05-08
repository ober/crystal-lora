#!/usr/bin/env python3
"""
Build the continued-pretraining corpus from the Crystal source tree.

Walks ~/mine/crystal for .cr / .md files, emits one JSONL line per file
in the form {"text": "..."} suitable for axolotl `type: completion`.

This is Stage 1 of the staged pipeline: teach the model the token
distribution of Crystal source before any SFT Q/A training.

Usage:
  python3 build_cpt_corpus.py
  → cpt_corpus.jsonl  (alongside this script)
"""

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
CRYSTAL_SRC = Path.home() / "mine" / "crystal"
CRYSTAL_MCP = Path.home() / "mine" / "crystal-mcp"
OUT = REPO / "cpt_corpus.jsonl"

CODE_EXTS = {".cr"}
DOC_EXTS = {".md"}
# Skip third-party shards, the compiler internals (very Crystal-specific
# AST/codegen — not idiomatic Crystal user code), LLVM bindings, and the
# usual VCS/build dirs.
SKIP_DIRS = {
    ".git", ".github", ".circleci",
    "node_modules", "build", "dist", ".cache", "target",
    "lib",        # vendored shards under crystal/lib/
    "compiler",   # crystal/src/compiler — internal AST/codegen
    "llvm",       # crystal/src/llvm — LLVM-C bindings
    ".well-known",
}
MIN_BYTES = 64
MAX_BYTES = 256 * 1024


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext in CODE_EXTS or ext in DOC_EXTS:
                yield p, ext


def main():
    if not CRYSTAL_SRC.exists():
        sys.exit(f"error: {CRYSTAL_SRC} not found — clone the crystal repo first")

    n_code = 0
    n_doc = 0
    total_bytes = 0
    with open(OUT, "w") as out:
        for root, root_label in [(CRYSTAL_SRC, "crystal"), (CRYSTAL_MCP, "crystal-mcp")]:
            if not root.exists():
                print(f"  skipping {root} (not present)", file=sys.stderr)
                continue
            for path, ext in iter_files(root):
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                size = len(text.encode("utf-8"))
                if size < MIN_BYTES or size > MAX_BYTES:
                    continue
                rel = f"{root_label}/{path.relative_to(root)}"
                if ext in CODE_EXTS:
                    header = f"# FILE: {rel}\n"
                    n_code += 1
                else:
                    header = f"<!-- FILE: {rel} -->\n"
                    n_doc += 1
                out.write(json.dumps({"text": header + text}) + "\n")
                total_bytes += size

    print(f"Wrote {OUT}")
    print(f"  code files: {n_code}")
    print(f"  doc files:  {n_doc}")
    print(f"  total:      {(total_bytes / (1024 * 1024)):.1f} MB")


if __name__ == "__main__":
    main()
