#!/usr/bin/env python3
"""Mine the Crystal stdlib + cloned repos for compile-validated SFT Q/A pairs.

No LLM calls — all pairs come from existing real Crystal sources:

  1. Stdlib doc comments preceding `def` with `# ```` example blocks
       → "How do I use X#method?" + docstring + example
  2. README files in cloned repos: "## Usage" + adjacent ```crystal blocks
       → "How do I use the {repo} library?"
  3. Spec files: `describe X do; it "does Y" do; ...end` patterns
       → "How can I test that X does Y?"

Compile-validates every code block via `crystal build --no-codegen`,
discards failures.

Output format matches training_data_together.jsonl: chat messages with the
existing system prompt.

Usage:
  python3 build_sft_v3.py --out sft_v3_mined.jsonl
  python3 build_sft_v3.py --out sft_v3_mined.jsonl --no-validate    # skip compile gate
  python3 build_sft_v3.py --merge-existing  # also concat training_data_together.jsonl
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
STDLIB = Path.home() / "mine" / "crystal" / "src"
CLONES = REPO / "crystal_corpus_raw"

SYSTEM = ("You are an expert in Crystal, a statically-typed, Ruby-syntax-inspired language "
          "that compiles to native code via LLVM. You provide accurate, idiomatic Crystal code "
          "with correct types, `require` statements, and method signatures. Crystal looks like "
          "Ruby but is type-checked at compile time: no `method_missing`, no `eval`, methods "
          "overload on argument types, unions are first-class (`String | Nil`, abbreviated "
          "`String?`), generics use parens (`Array(Int32)`, `Hash(String, Int32)`), and macros "
          "run at compile time. Use `getter`/`setter`/`property` instead of Ruby's `attr_*`. "
          "Concurrency uses fibers and channels (`spawn`, `Channel(T)`). The standard test "
          "framework is the built-in `Spec` (via `require \"spec\"`), not RSpec. When writing "
          "code, always include the needed `require` statements.")

# Regex: a contiguous block of `# ...` comment lines followed by a `def ...` line.
# The doc-block can include `# ```` ... `# ```` example blocks.
DOC_DEF = re.compile(
    r"((?:^[ \t]*#[^\n]*\n)+)([ \t]*def\s+([^\n(]+)(?:\([^\n]*\))?(?:[ \t]*:[ \t]*[^\n]+)?)",
    re.MULTILINE,
)

# Inside a doc block, find `# ```` ... `# ```` example blocks (with leading indent).
DOC_EXAMPLE = re.compile(
    r"^[ \t]*#[ \t]*```[a-z]*\n((?:[ \t]*#[^\n]*\n)+?)[ \t]*#[ \t]*```",
    re.MULTILINE,
)

QUESTION_TEMPLATES = [
    "What does Crystal's `{name}` do, and can you show an example?",
    "How do I use `{name}` from Crystal's `{module}`?",
    "Show me how `{name}` works in Crystal, with a code example.",
    "Explain `{name}` from the Crystal `{module}` and give an example.",
    "How do I call `{name}` in Crystal?",
]


def strip_doc_comment(block: str) -> str:
    """Strip leading `# ` from each line of a Crystal doc block."""
    out = []
    for line in block.splitlines():
        s = line.lstrip()
        if s.startswith("# "):
            out.append(s[2:])
        elif s.startswith("#"):
            out.append(s[1:].lstrip())
        else:
            out.append("")
    return "\n".join(out).strip()


def extract_example_code(doc_block: str) -> str | None:
    m = DOC_EXAMPLE.search(doc_block)
    if not m:
        return None
    raw = m.group(1)
    return strip_doc_comment(raw)


def module_label_from_path(p: Path) -> str:
    """`/.../crystal/src/io/buffered.cr` → `io/buffered`."""
    parts = p.relative_to(STDLIB).with_suffix("").parts
    return "/".join(parts) if parts else p.stem


def method_short_name(sig: str) -> str:
    """`def foo(x : Int32) : String` → `foo`."""
    sig = sig.strip()
    if sig.startswith("def "):
        sig = sig[4:]
    sig = sig.split("(")[0].split(":")[0].strip()
    sig = sig.replace("self.", "")
    return sig


def mine_stdlib(stats: dict):
    """Yield (question, doc, example_code, sig) tuples from stdlib doc-defs."""
    for path in sorted(STDLIB.rglob("*.cr")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        module = module_label_from_path(path)
        for m in DOC_DEF.finditer(text):
            doc_block, sig_line, sig_after_def = m.group(1), m.group(2), m.group(3)
            doc_text = strip_doc_comment(doc_block)
            if len(doc_text) < 30:
                stats["stdlib_skipped_short_doc"] += 1
                continue
            example = extract_example_code(doc_block)
            if not example:
                stats["stdlib_skipped_no_example"] += 1
                continue
            name = method_short_name(sig_after_def)
            if not name or name.startswith("_"):
                continue
            # Emit 2 question variants per def (different phrasings) for diversity
            base_idx = abs(hash(module + name)) % len(QUESTION_TEMPLATES)
            for offset in (0, 2):
                tmpl = QUESTION_TEMPLATES[(base_idx + offset) % len(QUESTION_TEMPLATES)]
                q = tmpl.format(name=name, module=module)
                stats["stdlib_candidates"] += 1
                yield q, doc_text, example, sig_line.strip()


def mine_readmes(stats: dict):
    """Yield (question, prose, code) from `## Usage`-like sections in repo READMEs."""
    if not CLONES.exists():
        return
    USAGE_HEADER = re.compile(
        r"^#{1,3}\s*(usage|example|getting started|quick start|how to use)",
        re.IGNORECASE | re.MULTILINE,
    )
    CODE_BLOCK = re.compile(r"```(?:crystal|cr)\n(.*?)```", re.DOTALL)
    for repo_dir in sorted(CLONES.iterdir()):
        if not repo_dir.is_dir():
            continue
        for readme_name in ("README.md", "Readme.md", "readme.md"):
            readme = repo_dir / readme_name
            if readme.exists():
                break
        else:
            continue
        try:
            text = readme.read_text(errors="replace")
        except OSError:
            continue
        # Find each Usage-like section, then grab the first crystal code block in it.
        headers = list(USAGE_HEADER.finditer(text))
        for i, h in enumerate(headers):
            section_start = h.end()
            section_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            section = text[section_start:section_end]
            cb = CODE_BLOCK.search(section)
            if not cb:
                continue
            code = cb.group(1).strip()
            if len(code) < 30 or len(code) > 4000:
                continue
            # Pre-code prose: up to first 400 chars of the section before the block
            prose = section[:cb.start()].strip()[:400]
            repo_name = repo_dir.name.replace("__", "/")
            q = f"How do I use the `{repo_name}` Crystal library? Show a usage example."
            stats["readme_candidates"] += 1
            yield q, prose, code, None


def mine_specs(stats: dict):
    """Yield Q/A pairs from spec files. Loose: nested `end`s defeat regex
    block-matching, so we scan for `it "..." do … end` blocks line-by-line
    using `do/end` depth tracking, and look for the nearest enclosing
    `describe Klass` for context.
    """
    if not CLONES.exists():
        return
    DESCRIBE_LINE = re.compile(r'describe\s+([A-Z][\w:]+(?:\.[\w]+|#\w+)?)')
    IT_OPEN = re.compile(r'^([ \t]*)it\s+"([^"\n]{4,120})"\s+do\s*$')

    for spec_path in CLONES.rglob("*_spec.cr"):
        try:
            text = spec_path.read_text(errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        last_describe = None
        i = 0
        while i < len(lines):
            line = lines[i]
            dm = DESCRIBE_LINE.search(line)
            if dm:
                last_describe = dm.group(1)
            it_m = IT_OPEN.match(line)
            if it_m:
                indent = it_m.group(1)
                title = it_m.group(2)
                # Collect block body until matching `end` at the same indent
                body_lines: list[str] = []
                depth = 1
                j = i + 1
                while j < len(lines):
                    bl = lines[j]
                    stripped = bl.strip()
                    # naive depth tracking on do/end keywords
                    if (re.search(r'\bdo\b\s*(\|[^|]*\|)?\s*$', stripped)
                            or re.match(r'^(class|module|def|begin|case|if|unless|while|until)\b', stripped)
                            or stripped.endswith(' do')):
                        depth += 1
                    if stripped == "end" or stripped.startswith("end "):
                        depth -= 1
                        if depth == 0:
                            break
                    body_lines.append(bl)
                    j += 1
                inner = "\n".join(body_lines).rstrip()
                i = j + 1
                if len(inner) < 20 or len(inner) > 2000 or not last_describe:
                    continue
                code = (
                    f'require "spec"\n\n'
                    f'describe {last_describe} do\n'
                    f'  it "{title}" do\n{inner}\n  end\nend'
                )
                q = f'How can I write a Crystal spec that asserts `{last_describe}` "{title}"?'
                stats["spec_candidates"] += 1
                yield q, None, code, None
            else:
                i += 1


def assistant_answer(doc: str | None, code: str, sig: str | None) -> str:
    """Compose a markdown assistant message: prose + sig + crystal code block."""
    parts = []
    if doc:
        parts.append(doc)
    if sig:
        parts.append(f"Signature:\n\n```crystal\n{sig}\nend\n```")
    parts.append(f"Example:\n\n```crystal\n{code}\n```")
    return "\n\n".join(parts)


def looks_like_full_program(code: str) -> bool:
    """Heuristic: code references a top-level def/class or `puts`/`p`/`pp`/`require` — likely runnable."""
    return bool(re.search(r"^\s*(class|module|def|puts|p |pp |require)", code, re.MULTILINE))


_MISSING_SHARD = re.compile(r"can't find file '[^']+'", re.IGNORECASE)


def compile_check(code: str, timeout: int = 10) -> bool:
    """Crystal-build the snippet. Accepts 'can't find file' (missing
    third-party shard) as PASS — it's a dependency issue, not a syntax issue."""
    if not code.strip():
        return False
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cr", delete=False) as f:
            f.write(code)
            tmp = f.name
        r = subprocess.run(
            ["crystal", "build", "--no-codegen", tmp],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0:
            return True
        err = r.stderr + r.stdout
        if _MISSING_SHARD.search(err) and not re.search(
            r"\b(syntax error|undefined method|undefined constant|expected.*but)\b", err
        ):
            return True
        return False
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


def _validate_one(args):
    idx, code = args
    return idx, compile_check(code, timeout=8)


def validate_parallel(items: list[tuple[str, str, dict]], procs: int) -> list[bool]:
    """Compile-check a list of (question, code, payload) in parallel.
    Returns a parallel list of bools."""
    if not items:
        return []
    inputs = [(i, code) for i, (_q, code, _p) in enumerate(items)]
    results = [False] * len(items)
    with mp.Pool(processes=procs) as pool:
        for done, (idx, ok) in enumerate(pool.imap_unordered(_validate_one, inputs, chunksize=4)):
            results[idx] = ok
            if (done + 1) % 200 == 0:
                print(f"    validated {done+1}/{len(items)}", flush=True)
    return results


def make_chat(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sft_v3_mined.jsonl")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip the compile gate (much faster, lower-quality output)")
    ap.add_argument("--merge-existing", action="store_true",
                    help="Also include training_data_together.jsonl in the output")
    ap.add_argument("--max-stdlib", type=int, default=10000)
    ap.add_argument("--max-readme", type=int, default=2000)
    ap.add_argument("--max-spec",   type=int, default=12000)
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 2),
                    help="Parallel compile-check workers")
    args = ap.parse_args()

    stats = {
        "stdlib_candidates": 0, "stdlib_skipped_short_doc": 0, "stdlib_skipped_no_example": 0,
        "stdlib_kept": 0, "stdlib_compile_fail": 0,
        "readme_candidates": 0, "readme_kept": 0, "readme_compile_fail": 0,
        "spec_candidates": 0, "spec_kept": 0, "spec_compile_fail": 0,
        "dedup": 0,
    }
    seen_q = set()
    out_path = REPO / args.out
    pairs: list[dict] = []

    def add(question, answer, source_key):
        sha = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        if sha in seen_q:
            stats["dedup"] += 1
            return
        seen_q.add(sha)
        pairs.append(make_chat(question, answer))
        stats[f"{source_key}_kept"] += 1

    def collect(label, gen, cap):
        items = []  # list of (question, code, payload)
        for q, doc, code, sig in gen:
            if len(items) >= cap:
                break
            items.append((q, code, (doc, sig)))
        print(f"  {label}: collected {len(items)} candidates", flush=True)
        return items

    print("Mining stdlib doc-comments…", flush=True)
    stdlib_items = collect("stdlib", mine_stdlib(stats), args.max_stdlib)
    print("Mining cloned-repo READMEs…", flush=True)
    readme_items = collect("readme", mine_readmes(stats), args.max_readme)
    print("Mining spec files…", flush=True)
    spec_items = collect("spec",   mine_specs(stats),   args.max_spec)

    if not args.no_validate:
        print(f"\nValidating in parallel with {args.procs} workers…", flush=True)
        for label, items in (("stdlib", stdlib_items), ("readme", readme_items), ("spec", spec_items)):
            print(f"  validating {label} ({len(items)})…", flush=True)
            ok = validate_parallel(items, args.procs)
            stats[f"{label}_compile_fail"] = ok.count(False)
            for idx, (q, code, payload) in enumerate(items):
                if not ok[idx]:
                    continue
                if label == "spec":
                    ans = f"```crystal\n{code}\n```"
                else:
                    doc, sig = payload
                    ans = assistant_answer(doc, code, sig)
                add(q, ans, label)
    else:
        # No-validate: keep everything
        for label, items in (("stdlib", stdlib_items), ("readme", readme_items), ("spec", spec_items)):
            for q, code, payload in items:
                if label == "spec":
                    ans = f"```crystal\n{code}\n```"
                else:
                    doc, sig = payload
                    ans = assistant_answer(doc, code, sig)
                add(q, ans, label)

    # Merge existing if requested
    if args.merge_existing:
        existing = REPO / "training_data_together.jsonl"
        n_existing = 0
        with existing.open() as f:
            for line in f:
                pairs.append(json.loads(line))
                n_existing += 1
        print(f"Merged {n_existing} existing SFT pairs.")

    # Write
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k:36s} {v}")
    print(f"\n  TOTAL pairs in {out_path.name}: {len(pairs)}")


if __name__ == "__main__":
    main()
