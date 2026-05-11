#!/usr/bin/env python3
"""LLM-augment SFT: have Claude Haiku 4.5 (via OpenRouter) generate Q/A pairs
from real Crystal source files in cpt_corpus_v3, then compile-gate every code
block in the answers.

Reads OPENROUTER_API_KEY from ~/.hermes/.env (slightly malformed format with
'sk-or-v1-' prefix grepped out) or env.

Cost estimate: ~5K source files × 1 prompt × ~1.5K input + ~600 output tokens
  ≈ 7.5M input + 3M output tokens
  ≈ $6 + $12 ≈ $18 (Haiku 4.5)

Compile-gate ensures we don't imprint Claude's mistakes onto the trained model.

Usage:
  python3 build_sft_llm.py --files 5000 --out sft_v3_llm.jsonl
  python3 build_sft_llm.py --files 100 --dry-run    # preview prompts
"""

import argparse
import asyncio
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parent
SYSTEM_FOR_TRAINING = (
    "You are an expert in Crystal, a statically-typed, Ruby-syntax-inspired language "
    "that compiles to native code via LLVM. You provide accurate, idiomatic Crystal code "
    "with correct types, `require` statements, and method signatures. Crystal looks like "
    "Ruby but is type-checked at compile time: no `method_missing`, no `eval`, methods "
    "overload on argument types, unions are first-class (`String | Nil`, abbreviated "
    "`String?`), generics use parens (`Array(Int32)`, `Hash(String, Int32)`), and macros "
    "run at compile time. Use `getter`/`setter`/`property` instead of Ruby's `attr_*`. "
    "Concurrency uses fibers and channels (`spawn`, `Channel(T)`). The standard test "
    "framework is the built-in `Spec` (via `require \"spec\"`), not RSpec. When writing "
    "code, always include the needed `require` statements."
)

GENERATOR_PROMPT = """You are generating training Q/A pairs for a Crystal language model. Read the Crystal source below and produce 2 Q/A pairs about it. Each pair must be:

  - A natural, focused question a developer would ask (NOT "what does this file do" — be specific to a method, idiom, or pattern in the file)
  - A precise answer that includes a SHORT, SELF-CONTAINED Crystal code example (≤ 30 lines) that COMPILES on its own (uses only stdlib unless `require` is shown)
  - The answer must show idiomatic Crystal — proper types, `getter`/`property` not `attr_*`, no Ruby-isms

Output STRICT JSON with this exact shape (no markdown, no preamble):
{"pairs": [{"q": "...", "a": "..."}, {"q": "...", "a": "..."}]}

The `a` field must be markdown including a ```crystal fenced code block.

Source file: %FILENAME%
```crystal
%CODE%
```"""


def load_api_key() -> str:
    if k := os.environ.get("OPENROUTER_API_KEY"):
        return k
    env_file = Path.home() / ".hermes" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            m = re.search(r'sk-or-v1-[a-zA-Z0-9]+', line)
            if m:
                return m.group(0)
    raise SystemExit("OPENROUTER_API_KEY not found")


def call_openrouter(api_key: str, user_prompt: str, max_retries: int = 3) -> str | None:
    body = json.dumps({
        "model": "anthropic/claude-haiku-4.5",
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": 1500,
        "temperature": 0.6,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                obj = json.loads(r.read())
            return obj["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                ConnectionError, KeyError, json.JSONDecodeError) as e:
            wait = 2 ** attempt
            print(f"    api error attempt {attempt+1}: {e}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
        except Exception as e:
            print(f"    unexpected api error attempt {attempt+1}: {type(e).__name__}: {e}; sleeping {2**attempt}s", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def parse_pairs(content: str) -> list[dict]:
    # Strip leading markdown / preamble if any
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*\n", "", content)
        content = re.sub(r"\n```\s*$", "", content)
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract a JSON object substring
        m = re.search(r"\{.*\"pairs\".*\}", content, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    pairs = obj.get("pairs", [])
    out = []
    for p in pairs:
        q = (p.get("q") or "").strip()
        a = (p.get("a") or "").strip()
        if len(q) > 10 and len(a) > 30:
            out.append({"q": q, "a": a})
    return out


def extract_code(answer_md: str) -> str | None:
    m = re.search(r"```(?:crystal|cr)\s*\n(.*?)```", answer_md, re.DOTALL)
    return m.group(1).strip() if m else None


_MISSING_SHARD = re.compile(r"can't find file '[^']+'", re.IGNORECASE)


def compile_check(code: str, timeout: int = 8) -> bool:
    """Compile-gate the code. Accepts 'can't find file' (missing third-party shard)
    as PASS since it's a dependency issue, not a syntax/type issue."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cr", delete=False) as f:
            f.write(code)
            tmp = f.name
        r = subprocess.run(["crystal", "build", "--no-codegen", tmp],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return True
        # Soft-pass: missing shard is the only error
        err = (r.stderr + r.stdout)
        if _MISSING_SHARD.search(err) and not re.search(r"\b(syntax error|undefined method|undefined constant|expected.*but)\b", err):
            return True
        return False
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


def make_chat(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_FOR_TRAINING},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def sample_corpus_files(corpus_path: Path, n: int) -> list[tuple[str, str]]:
    """Read cpt_corpus_v3_merged.jsonl and sample N entries that look like real Crystal code."""
    rng = random.Random(42)
    candidates = []
    with corpus_path.open() as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            # Skip docs (markdown). Pick code files with reasonable size.
            if not text.startswith("# FILE:"):
                continue
            first_line, _, body = text.partition("\n")
            if 200 < len(body) < 6000:
                fname = first_line.replace("# FILE: ", "").strip()
                candidates.append((fname, body))
    print(f"Corpus has {len(candidates)} eligible code files (200–6000 B). Sampling {n}.")
    rng.shuffle(candidates)
    return candidates[:n]


def process_one(args):
    """Worker: call API, parse, compile-check, return list of validated pairs.
    Catches *all* exceptions so a single worker failure can't kill the pool."""
    try:
        api_key, fname, code = args
        prompt = (GENERATOR_PROMPT
                  .replace("%FILENAME%", fname)
                  .replace("%CODE%", code))
        resp = call_openrouter(api_key, prompt)
        if not resp:
            return []
        pairs = parse_pairs(resp)
        out = []
        for p in pairs:
            cb = extract_code(p["a"])
            if cb and compile_check(cb):
                out.append((p["q"], p["a"]))
        return out
    except Exception as e:
        print(f"    worker error: {type(e).__name__}: {e}", file=sys.stderr)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=2000)
    ap.add_argument("--corpus", default="cpt_corpus_v3_merged.jsonl")
    ap.add_argument("--out",    default="sft_v3_llm.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = load_api_key()
    corpus_path = REPO / args.corpus
    sampled = sample_corpus_files(corpus_path, args.files)
    if not sampled:
        sys.exit("no candidate files found")

    if args.dry_run:
        print("First 3 prompts that would be sent:\n")
        for fname, code in sampled[:3]:
            prompt = (GENERATOR_PROMPT
                      .replace("%FILENAME%", fname)
                      .replace("%CODE%", code[:600] + "…"))
            print("=" * 60)
            print(prompt[:800])
        return

    inputs = [(api_key, fn, code) for fn, code in sampled]
    out_path = REPO / args.out
    n_pairs = n_valid = n_failed = 0
    seen = set()

    with mp.Pool(processes=args.workers) as pool, out_path.open("w") as fout:
        for i, pairs in enumerate(pool.imap_unordered(process_one, inputs, chunksize=1)):
            n_pairs += 2  # we asked for 2 per file; pairs may be 0..2
            for q, a in pairs:
                sha = hashlib.sha256(q.encode()).hexdigest()[:16]
                if sha in seen:
                    continue
                seen.add(sha)
                fout.write(json.dumps(make_chat(q, a)) + "\n")
                n_valid += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(inputs)}] kept={n_valid} this-batch={len(pairs)}", flush=True)

    print(f"\n=== Summary ===")
    print(f"  Files processed: {len(inputs)}")
    print(f"  Pairs requested: ~{n_pairs}")
    print(f"  Pairs validated: {n_valid}")
    print(f"  Output:          {out_path}")


if __name__ == "__main__":
    main()
