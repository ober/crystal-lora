#!/usr/bin/env python3
"""
Eval harness: compare a trained Crystal model against the untrained base
on Crystal-specific Ruby→Crystal divergence points.

Each prompt has a small set of `must_have` substrings (positive signals — Crystal
idioms we want to see) and `must_not` substrings (Ruby-isms we want suppressed).
Score per response = (must_have hits) - (must_not hits). Sum across prompts.

Usage:
    python3 eval_crystal.py --models jaimef/crystal-qwen3.6-30b qwen3-coder:30b
"""

import argparse
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field


@dataclass
class Prompt:
    name: str
    prompt: str
    must_have: list[str] = field(default_factory=list)   # Crystal idioms (positive)
    must_not: list[str] = field(default_factory=list)    # Ruby-isms (negative)
    case_sensitive: bool = True


# Each prompt targets a specific Ruby→Crystal divergence point.
# `must_have` = idiomatic Crystal patterns we want to see.
# `must_not` = Ruby-isms that should be absent.
PROMPTS: list[Prompt] = [
    Prompt(
        name="getter_setter",
        prompt="Write a Crystal class `Person` with a `name : String` accessor (read+write). Just the class, no main.",
        must_have=["property name"],
        must_not=["attr_accessor", "attr_reader", "attr_writer"],
    ),
    Prompt(
        name="generics_paren",
        prompt="Write a Crystal class `Stack(T)` with push, pop, and peek methods. Just the class.",
        must_have=["class Stack(T)", "Array(T)"],
        must_not=["Stack<T>", "Array<T>"],
    ),
    Prompt(
        name="union_type",
        prompt="Write a Crystal method `find_user` that takes an id : Int32 and returns either a User instance or nil. Show the method signature and a stub body.",
        must_have=["User?"],   # Crystal's nil-union shorthand. (User | Nil) also acceptable but we check primary form.
        must_not=["Optional", "Maybe"],
    ),
    Prompt(
        name="concurrency_spawn",
        prompt="Write Crystal code that runs three concurrent computations and collects their results via a channel. Just the code, no explanation.",
        must_have=["spawn", "Channel"],
        must_not=["Thread.new", "Thread.start", "Mutex.new"],
    ),
    Prompt(
        name="spec_framework",
        prompt="Write a Crystal spec file that tests an `add(a, b)` method. Use the standard Crystal test framework. Just the code.",
        must_have=['require "spec"', "describe", "it "],
        must_not=['require "rspec"', "RSpec.describe", "expect("],
    ),
    Prompt(
        name="hash_literal_typed",
        prompt="In Crystal, write code that creates an empty Hash mapping String to Int32, then adds three entries.",
        must_have=["Hash(String, Int32)"],
        must_not=["Hash<String, Int32>", "Hash<String,Int32>"],
    ),
    Prompt(
        name="block_yield",
        prompt="Write a Crystal method `each_squared(arr : Array(Int32))` that yields each element's square. Use Crystal's block syntax.",
        must_have=["yield"],
        must_not=["&block", "block.call", "block_given?", "Proc.new"],
    ),
    Prompt(
        name="exception_class",
        prompt="Write a Crystal exception subclass `ParseError` and a method that raises it. Just the code.",
        must_have=["class ParseError < Exception", "raise ParseError"],
        must_not=["StandardError", "RuntimeError.new"],
    ),
    Prompt(
        name="no_method_missing",
        prompt="In Crystal, how do you implement dynamic method dispatch like Ruby's method_missing? Give a one-paragraph answer.",
        must_have=["macro", "compile time"],
        must_not=["method_missing"],
        case_sensitive=False,
    ),
    Prompt(
        name="instance_var_typed",
        prompt="Write a Crystal class `Counter` with an `@count : Int32` initialized to 0, and a method `increment` that adds 1. Just the class.",
        must_have=["@count : Int32", "@count = 0"],
        must_not=["@count = nil", "attr_"],
    ),
]


def run_ollama(model: str, prompt: str, timeout: int = 180) -> str:
    """Invoke `ollama run MODEL` with prompt on stdin, capture stdout."""
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return f"<ERROR rc={proc.returncode}>\n{proc.stderr}"
    # Strip ANSI escape sequences (ollama spinner garbage)
    ansi = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\[\?2026[hl]|\[\?25[hl]|\[\?[0-9]+[hl]")
    cleaned = ansi.sub("", proc.stdout)
    cleaned = re.sub(r"\[K", "", cleaned)
    cleaned = re.sub(r"\[1G", "", cleaned)
    cleaned = re.sub(r"\[A", "", cleaned)
    cleaned = re.sub(r"\[2K", "", cleaned)
    return cleaned.strip()


def score_response(p: Prompt, text: str) -> tuple[int, dict]:
    """Return (score, detail). Score = #must_have hit - #must_not hit."""
    haystack = text if p.case_sensitive else text.lower()
    detail = {"have_hits": [], "have_miss": [], "not_hits": [], "not_clean": []}
    score = 0
    for needle in p.must_have:
        n = needle if p.case_sensitive else needle.lower()
        if n in haystack:
            score += 1
            detail["have_hits"].append(needle)
        else:
            detail["have_miss"].append(needle)
    for needle in p.must_not:
        n = needle if p.case_sensitive else needle.lower()
        if n in haystack:
            score -= 1
            detail["not_hits"].append(needle)
        else:
            detail["not_clean"].append(needle)
    return score, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="Ollama model names")
    ap.add_argument("--out", default="eval_results.json", help="Save raw outputs to JSON")
    ap.add_argument("--show-text", action="store_true", help="Print full responses")
    args = ap.parse_args()

    results: dict = {m: {} for m in args.models}
    totals = {m: 0 for m in args.models}
    n_prompts = len(PROMPTS)

    print(f"Running {n_prompts} prompts × {len(args.models)} models = {n_prompts * len(args.models)} generations\n")

    for i, p in enumerate(PROMPTS, 1):
        print(f"[{i}/{n_prompts}] {p.name}")
        print(f"  prompt: {p.prompt[:80]}...")
        for m in args.models:
            try:
                text = run_ollama(m, p.prompt)
            except subprocess.TimeoutExpired:
                text = "<TIMEOUT>"
            score, detail = score_response(p, text)
            totals[m] += score
            results[m][p.name] = {"score": score, "detail": detail, "text": text}
            print(f"  {m:38s}  score={score:+d}  have={detail['have_hits']}  ruby_isms={detail['not_hits']}")
            if args.show_text:
                print("    ---")
                print(textwrap.indent(text[:600], "    "))
                print("    ---")
        print()

    print("=" * 70)
    print("FINAL TOTALS")
    print("=" * 70)
    for m in args.models:
        max_score = sum(len(p.must_have) + len(p.must_not) for p in PROMPTS)
        print(f"  {m:40s}  {totals[m]:+d} / {max_score} possible")

    with open(args.out, "w") as f:
        json.dump({"totals": totals, "results": results}, f, indent=2)
    print(f"\nSaved raw outputs to {args.out}")


if __name__ == "__main__":
    main()
