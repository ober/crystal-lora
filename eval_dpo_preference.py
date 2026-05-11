#!/usr/bin/env python3
"""
Direct preference test: for each DPO instruction, run both models and check
whether the response uses Crystal idioms (chosen-style) or Ruby-isms (rejected-style).

Uses the ACTUAL training instructions (data leakage by design — we want to know
if the model can reproduce its own training data better than the base).

Score per response = (Crystal-idiom hits) - (Ruby-ism hits).
The MARGIN (trained - base) is the proof of training effect.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Ruby-isms vs Crystal idioms. Each pattern is (regex, list-of-models-it-favors).
# When a pattern hits, score +1 for "crystal" or "ruby".
CRYSTAL_PATTERNS = [
    r"\bproperty\s+\w+\s*:",         # property name : Type
    r"\bgetter\s+\w+\s*:",
    r"\bsetter\s+\w+\s*:",
    r"\bspawn\b",                    # spawn (concurrency)
    r"\bChannel\(",                  # Channel(T)
    r"\bArray\(",                    # Array(T) generics
    r"\bHash\(",                     # Hash(K, V)
    r"\bTuple\(",
    r"\bclass\s+\w+\(\w+\)",         # class Foo(T) generics
    r"\| Nil\b",                     # union with Nil
    r"\?\s*$",                       # nilable shorthand at line end
    r'require "spec"',               # standard test framework
    r"\.should\s+eq",                # spec assertion
    r"\bmacro\b",                    # macros (compile-time)
    r"\bcompile[- ]time\b",
    r"NamedTuple",
    r"\bof\s+\w+\b",                 # `[] of Int32`
    r"@\w+\s*:\s*\w+",               # typed instance var: @x : Int32
]

RUBY_PATTERNS = [
    r"\battr_accessor\b",
    r"\battr_reader\b",
    r"\battr_writer\b",
    r"\bThread\.new\b",
    r"\bThread\.start\b",
    r"\bMutex\.new\b",
    r"\bQueue\.new\b",
    r"Array<\w+>",                   # Java/C# style generics
    r"Hash<\w+",
    r"\bclass\s+\w+<\w+>",           # class Foo<T>
    r"\bmethod_missing\b",
    r'require ["\']rspec["\']',
    r"\bRSpec\.",
    r"\bexpect\(",                   # rspec assertion
    r"\.to eq\(",                    # rspec
    r"\bblock_given\?",
    r"\bProc\.new\b",
    r"&block\b",
    r"\bblock\.call\b",
    r"\bsuper_method\b",
    r"\bdefine_method\b",
    r"\beval\(",
    r"\binstance_eval\b",
    r"\bclass_eval\b",
    r"\b__send__\b",
]


def strip_ansi(s: str) -> str:
    """Strip ANSI escape sequences and ollama's spinner garbage."""
    s = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", s)
    s = re.sub(r"\[\?2026[hl]", "", s)
    s = re.sub(r"\[\?25[hl]", "", s)
    s = re.sub(r"\[\?[0-9]+[hl]", "", s)
    s = re.sub(r"\[K", "", s)
    s = re.sub(r"\[1G", "", s)
    s = re.sub(r"\[A", "", s)
    s = re.sub(r"\[2K", "", s)
    return s.strip()


def run_ollama(model: str, system: str, prompt: str, timeout: int = 120) -> str:
    """Use ollama API directly (more reliable than `ollama run` for system prompts)."""
    body = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 400, "num_ctx": 4096},
    }
    proc = subprocess.run(
        ["curl", "-s", "-X", "POST", "http://localhost:11434/api/generate",
         "-H", "Content-Type: application/json", "-d", json.dumps(body)],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        return f"<ERROR rc={proc.returncode}>"
    try:
        return json.loads(proc.stdout)["response"]
    except (json.JSONDecodeError, KeyError) as e:
        return f"<PARSE_ERROR {e}: {proc.stdout[:200]}>"


def count_patterns(text: str, patterns: list[str]) -> tuple[int, list[str]]:
    """Count regex matches; return (count, list of matched pattern names)."""
    hits = []
    for p in patterns:
        m = re.search(p, text)
        if m:
            hits.append(p)
    return len(hits), hits


def score_response(text: str) -> dict:
    crystal_count, crystal_hits = count_patterns(text, CRYSTAL_PATTERNS)
    ruby_count, ruby_hits = count_patterns(text, RUBY_PATTERNS)
    return {
        "crystal_count": crystal_count,
        "ruby_count": ruby_count,
        "preference": crystal_count - ruby_count,
        "crystal_hits": crystal_hits,
        "ruby_hits": ruby_hits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--pairs-file", default="dpo_pairs.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="Run only first N pairs (0 = all)")
    ap.add_argument("--out", default="eval_dpo_results.json")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in Path(args.pairs_file).read_text().splitlines() if l.strip()]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Running {len(pairs)} DPO pairs × {len(args.models)} models = {len(pairs) * len(args.models)} generations\n")

    results: dict = {m: [] for m in args.models}
    totals = {m: {"crystal": 0, "ruby": 0, "preference": 0, "n_pairs_won_chosen": 0, "n_pairs_won_rejected": 0} for m in args.models}

    for i, pair in enumerate(pairs, 1):
        sys.stdout.write(f"[{i}/{len(pairs)}] ")
        sys.stdout.flush()
        # Reference scores: what the chosen vs rejected response themselves contain
        ref_chosen = score_response(pair["chosen_response"])
        ref_rejected = score_response(pair["rejected_response"])
        for m in args.models:
            text = strip_ansi(run_ollama(m, pair["system"], pair["instruction"]))
            sc = score_response(text)
            sc["text"] = text
            sc["pair_idx"] = i - 1
            results[m].append(sc)
            totals[m]["crystal"] += sc["crystal_count"]
            totals[m]["ruby"] += sc["ruby_count"]
            totals[m]["preference"] += sc["preference"]
            # Chosen-style win: response contains a Crystal idiom from chosen AND has more Crystal than Ruby
            if sc["preference"] > 0:
                totals[m]["n_pairs_won_chosen"] += 1
            elif sc["preference"] < 0:
                totals[m]["n_pairs_won_rejected"] += 1
            sys.stdout.write(f"{m[:25]:25s}: pref={sc['preference']:+d}  ")
            sys.stdout.flush()
        # Print reference for context
        sys.stdout.write(f"  [ref chosen={ref_chosen['preference']:+d} rej={ref_rejected['preference']:+d}]\n")

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    for m in args.models:
        t = totals[m]
        print(f"\n  {m}")
        print(f"    Total Crystal-idiom hits:  {t['crystal']}")
        print(f"    Total Ruby-ism hits:       {t['ruby']}")
        print(f"    Net preference (C - R):    {t['preference']:+d}  (mean {t['preference']/len(pairs):+.2f}/pair)")
        print(f"    Pairs leaning chosen:      {t['n_pairs_won_chosen']}/{len(pairs)}")
        print(f"    Pairs leaning rejected:    {t['n_pairs_won_rejected']}/{len(pairs)}")

    if len(args.models) == 2:
        m1, m2 = args.models
        delta = totals[m1]["preference"] - totals[m2]["preference"]
        print(f"\n  HEAD-TO-HEAD: {m1} vs {m2}")
        print(f"    Preference delta (m1 - m2): {delta:+d}  (mean {delta/len(pairs):+.2f}/pair)")
        if abs(delta) < len(pairs) * 0.5:
            print("    => Difference is small relative to noise; training effect is weak/absent.")
        elif delta > 0:
            print(f"    => {m1} prefers Crystal idioms more than {m2}. TRAINING IMPRINTED.")
        else:
            print(f"    => {m2} (UNTRAINED?) prefers Crystal idioms more than {m1}. Suspect training regression.")

    Path(args.out).write_text(json.dumps({"totals": totals, "results": results}, indent=2))
    print(f"\nSaved raw outputs to {args.out}")


if __name__ == "__main__":
    main()
