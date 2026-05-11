#!/usr/bin/env python3
"""
Per-pair similarity test:
  similarity_to_chosen   - similarity_to_rejected   = "chosen lean"
For each model, sum the lean across all pairs.
If trained model leans toward chosen more than base, training imprinted.

This avoids the verbosity trap of pattern-counting: if a model writes more,
it hits more patterns. Similarity to a *specific target* is volume-invariant.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from difflib import SequenceMatcher


def strip_ansi(s: str) -> str:
    s = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", s)
    s = re.sub(r"\[\?2026[hl]|\[\?25[hl]|\[\?[0-9]+[hl]", "", s)
    s = re.sub(r"\[K|\[1G|\[A|\[2K", "", s)
    return s.strip()


def normalize(text: str) -> str:
    """Strip code fences, normalize whitespace, lowercase."""
    text = re.sub(r"```\w*", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity on word tokens."""
    ta = set(re.findall(r"[\w@:?]+", a))
    tb = set(re.findall(r"[\w@:?]+", b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def char_ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio on normalized text."""
    return SequenceMatcher(None, a, b).ratio()


def score(response: str, chosen: str, rejected: str) -> dict:
    rn = normalize(response)
    cn = normalize(chosen)
    jn = normalize(rejected)
    return {
        "tok_sim_chosen": token_jaccard(rn, cn),
        "tok_sim_rejected": token_jaccard(rn, jn),
        "tok_lean": token_jaccard(rn, cn) - token_jaccard(rn, jn),
        "char_sim_chosen": char_ratio(rn, cn),
        "char_sim_rejected": char_ratio(rn, jn),
        "char_lean": char_ratio(rn, cn) - char_ratio(rn, jn),
    }


def run_ollama(model: str, system: str, prompt: str, timeout: int = 120) -> str:
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
        return f"<PARSE_ERROR {e}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--pairs-file", default="dpo_pairs.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="eval_similarity.json")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in Path(args.pairs_file).read_text().splitlines() if l.strip()]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Pairs: {len(pairs)}, models: {args.models}\n")

    results = {m: [] for m in args.models}
    sums = {m: {"tok_lean": 0.0, "char_lean": 0.0, "wins_chosen": 0, "wins_rejected": 0} for m in args.models}

    for i, pair in enumerate(pairs, 1):
        sys.stdout.write(f"[{i:2d}/{len(pairs)}] ")
        sys.stdout.flush()
        for m in args.models:
            text = strip_ansi(run_ollama(m, pair["system"], pair["instruction"]))
            sc = score(text, pair["chosen_response"], pair["rejected_response"])
            sc["text"] = text
            sc["pair_idx"] = i - 1
            results[m].append(sc)
            sums[m]["tok_lean"] += sc["tok_lean"]
            sums[m]["char_lean"] += sc["char_lean"]
            if sc["tok_lean"] > 0:
                sums[m]["wins_chosen"] += 1
            elif sc["tok_lean"] < 0:
                sums[m]["wins_rejected"] += 1
            sys.stdout.write(f"{m[:25]:25s} tok_lean={sc['tok_lean']:+.3f} ")
        sys.stdout.write("\n")

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS  (positive = leans toward CHOSEN training response)")
    print("=" * 80)
    for m in args.models:
        s = sums[m]
        print(f"\n  {m}")
        print(f"    Sum token-similarity lean:  {s['tok_lean']:+.3f}  (mean {s['tok_lean']/len(pairs):+.4f}/pair)")
        print(f"    Sum char-similarity lean:   {s['char_lean']:+.3f}  (mean {s['char_lean']/len(pairs):+.4f}/pair)")
        print(f"    Pairs where response leans toward chosen:    {s['wins_chosen']}/{len(pairs)}")
        print(f"    Pairs where response leans toward rejected:  {s['wins_rejected']}/{len(pairs)}")

    if len(args.models) == 2:
        m1, m2 = args.models
        delta_tok = sums[m1]["tok_lean"] - sums[m2]["tok_lean"]
        delta_char = sums[m1]["char_lean"] - sums[m2]["char_lean"]
        print(f"\n  HEAD-TO-HEAD DELTA: {m1} - {m2}")
        print(f"    Token-lean delta:  {delta_tok:+.3f}  ({delta_tok/len(pairs):+.4f}/pair)")
        print(f"    Char-lean delta:   {delta_char:+.3f}  ({delta_char/len(pairs):+.4f}/pair)")

    Path(args.out).write_text(json.dumps({"sums": sums, "results": results}, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
