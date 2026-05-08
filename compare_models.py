#!/usr/bin/env python3
"""Run verify_model.py's TEST_CASES against base + fine-tuned, print side-by-side."""
import os, sys, json, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from verify_model import TEST_CASES, SYSTEM_PROMPT
from openai import OpenAI


def run_one(client, model, prompt):
    t0 = time.time()
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return r.choices[0].message.content, time.time() - t0


def score(answer, must_contain, must_not_contain):
    a = answer.lower()
    issues = []
    for t in must_contain:
        if t.lower() not in a:
            issues.append(f"missing '{t}'")
    for t in must_not_contain:
        if t.lower() in a:
            issues.append(f"contains '{t}'")
    return (len(issues) == 0), issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--together-key", default=os.environ.get("TOGETHER_API_KEY"))
    p.add_argument("--base", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p.add_argument("--ollama-model", default="jaimef/crystal-qwen")
    p.add_argument("--out", default="comparison.json")
    args = p.parse_args()

    if not args.together_key:
        tk = Path.home() / ".together-ai.token"
        if tk.exists():
            args.together_key = tk.read_text().strip()
        else:
            sys.exit("Need TOGETHER_API_KEY or ~/.together-ai.token")

    base_client = OpenAI(base_url="https://api.together.xyz/v1", api_key=args.together_key)
    ft_client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

    results = []
    base_pass = ft_pass = 0
    for i, t in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {t['prompt']}", flush=True)
        try:
            base_ans, base_dt = run_one(base_client, args.base, t["prompt"])
            base_ok, base_issues = score(base_ans, t["must_contain"], t["must_not_contain"])
        except Exception as e:
            base_ans, base_dt, base_ok, base_issues = f"ERROR: {e}", 0, False, ["error"]
        try:
            ft_ans, ft_dt = run_one(ft_client, args.ollama_model, t["prompt"])
            ft_ok, ft_issues = score(ft_ans, t["must_contain"], t["must_not_contain"])
        except Exception as e:
            ft_ans, ft_dt, ft_ok, ft_issues = f"ERROR: {e}", 0, False, ["error"]

        if base_ok: base_pass += 1
        if ft_ok: ft_pass += 1

        marker_b = "PASS" if base_ok else f"FAIL ({', '.join(base_issues)})"
        marker_f = "PASS" if ft_ok else f"FAIL ({', '.join(ft_issues)})"
        print(f"  base:  {marker_b} ({base_dt:.1f}s)")
        print(f"  ft:    {marker_f} ({ft_dt:.1f}s)")

        results.append({
            "i": i, "prompt": t["prompt"],
            "must_contain": t["must_contain"], "must_not_contain": t["must_not_contain"],
            "base": {"ok": base_ok, "issues": base_issues, "answer": base_ans, "dt": base_dt},
            "ft": {"ok": ft_ok, "issues": ft_issues, "answer": ft_ans, "dt": ft_dt},
        })

    n = len(TEST_CASES)
    print(f"\n{'='*60}")
    print(f"Base    (Qwen3-Coder-30B):     {base_pass}/{n}  ({100*base_pass//n}%)")
    print(f"FT      (jaimef/crystal-qwen): {ft_pass}/{n}  ({100*ft_pass//n}%)")
    print(f"Delta:  +{ft_pass - base_pass}")

    Path(args.out).write_text(json.dumps({
        "base_pass": base_pass, "ft_pass": ft_pass, "n": n,
        "results": results,
    }, indent=2))
    print(f"\nFull results: {args.out}")


if __name__ == "__main__":
    main()
