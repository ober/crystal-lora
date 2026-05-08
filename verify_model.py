#!/usr/bin/env python3
"""
Verify a fine-tuned Crystal-language model by running test prompts.

Works with any OpenAI-compatible API (Together AI, Ollama, RunPod, OpenRouter, etc.)

Usage:
  # Test against Ollama
  python3 verify_model.py --base-url http://localhost:11434/v1 --model crystal-qwen

  # Test against Together AI
  python3 verify_model.py --base-url https://api.together.xyz/v1 \\
      --model your-org/crystal-qwen --api-key $TOGETHER_API_KEY

  # Test against RunPod
  python3 verify_model.py --base-url https://api.runpod.ai/v2/<ID>/openai/v1 \\
      --model your-user/crystal-qwen-30b --api-key $RUNPOD_API_KEY
"""

import argparse
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Install the OpenAI SDK: pip install openai")
    sys.exit(1)


SYSTEM_PROMPT = (
    "You are an expert in Crystal, a statically-typed, Ruby-syntax-inspired "
    "language compiled via LLVM. Provide accurate, idiomatic Crystal code "
    "with correct types, requires, and method signatures."
)

# Each test asserts on terms that should (or must not) appear in the answer.
TEST_CASES = [
    {
        "prompt": "How do I declare a class with type-annotated properties in Crystal?",
        "must_contain": ["property", " : "],
        "must_not_contain": ["attr_accessor", "attr_reader"],
    },
    {
        "prompt": "What's the Crystal equivalent of Ruby's attr_accessor?",
        "must_contain": ["property"],
        "must_not_contain": ["attr_accessor"],
    },
    {
        "prompt": "Show me how to spawn a fiber and use a Channel in Crystal.",
        "must_contain": ["spawn", "Channel"],
        "must_not_contain": ["Thread.new"],
    },
    {
        "prompt": "How do I parse JSON into a typed class in Crystal?",
        "must_contain": ["JSON::Serializable", "from_json"],
        "must_not_contain": [],
    },
    {
        "prompt": "How do I write a basic HTTP server in Crystal?",
        "must_contain": ["HTTP::Server", "require \"http/server\""],
        "must_not_contain": ["Sinatra"],
    },
    {
        "prompt": "How do I write a spec test in Crystal?",
        "must_contain": ["require \"spec\"", "describe", "it "],
        "must_not_contain": ["RSpec", "rspec"],
    },
    {
        "prompt": "How do I declare a generic class in Crystal?",
        "must_contain": ["(T)"],
        "must_not_contain": ["<T>"],
    },
    {
        "prompt": "What's the difference between Tuple and NamedTuple in Crystal?",
        "must_contain": ["NamedTuple", "Tuple"],
        "must_not_contain": [],
    },
    {
        "prompt": "How do I call a C function from Crystal?",
        "must_contain": ["lib", "fun"],
        "must_not_contain": ["FFI::Library", "extern"],
    },
    {
        "prompt": "How do I declare a nilable type in Crystal?",
        "must_contain": ["Nil", "?"],
        "must_not_contain": ["Optional<", "None"],
    },
]


def run_test(client, model, test_case):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_case["prompt"]},
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    answer_lower = answer.lower()

    passed = True
    issues = []

    for term in test_case["must_contain"]:
        if term.lower() not in answer_lower:
            passed = False
            issues.append(f"missing '{term}'")

    for term in test_case["must_not_contain"]:
        if term.lower() in answer_lower:
            passed = False
            issues.append(f"contains wrong term '{term}'")

    return passed, answer, issues


def main():
    parser = argparse.ArgumentParser(description="Verify Crystal LoRA model")
    parser.add_argument("--base-url", required=True, help="API base URL")
    parser.add_argument("--model", required=True, help="Model name/ID")
    parser.add_argument("--api-key", default="not-needed", help="API key")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full responses")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    total = len(TEST_CASES)
    passed = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{total}] {test['prompt'][:70]}...")

        try:
            ok, answer, issues = run_test(client, args.model, test)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if ok:
            passed += 1
            print(f"  PASS")
        else:
            print(f"  FAIL: {', '.join(issues)}")

        if args.verbose:
            print(f"  Response: {answer[:200]}...")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed ({100*passed//total}%)")

    if passed >= total * 0.8:
        print("Model looks good for Crystal!")
    elif passed >= total * 0.5:
        print("Partial success — consider more training epochs or data.")
    else:
        print("Model needs more training. Check data format and hyperparameters.")


if __name__ == "__main__":
    main()
