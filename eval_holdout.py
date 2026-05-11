#!/usr/bin/env python3
"""
Held-out Crystal eval. Questions phrased as natural tasks, NOT as
"Crystal-idiomatic version of this Ruby form: [snippet]" (which is what
the training data uses). Tests generalization, not memorization.

Per question, score = idiom_score + compile_score:
  idiom_score = (must_have hits) - (must_not hits)
  compile_score = +5 if extracted Crystal code block compiles cleanly, 0 otherwise

Compile gate runs `crystal build --no-codegen` on the FIRST extracted ```crystal
block. If no code block, compile_score = 0.

Usage:
  python3 eval_holdout.py --models jaimef/crystal-qwen3.6-30b qwen3-coder:30b
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

CRYSTAL_SYSTEM = (
    "You are an expert in Crystal, a statically-typed Ruby-syntax-inspired language "
    "compiled to native code via LLVM. Write idiomatic Crystal: type annotations on "
    "instance vars and method args; getter/setter/property (not attr_*); generics use "
    "(T) not <T>; spawn/Channel(T) for concurrency (not Thread.new); built-in Spec "
    "framework (not RSpec); unions written as `Type?` or `Type | Nil`."
)


@dataclass
class Q:
    name: str
    prompt: str
    must_have: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    case_sensitive: bool = True
    # If non-empty, the extracted code block is wrapped/prefixed with this before compile-check.
    # Useful when the model is expected to produce a fragment, not a full program.
    compile_wrap: str = "{code}"
    compile_check: bool = True


# Held-out questions. None of these match the "Crystal-idiomatic version of this Ruby form:"
# template used in training. Every question phrased as a task or natural inquiry.
QUESTIONS: list[Q] = [
    Q(
        name="task_inventory_class",
        prompt=(
            "I'm writing inventory tracking software in Crystal. Write a class `Item` "
            "with name (string), quantity (integer), and an optional description (nilable "
            "string). The class should have read-write access to all three fields and a "
            "constructor that takes name and quantity (description defaults to nil). Just the class."
        ),
        must_have=["property name", "property quantity", "property description", " : Int32", " : String", "String?"],
        must_not=["attr_accessor", "attr_reader", "attr_writer"],
    ),
    Q(
        name="task_concurrent_downloads",
        prompt=(
            "Write Crystal code that fetches three URLs concurrently and prints each "
            "response's status code as it arrives. Don't use any external shards — only "
            "stdlib. Just the code."
        ),
        must_have=["spawn", "Channel", "HTTP::Client"],
        must_not=["Thread.new", "Mutex.new", "Queue.new", "Async."],
    ),
    Q(
        name="task_word_count",
        prompt=(
            "Write a Crystal method `word_count(text : String) : Hash(String, Int32)` that "
            "returns a hash mapping each lowercased word to its frequency. Just the method."
        ),
        must_have=["Hash(String, Int32)", "def word_count"],
        must_not=["Hash<String", "Hash<String,Int32>"],
        compile_wrap="{code}\n\nputs word_count(\"the quick the\")\n",
    ),
    Q(
        name="task_generic_queue",
        prompt=(
            "Implement a simple bounded FIFO queue type in Crystal that holds elements "
            "of any type T. It needs `enqueue(item : T)`, `dequeue : T?`, and `size : Int32`. "
            "Use Array internally. Just the class definition."
        ),
        must_have=["class", "(T)", "Array(T)", "T?"],
        must_not=["<T>", "Array<T>", "Optional", "Maybe"],
    ),
    Q(
        name="task_spec_calculator",
        prompt=(
            "I have a `Calculator` class with methods `add(a, b)` and `divide(a, b)`. "
            "Write a spec file that tests both methods, including the divide-by-zero case. "
            "Use the standard Crystal test framework. Just the spec code."
        ),
        must_have=['require "spec"', "describe", "it ", ".should"],
        must_not=['require "rspec"', "RSpec.describe", "expect(", ".to eq("],
    ),
    Q(
        name="task_json_parse",
        prompt=(
            "I have a JSON string like `{\"name\":\"Alice\",\"age\":30}`. Show me how to "
            "parse it in Crystal into a strongly-typed struct using the JSON::Serializable "
            "module. Just the code."
        ),
        must_have=["JSON::Serializable", "include JSON::Serializable", "from_json"],
        must_not=["JSON.parse(", "attr_accessor"],
    ),
    Q(
        name="task_exception_chain",
        prompt=(
            "Define a Crystal exception hierarchy: `DatabaseError` as the base, and "
            "`ConnectionError` and `QueryError` as subclasses. Write a method `query(sql)` "
            "that raises QueryError with a message. Just the code."
        ),
        must_have=["class DatabaseError < Exception", "class ConnectionError < DatabaseError", "class QueryError < DatabaseError", "raise QueryError"],
        must_not=["StandardError", "RuntimeError.new"],
    ),
    Q(
        name="task_nilable_dig",
        prompt=(
            "I have a nested `Hash(String, Hash(String, Int32))` and want to safely look up "
            "`outer_key/inner_key`, returning nil if either is missing. Write the lookup "
            "method in Crystal. Just the method."
        ),
        must_have=["Int32?", "&."],
        must_not=["dig(", "&block", "block_given?"],
    ),
    Q(
        name="task_macro_getter",
        prompt=(
            "Write a Crystal macro `define_constant(name, value)` that expands to a "
            "constant definition at compile time. Show the macro and an example usage."
        ),
        must_have=["macro define_constant", "{{name", "{{value"],
        must_not=["method_missing", "define_method", "instance_eval", "eval("],
    ),
    Q(
        name="task_struct_point",
        prompt=(
            "Define a Crystal `Point` as a struct (not a class) with x and y as Float64 "
            "fields, both readable and writable. Add a method `distance_to(other : Point) : Float64`. "
            "Just the struct."
        ),
        must_have=["struct Point", "property x", "property y", " : Float64", "def distance_to"],
        must_not=["class Point", "attr_accessor"],
    ),
    Q(
        name="task_enum_status",
        prompt=(
            "Define a Crystal enum `OrderStatus` with values Pending, Shipped, Delivered, "
            "Cancelled. Show how to switch on it using `case`/`when`. Just the code."
        ),
        must_have=["enum OrderStatus", "case ", "when "],
        must_not=["module OrderStatus", "STATUS = "],
    ),
    Q(
        name="task_named_tuple",
        prompt=(
            "Show how to use a NamedTuple in Crystal to represent a 2D point with x and y "
            "(both Int32). Create one and access both fields. Just the code."
        ),
        must_have=["NamedTuple", "x:", "y:"],
        must_not=["{:x =>", "OpenStruct", "Hash{"],
        compile_wrap="point = {code}\nputs point[:x]\nputs point[:y]\n",
        compile_check=False,  # NamedTuple literal varies; skip strict compile
    ),
    Q(
        name="task_block_yield_typed",
        prompt=(
            "Write a Crystal method `with_timing(&block)` that yields, measures elapsed "
            "wall time using `Time.monotonic`, prints it, and returns the block's value. "
            "Just the method."
        ),
        must_have=["yield", "Time.monotonic"],
        must_not=["block.call", "block_given?", "Proc.new", "&block.call"],
    ),
    Q(
        name="task_record",
        prompt=(
            "In Crystal, use the `record` macro to define a value type `Money` with `amount : Int64` "
            "and `currency : String`. Just the one-line definition."
        ),
        must_have=["record Money", "Int64", "String"],
        must_not=["class Money", "Struct.new", "attr_"],
    ),
    Q(
        name="task_fiber_select",
        prompt=(
            "I have two `Channel(Int32)` and want to receive from whichever has data first "
            "in Crystal. Show me using `select` with `when` clauses. Just the code."
        ),
        must_have=["select", "when ", ".receive"],
        must_not=["IO.select", "Thread.new", "Mutex"],
    ),
    Q(
        name="task_typeof",
        prompt=(
            "How do you get the compile-time type of an expression in Crystal? Give a "
            "one-paragraph answer with a small example."
        ),
        must_have=["typeof"],
        must_not=["Object#class", ".class.name"],
        case_sensitive=False,
    ),
    Q(
        name="task_initialize_typed",
        prompt=(
            "Write a Crystal class `User` whose `initialize` takes `name : String` and "
            "`email : String` and assigns them to instance variables. Use the shorthand "
            "form. Just the class."
        ),
        must_have=["def initialize(@name : String", "@email : String"],
        must_not=["@name = name", "attr_"],
    ),
    Q(
        name="task_array_filter_map",
        prompt=(
            "Given `numbers = [1, 2, 3, 4, 5]` in Crystal, return a new array containing "
            "the squares of only the even numbers. Use a chained call. Just the expression."
        ),
        must_have=[".select", ".map"],
        must_not=["filter_map", "lazy", ".each_with_object"],
        compile_wrap="numbers = [1, 2, 3, 4, 5]\nresult = {code}\nputs result\n",
        compile_check=False,
    ),
    Q(
        name="task_io_read_lines",
        prompt=(
            "Write Crystal code that reads `/etc/hosts` line by line and prints each line "
            "with its line number prefixed. Use stdlib only. Just the code."
        ),
        must_have=["File.", "each_line"],
        must_not=["File.foreach", "IO.foreach", "open(", ".readlines.each_with_index"],
    ),
    Q(
        name="task_module_mixin",
        prompt=(
            "Define a Crystal module `Greetable` with an abstract method `name : String` "
            "and a method `greet` that uses it. Then a class `Person` that includes the "
            "module. Just the code."
        ),
        must_have=["module Greetable", "abstract def name", "include Greetable"],
        must_not=["extend self", "ClassMethods"],
    ),
    Q(
        name="task_string_format",
        prompt=(
            "Format the string `\"Total: $%.2f\"` with the value 1234.5 in Crystal. Show "
            "two ways: sprintf-style and string interpolation with format. Just the code."
        ),
        must_have=["sprintf", "%"],
        must_not=["printf", "String#%", ".format("],
    ),
    Q(
        name="task_optional_arg",
        prompt=(
            "Write a Crystal method `greet(name : String, greeting : String = \"Hello\")` "
            "that returns a string. Just the method and one example call."
        ),
        must_have=["def greet(name : String, greeting : String = "],
        must_not=["**kwargs", "*args", "block_given?"],
    ),
    Q(
        name="task_splat",
        prompt=(
            "Write a Crystal method `sum(*nums : Int32) : Int32` that sums any number of "
            "integer args. Just the method."
        ),
        must_have=["def sum(*nums : Int32)", ".sum"],
        must_not=["*args", "**kwargs", "splat"],
    ),
    Q(
        name="task_unless_postfix",
        prompt=(
            "Show idiomatic Crystal for: print 'OK' to stdout only when a variable `errors` "
            "is empty. Use postfix-if (or unless) form. Just the line."
        ),
        must_have=["puts", "if ", ".empty?"],
        must_not=["unless errors.any?", "if errors.size == 0"],
    ),
    Q(
        name="task_responds_to",
        prompt=(
            "How do you check at compile time whether a Crystal type responds to a particular "
            "method? Give a one-line example using the standard macro/check."
        ),
        must_have=["responds_to?"],
        must_not=["respond_to?", "method_defined?"],
        case_sensitive=False,
    ),
    Q(
        name="task_pointer_basics",
        prompt=(
            "Show how to allocate a buffer of 10 Int32s on the heap in Crystal using "
            "Pointer. Just the line."
        ),
        must_have=["Pointer(Int32).malloc", "10"],
        must_not=["Array.new", "calloc", "C.malloc"],
    ),
    Q(
        name="task_overload",
        prompt=(
            "In Crystal, define two methods both named `process` — one taking `Int32` and "
            "one taking `String` — and show that they overload on type. Just the code."
        ),
        must_have=["def process(x : Int32)", "def process(x : String)"],
        must_not=["case x.class", "if x.is_a?", "method_missing"],
    ),
    Q(
        name="task_class_method",
        prompt=(
            "Write a Crystal class `Logger` with a class-level method `instance` returning "
            "the singleton. Just the class."
        ),
        must_have=["def self.instance", "@@", "class Logger"],
        must_not=["@instance = nil unless", "Singleton"],
    ),
    Q(
        name="task_yield_with_value",
        prompt=(
            "Write a Crystal method `times_two_each(arr : Array(Int32))` that yields each "
            "element multiplied by 2. Just the method."
        ),
        must_have=["yield", "* 2"],
        must_not=["block.call", "Proc.new", "block_given?"],
    ),
    Q(
        name="task_bit_field",
        prompt=(
            "How do you check if the third bit (value 4) is set in a Crystal `UInt8`? Give "
            "a one-line expression using bitwise operators."
        ),
        must_have=["&", "0", "!= 0"],
        must_not=["bit?(", ".bit"],
        case_sensitive=False,
    ),
]


def strip_ansi(s: str) -> str:
    s = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", s)
    s = re.sub(r"\[\?2026[hl]|\[\?25[hl]|\[\?[0-9]+[hl]", "", s)
    s = re.sub(r"\[K|\[1G|\[A|\[2K", "", s)
    return s.strip()


def run_ollama(model: str, system: str, prompt: str, timeout: int = 180) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 600, "num_ctx": 4096},
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


def extract_crystal_block(text: str) -> str | None:
    """Extract the FIRST ```crystal (or unlabeled ```) code block."""
    m = re.search(r"```(?:crystal|cr)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1)
    return None


def compile_check(code: str, wrap: str = "{code}") -> tuple[bool, str]:
    """Run `crystal build --no-codegen` on code. Returns (ok, stderr)."""
    full = wrap.format(code=code)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cr", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        proc = subprocess.run(
            ["crystal", "build", "--no-codegen", "--error-trace", path],
            capture_output=True, text=True, timeout=30,
        )
        return proc.returncode == 0, proc.stderr[-500:] if proc.stderr else ""
    except subprocess.TimeoutExpired:
        return False, "<COMPILE_TIMEOUT>"
    finally:
        Path(path).unlink(missing_ok=True)


def score_question(q: Q, text: str) -> dict:
    haystack = text if q.case_sensitive else text.lower()
    have_hits, have_miss = [], []
    not_hits, not_clean = [], []
    for needle in q.must_have:
        n = needle if q.case_sensitive else needle.lower()
        (have_hits if n in haystack else have_miss).append(needle)
    for needle in q.must_not:
        n = needle if q.case_sensitive else needle.lower()
        (not_hits if n in haystack else not_clean).append(needle)

    idiom_score = len(have_hits) - len(not_hits)

    compile_ok = False
    compile_err = ""
    code_block = extract_crystal_block(text)
    if q.compile_check and code_block:
        compile_ok, compile_err = compile_check(code_block, q.compile_wrap)
    compile_score = 5 if compile_ok else 0

    return {
        "idiom_score": idiom_score,
        "compile_score": compile_score,
        "total_score": idiom_score + compile_score,
        "have_hits": have_hits,
        "have_miss": have_miss,
        "not_hits": not_hits,
        "compile_attempted": q.compile_check and code_block is not None,
        "compile_ok": compile_ok,
        "compile_err": compile_err if not compile_ok else "",
        "had_code_block": code_block is not None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="eval_holdout.json")
    ap.add_argument("--system", default=CRYSTAL_SYSTEM,
                    help="System prompt (default: built-in Crystal-aware)")
    ap.add_argument("--no-system", action="store_true",
                    help="Pass empty system prompt (for testing system-prompt impact)")
    args = ap.parse_args()

    system = "" if args.no_system else args.system
    print(f"System prompt: {'<EMPTY>' if not system else system[:80] + '...'}\n")
    print(f"Running {len(QUESTIONS)} held-out questions × {len(args.models)} models\n")

    results = {m: [] for m in args.models}
    totals = {m: {"idiom": 0, "compile": 0, "total": 0, "compile_pass": 0, "compile_attempts": 0, "had_code": 0} for m in args.models}

    for i, q in enumerate(QUESTIONS, 1):
        sys.stdout.write(f"[{i:2d}/{len(QUESTIONS)}] {q.name:30s} ")
        sys.stdout.flush()
        for m in args.models:
            text = strip_ansi(run_ollama(m, system, q.prompt))
            sc = score_question(q, text)
            sc["text"] = text
            sc["question"] = q.name
            results[m].append(sc)
            totals[m]["idiom"] += sc["idiom_score"]
            totals[m]["compile"] += sc["compile_score"]
            totals[m]["total"] += sc["total_score"]
            if sc["compile_attempted"]:
                totals[m]["compile_attempts"] += 1
                if sc["compile_ok"]:
                    totals[m]["compile_pass"] += 1
            if sc["had_code_block"]:
                totals[m]["had_code"] += 1
            sys.stdout.write(f"{m[:25]:25s} idiom={sc['idiom_score']:+d} comp={'Y' if sc['compile_ok'] else ('N' if sc['compile_attempted'] else '-')} ")
        sys.stdout.write("\n")

    print("\n" + "=" * 80)
    print("HELD-OUT EVAL RESULTS")
    print("=" * 80)
    n = len(QUESTIONS)
    for m in args.models:
        t = totals[m]
        print(f"\n  {m}")
        print(f"    Idiom score:        {t['idiom']:+d}  (mean {t['idiom']/n:+.2f}/q)")
        print(f"    Compile pass:       {t['compile_pass']}/{t['compile_attempts']} attempted  ({100*t['compile_pass']/max(1,t['compile_attempts']):.0f}%)")
        print(f"    Code blocks emitted:{t['had_code']}/{n}")
        print(f"    Combined total:     {t['total']:+d}")

    if len(args.models) == 2:
        m1, m2 = args.models
        d_idiom = totals[m1]["idiom"] - totals[m2]["idiom"]
        d_comp = totals[m1]["compile_pass"] - totals[m2]["compile_pass"]
        d_total = totals[m1]["total"] - totals[m2]["total"]
        print(f"\n  HEAD-TO-HEAD: {m1} - {m2}")
        print(f"    Idiom delta:        {d_idiom:+d}")
        print(f"    Compile-pass delta: {d_comp:+d}")
        print(f"    Total delta:        {d_total:+d}")

    Path(args.out).write_text(json.dumps({"totals": totals, "results": results}, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
