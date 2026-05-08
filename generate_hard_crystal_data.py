#!/usr/bin/env python3
"""
Generate compiler-checked Crystal SFT and DPO training data.

This intentionally favors small, task-shaped examples over doc dumps.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "hard_data"
SYSTEM = (
    "You are an expert Crystal programmer. Write idiomatic Crystal, not Ruby. "
    "When code is requested, return Crystal code that compiles."
)


NAMES = ["User", "Account", "Job", "Metric", "Config", "Event", "Task", "Point", "Token", "Record"]
FIELDS = [
    ("name", "String", '"Ada"'),
    ("age", "Int32", "36"),
    ("active", "Bool", "true"),
    ("score", "Float64", "98.5"),
    ("count", "Int64", "42_i64"),
    ("tag", "Symbol", ":ok"),
]


def run_crystal(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="crystal-hard-") as td:
        path = Path(td) / "main.cr"
        path.write_text(code)
        cmd = ["crystal", "build", "--no-codegen", str(path)]
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=20)
        return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def fenced(code: str) -> str:
    return f"```crystal\n{code.rstrip()}\n```"


def sft(prompt: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM, "weight": 0},
            {"role": "user", "content": prompt, "weight": 0},
            {"role": "assistant", "content": answer, "weight": 1},
        ]
    }


def dpo(prompt: str, chosen: str, rejected: str) -> dict:
    return {
        "input": {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]},
        "preferred_output": [{"role": "assistant", "content": chosen}],
        "non_preferred_output": [{"role": "assistant", "content": rejected}],
    }


def add_verified(rows, prefs, prompt, code, explanation, rejected):
    ok, err = run_crystal(code)
    if not ok:
        return False, err
    answer = f"{explanation}\n\n{fenced(code)}"
    rows.append(sft(prompt, answer))
    prefs.append(dpo(prompt, answer, rejected))
    return True, ""


def gen_property(rows, prefs):
    made = 0
    for cls in NAMES:
        for field, typ, value in FIELDS:
            code = f"""class {cls}
  property {field} : {typ}

  def initialize(@{field} : {typ})
  end
end

item = {cls}.new({value})
puts item.{field}
"""
            prompt = f"Write a Crystal class `{cls}` with a typed `{field}` property."
            rejected = f"Use Ruby-style accessors:\n\n```ruby\nclass {cls}\n  attr_accessor :{field}\nend\n```"
            ok, _ = add_verified(rows, prefs, prompt, code, "Use `property`, with a Crystal type annotation.", rejected)
            made += ok
    return made


def gen_generics(rows, prefs):
    made = 0
    for cls in ["Box", "Stack", "Cell", "Result", "Cache", "Wrapper", "Slot", "Node"]:
        code = f"""class {cls}(T)
  def initialize(@value : T)
  end

  def value : T
    @value
  end
end

box = {cls}(Int32).new(42)
puts box.value
"""
        prompt = f"Declare a generic Crystal class `{cls}` for a value of type T."
        rejected = f"Use angle brackets:\n\n```crystal\nclass {cls}<T>\n  def initialize(@value : T)\n  end\nend\n```"
        ok, _ = add_verified(rows, prefs, prompt, code, "Crystal generics use parentheses, not angle brackets.", rejected)
        made += ok
    return made


def gen_nilables(rows, prefs):
    made = 0
    for name in ["title", "email", "token", "path", "host", "label", "message", "owner"]:
        code = f"""def length_of({name} : String?) : Int32
  if value = {name}
    value.size
  else
    0
  end
end

puts length_of("abc")
puts length_of(nil)
"""
        prompt = f"Write Crystal code that safely handles nilable string `{name}`."
        rejected = f"Use it directly:\n\n```crystal\ndef length_of({name} : String?)\n  {name}.size\nend\n```"
        ok, _ = add_verified(rows, prefs, prompt, code, "Bind the nilable value in an `if` so Crystal narrows it to `String`.", rejected)
        made += ok
    return made


def gen_collections(rows, prefs):
    made = 0
    specs = [
        ("Array(Int32)", "[1, 2, 3]", "values.sum"),
        ("Hash(String, Int32)", '{"a" => 1, "b" => 2}', 'values["a"] + values["b"]'),
        ("Set(String)", 'Set{"red", "blue"}', 'values.includes?("red")'),
        ("Deque(Int32)", "Deque{1, 2, 3}", "values.shift"),
    ]
    for typ, literal, expr in specs:
        code = f"""require "set"
require "deque"

values = {literal}
puts {expr}
"""
        prompt = f"Show idiomatic Crystal usage of `{typ}`."
        rejected = "Use untyped Ruby collection syntax and assume dynamic types."
        ok, _ = add_verified(rows, prefs, prompt, code, "Use typed stdlib collections and Crystal methods.", rejected)
        made += ok
    return made


def gen_specs(rows, prefs):
    made = 0
    for op, body, expected in [
        ("double", "x * 2", "8"),
        ("square", "x * x", "16"),
        ("label", '"item-#{x}"', '"item-4"'),
    ]:
        code = f"""require "spec"

def {op}(x : Int32)
  {body}
end

describe "#{op}" do
  it "works" do
    {op}(4).should eq({expected})
  end
end
"""
        prompt = f"Write a Crystal spec for a `{op}` helper."
        rejected = "Use RSpec:\n\n```ruby\nrequire \"rspec\"\ndescribe \"helper\" do\nend\n```"
        ok, _ = add_verified(rows, prefs, prompt, code, "Crystal uses the built-in `spec` library, not RSpec.", rejected)
        made += ok
    return made


def gen_enums_and_unions(rows, prefs):
    made = 0
    for enum in ["State", "Color", "Mode"]:
        code = f"""enum {enum}
  Ready
  Running
  Done
end

def finished?(value : {enum}) : Bool
  value.done?
end

puts finished?({enum}::Done)
"""
        prompt = f"Write a Crystal enum `{enum}` and test one value."
        rejected = "Use string constants and compare arbitrary strings."
        ok, _ = add_verified(rows, prefs, prompt, code, "Use `enum` and generated predicate methods like `.done?`.", rejected)
        made += ok
    for typ in ["Int32 | String", "String | Nil", "Bool | Int32"]:
        code = f"""def describe(value : {typ}) : String
  case value
  when Int32
    "int"
  when String
    "string"
  when Bool
    "bool"
  when Nil
    "nil"
  else
    "other"
  end
end

puts describe(1)
"""
        prompt = f"Handle a Crystal union type `{typ}` with type narrowing."
        rejected = "Use `is_a?` checks but then call methods without narrowing."
        ok, _ = add_verified(rows, prefs, prompt, code, "Use `case`/`when` so Crystal narrows the union type.", rejected)
        made += ok
    return made


def gen_blocks_and_overloads(rows, prefs):
    made = 0
    for method in ["transform", "collect_value", "with_item"]:
        code = f"""def {method}(x : Int32, &block : Int32 -> String) : String
  block.call(x)
end

puts {method}(3) {{ |n| "value=#{{n}}" }}
"""
        prompt = f"Write a Crystal method `{method}` that takes a typed block."
        rejected = "Use an untyped Ruby block signature and rely on runtime dispatch."
        ok, _ = add_verified(rows, prefs, prompt, code, "Crystal block argument and return types can be annotated.", rejected)
        made += ok
    for method in ["parse_id", "normalize", "render"]:
        code = f"""def {method}(value : Int32) : String
  value.to_s
end

def {method}(value : String) : String
  value.strip
end

puts {method}(12)
puts {method}(" ok ")
"""
        prompt = f"Show Crystal method overloads for `{method}`."
        rejected = "Use one dynamic method and inspect classes at runtime like Ruby."
        ok, _ = add_verified(rows, prefs, prompt, code, "Crystal supports overloads by argument type.", rejected)
        made += ok
    return made


def gen_channels(rows, prefs):
    code = """channel = Channel(Int32).new

spawn do
  3.times do |i|
    channel.send(i)
  end
  channel.close
end

while value = channel.receive?
  puts value
end
"""
    rejected = """Use OS threads like Ruby:

```crystal
Thread.new { channel << 1 }
```"""
    ok, _ = add_verified(
        rows, prefs,
        "Show a small Crystal producer using `spawn` and `Channel(Int32)`.",
        code,
        "Crystal uses fibers with `spawn` and typed channels.",
        rejected,
    )
    return int(ok)


def gen_json(rows, prefs):
    code = """require "json"

struct Person
  include JSON::Serializable

  property name : String
  property age : Int32
end

person = Person.from_json(%({"name":"Ada","age":36}))
puts person.name
"""
    rejected = """Parse into an untyped hash and cast the object:

```crystal
person = JSON.parse(json).as(Person)
```"""
    ok, _ = add_verified(rows, prefs, "Parse JSON into a typed Crystal struct.", code, "Use `JSON::Serializable` and `.from_json`.", rejected)
    return int(ok)


def gen_ffi(rows, prefs):
    code = """lib LibC
  fun puts(str : UInt8*) : Int32
end

LibC.puts("hello".to_unsafe)
"""
    rejected = """Use Ruby FFI:

```ruby
extend FFI::Library
extern "C" int puts(char*)
```"""
    ok, _ = add_verified(rows, prefs, "Call C `puts` from Crystal.", code, "Crystal FFI uses `lib` and `fun` declarations.", rejected)
    return int(ok)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_curated_existing() -> list[dict]:
    path = ROOT / "training_data.jsonl"
    if not path.exists():
        return []
    keep_prefixes = (
        "convention:",
        "divergence:",
        "sample:",
        "spec:",
        "mcp-spec:",
        "mcp-src:",
    )
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        src = row.get("source", "")
        if src.startswith(keep_prefixes):
            rows.append({"messages": row["conversations"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--valid-frac", type=float, default=0.12)
    args = ap.parse_args()

    if not shutil.which("crystal"):
        raise SystemExit("crystal not found on PATH")

    random.seed(args.seed)
    rows: list[dict] = []
    prefs: list[dict] = []
    counts = {
        "properties": gen_property(rows, prefs),
        "generics": gen_generics(rows, prefs),
        "nilables": gen_nilables(rows, prefs),
        "collections": gen_collections(rows, prefs),
        "specs": gen_specs(rows, prefs),
        "enums_unions": gen_enums_and_unions(rows, prefs),
        "blocks_overloads": gen_blocks_and_overloads(rows, prefs),
        "channels": gen_channels(rows, prefs),
        "json": gen_json(rows, prefs),
        "ffi": gen_ffi(rows, prefs),
    }

    random.shuffle(rows)
    split = max(1, int(len(rows) * (1 - args.valid_frac)))
    train, valid = rows[:split], rows[split:]

    mixed = load_curated_existing() + rows
    random.shuffle(mixed)
    mixed_split = max(1, int(len(mixed) * (1 - args.valid_frac)))
    mixed_train, mixed_valid = mixed[:mixed_split], mixed[mixed_split:]

    OUT.mkdir(exist_ok=True)
    write_jsonl(OUT / "crystal_hard_sft_train.jsonl", train)
    write_jsonl(OUT / "crystal_hard_sft_valid.jsonl", valid)
    write_jsonl(OUT / "crystal_hard_dpo.jsonl", prefs)
    write_jsonl(OUT / "crystal_mixed_sft_train.jsonl", mixed_train)
    write_jsonl(OUT / "crystal_mixed_sft_valid.jsonl", mixed_valid)

    chars = sum(len(json.dumps(r, ensure_ascii=False)) for r in mixed_train + mixed_valid + prefs)
    tokens = chars // 4
    print(f"verified_sft={len(rows)} curated_existing={len(mixed)-len(rows)} dpo={len(prefs)}")
    print(f"files={OUT}")
    print(f"rough_mixed_plus_dpo_tokens={tokens} sft_2epoch_cost_usd~{tokens*2/1_000_000*1.5:.2f}")


if __name__ == "__main__":
    main()
