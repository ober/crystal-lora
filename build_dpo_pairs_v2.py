#!/usr/bin/env python3
"""
Generate the v2 DPO dataset: keep the 74 hand-curated pairs from
build_dpo_pairs.py AND add ~500 programmatic variants to push the model
through more surface forms of each Ruby↔Crystal divergence.

The original pairs are high-quality but every one uses the same
"Show me how to write the Crystal-idiomatic version of this Ruby form:"
wrapper. Programmatic pairs use 4 alternate phrasings to break that
template-overfit failure mode.

Usage:
  python3 build_dpo_pairs_v2.py --out dpo_pairs_v2.jsonl
  python3 build_dpo_pairs_v2.py --validate    # crystal-build every chosen
"""

import argparse
import json
import re
import subprocess
import tempfile
from itertools import product
from pathlib import Path

# Reuse the original hand-curated pairs as-is.
import build_dpo_pairs

REPO = Path(__file__).resolve().parent
SYSTEM = build_dpo_pairs.SYSTEM

# Alternative instruction phrasings (the original uses only one).
INSTR_TEMPLATES = [
    "Show me how to write the Crystal-idiomatic version of this Ruby form:\n\n```ruby\n{ruby}\n```",
    "I have this Ruby code. What's the equivalent in Crystal?\n\n```ruby\n{ruby}\n```",
    "Convert to idiomatic Crystal:\n\n```ruby\n{ruby}\n```",
    "Rewrite this in Crystal:\n\n```ruby\n{ruby}\n```",
    "How would I express this in Crystal?\n\n```ruby\n{ruby}\n```",
]

CLASS_NAMES = ["User", "Account", "Product", "Order", "Customer", "Invoice",
               "Vehicle", "Animal", "Book", "Recipe", "Movie", "Song",
               "Player", "Team", "Article", "Comment", "Post", "Session",
               "Token", "Profile"]

FIELD_SETS = [
    [("name", "String")],
    [("name", "String"), ("age", "Int32")],
    [("title", "String"), ("price", "Float64")],
    [("id", "Int64"), ("email", "String")],
    [("count", "Int32")],
    [("balance", "Float64"), ("currency", "String")],
    [("first_name", "String"), ("last_name", "String"), ("active", "Bool")],
    [("x", "Int32"), ("y", "Int32")],
]


# ── Rule generators (each yields (ruby, crystal) tuples) ─────────────


def rule_attr_accessor():
    for cls, fields in product(CLASS_NAMES, FIELD_SETS):
        ruby_attrs = ", ".join(f":{n}" for n, _ in fields)
        ruby = f"class {cls}\n  attr_accessor {ruby_attrs}\nend"
        crystal_props = "\n".join(f"  property {n} : {t}" for n, t in fields)
        crystal = f"class {cls}\n{crystal_props}\n\n  def initialize({', '.join(f'@{n} : {t}' for n, t in fields)})\n  end\nend"
        yield ruby, crystal


def rule_attr_reader():
    for cls, fields in product(CLASS_NAMES, FIELD_SETS):
        ruby_attrs = ", ".join(f":{n}" for n, _ in fields)
        ruby = f"class {cls}\n  attr_reader {ruby_attrs}\nend"
        crystal_getters = "\n".join(f"  getter {n} : {t}" for n, t in fields)
        crystal = f"class {cls}\n{crystal_getters}\n\n  def initialize({', '.join(f'@{n} : {t}' for n, t in fields)})\n  end\nend"
        yield ruby, crystal


def rule_attr_writer():
    for cls, fields in product(CLASS_NAMES[:10], FIELD_SETS):
        ruby_attrs = ", ".join(f":{n}" for n, _ in fields)
        ruby = f"class {cls}\n  attr_writer {ruby_attrs}\nend"
        crystal_setters = "\n".join(f"  setter {n} : {t}" for n, t in fields)
        crystal = f"class {cls}\n{crystal_setters}\n\n  def initialize({', '.join(f'@{n} : {t}' for n, t in fields)})\n  end\nend"
        yield ruby, crystal


def rule_generic_class():
    names = ["Box", "Stack", "Queue", "Tree", "List", "Cell", "Wrapper",
             "Container", "Holder", "Pair"]
    types_used = ["Int32", "String", "Float64"]
    for n in names:
        for _ in types_used:
            ruby = (f"class {n}<T>\n"
                    f"  def initialize(value)\n"
                    f"    @value = value\n"
                    f"  end\n"
                    f"  attr_accessor :value\n"
                    f"end")
            crystal = (f"class {n}(T)\n"
                       f"  property value : T\n\n"
                       f"  def initialize(@value : T)\n"
                       f"  end\n"
                       f"end")
            yield ruby, crystal


def rule_array_generic():
    types_used = ["Int32", "String", "Float64", "Bool", "Char"]
    for t in types_used:
        ruby_t = (t.replace("Int32", "Integer")
                   .replace("Float64", "Float")
                   .replace("Bool", "TrueClass"))
        ruby = f"arr = Array<{ruby_t}>.new\narr.push(0)"
        crystal = f"arr = [] of {t}\narr << " + ('0' if t in ('Int32','Float64') else '"x"' if t=='String' else 'false' if t=='Bool' else "'a'")
        yield ruby, crystal


def rule_hash_generic():
    pairs = [("String", "Int32"), ("Symbol", "String"), ("Int32", "Float64"),
             ("String", "String"), ("String", "Bool")]
    for k, v in pairs:
        ruby_k = k.replace("Int32", "Integer").replace("Float64", "Float")
        ruby_v = v.replace("Int32", "Integer").replace("Float64", "Float").replace("Bool", "TrueClass")
        ruby = f"h = Hash<{ruby_k}, {ruby_v}>.new\nh[:a] = 1"
        crystal = f"h = {{}} of {k} => {v}"
        yield ruby, crystal


def rule_thread_new():
    bodies = [
        'puts "hello"',
        "do_work",
        "log.info \"started\"",
        "process_item(item)",
        "fetch_data",
    ]
    for body in bodies:
        ruby = f"Thread.new do\n  {body}\nend"
        crystal = f"spawn do\n  {body}\nend"
        yield ruby, crystal


def rule_queue():
    types_used = ["Int32", "String", "Bytes"]
    for t in types_used:
        ruby = "q = Queue.new\nq << item\nx = q.pop"
        crystal = f"ch = Channel({t}).new\nch.send(item)\nx = ch.receive"
        yield ruby, crystal


def rule_rspec_describe():
    classes = ["Calculator", "User", "Parser", "Fetcher", "Database",
               "Engine", "Worker", "Cache", "Auth"]
    for c in classes:
        ruby = (f"RSpec.describe {c} do\n"
                f"  it \"works\" do\n"
                f"    expect({c}.new).to be_a({c})\n"
                f"  end\n"
                f"end")
        crystal = (f'require "spec"\n\n'
                   f"describe {c} do\n"
                   f"  it \"works\" do\n"
                   f"    {c}.new.should be_a({c})\n"
                   f"  end\n"
                   f"end")
        yield ruby, crystal


def rule_expect_to_eq():
    cases = [
        ("1 + 1", "2"),
        ('"abc".size', "3"),
        ("user.name", '"alice"'),
        ("calculate(5)", "25"),
        ("arr.first", "1"),
        ("len(\"hello\")", "5"),
        ("counter", "0"),
    ]
    for actual, expected in cases:
        ruby = f"expect({actual}).to eq({expected})"
        crystal = f"({actual}).should eq({expected})"
        yield ruby, crystal


def rule_method_missing():
    ruby = ("class Dynamic\n"
            "  def method_missing(name, *args)\n"
            "    puts \"called #{name}\"\n"
            "  end\nend")
    crystal = ("class Dynamic\n"
               "  macro method_missing(call)\n"
               "    puts \"called {{call.name}}\"\n"
               "  end\nend")
    yield ruby, crystal


def rule_proc_new():
    sigs = [("x", "Int32", "x * 2"),
            ("s", "String", "s.upcase"),
            ("n", "Int32", "n + 1"),
            ("name", "String", "\"hello \" + name")]
    for var, t, body in sigs:
        ruby = f"p = Proc.new {{ |{var}| {body} }}\np.call(value)"
        crystal = f"p = ->({var} : {t}) {{ {body} }}\np.call(value)"
        yield ruby, crystal


def rule_block_call():
    ruby = ("def each_item(items, &block)\n"
            "  items.each { |i| block.call(i) }\nend")
    crystal = ("def each_item(items)\n"
               "  items.each { |i| yield i }\nend")
    yield ruby, crystal


def rule_struct_record():
    triples = [
        ("Point", [("x", "Int32"), ("y", "Int32")]),
        ("Money", [("amount", "Int64"), ("currency", "String")]),
        ("Range", [("from", "Int32"), ("to", "Int32")]),
        ("Color", [("r", "UInt8"), ("g", "UInt8"), ("b", "UInt8")]),
    ]
    for name, fields in triples:
        ruby_args = ", ".join(f":{n}" for n, _ in fields)
        ruby = f"{name} = Struct.new({ruby_args})"
        crystal_args = ", ".join(f"{n} : {t}" for n, t in fields)
        crystal = f"record {name}, {crystal_args}"
        yield ruby, crystal


def rule_safe_navigation():
    chains = [
        "user&.profile&.name",
        "config&.database&.url",
        "request&.headers&.[](\"X-Foo\")",
    ]
    for chain in chains:
        var = chain.split("&")[0]
        # Crystal narrows safely with if-let:
        ruby = f"name = {chain}"
        crystal = f"name = {var}.try(&." + ".try(&.".join(chain.split("&.")[1:]) + ")"*(chain.count("&.")-1) + ")"
        yield ruby, crystal


def rule_exception_subclass():
    names = ["ParseError", "ValidationError", "NetworkError",
             "ConfigError", "AuthError", "TimeoutError"]
    for n in names:
        ruby = f"class {n} < StandardError\nend"
        crystal = f"class {n} < Exception\nend"
        yield ruby, crystal


def rule_raise_runtime():
    msgs = ['"oops"', '"failed"', '"invalid input"', '"connection refused"']
    for m in msgs:
        ruby = f"raise RuntimeError.new({m})"
        crystal = f"raise {m}"
        yield ruby, crystal


def rule_send_dynamic():
    methods_ = ["do_thing", "process", "validate", "compute"]
    for m in methods_:
        ruby = f"obj.send(:{m})"
        crystal = f"obj.{m}"
        yield ruby, crystal


def rule_responds_to():
    methods_ = [":foo", ":bar", ":save", ":validate"]
    for m in methods_:
        ruby = f"obj.respond_to?({m})"
        crystal = f"obj.responds_to?({m})"
        yield ruby, crystal


def rule_kind_of():
    types_used = ["String", "Int32", "Array(Int32)", "Hash(String, Int32)"]
    for t in types_used:
        ruby_t = t.replace("Int32", "Integer").replace("Array(Integer)", "Array").replace("Hash(String, Integer)", "Hash")
        ruby = f"x.kind_of?({ruby_t})"
        crystal = f"x.is_a?({t})"
        yield ruby, crystal


def rule_to_proc():
    cases = ["[1,2,3].map(&:to_s)",
             "names.map(&:upcase)",
             "users.map(&:name)",
             "items.select(&:active?)"]
    for ruby in cases:
        crystal = ruby.replace("&:", "&.")
        yield ruby, crystal


def rule_lambda():
    cases = [("x", "Int32", "x * x"),
             ("name", "String", "name.upcase"),
             ("n", "Int32", "n.to_s")]
    for var, t, body in cases:
        ruby = f"f = lambda {{ |{var}| {body} }}"
        crystal = f"f = ->({var} : {t}) {{ {body} }}"
        yield ruby, crystal


def rule_string_length():
    ruby = "name.length"
    crystal = "name.size"
    yield ruby, crystal


def rule_first_nilable():
    ruby = "x = arr.first  # might be nil"
    crystal = "x = arr.first?  # nil-returning variant"
    yield ruby, crystal


def rule_inject():
    ruby = "[1, 2, 3].inject(0) { |sum, x| sum + x }"
    crystal = "[1, 2, 3].reduce(0) { |sum, x| sum + x }"
    yield ruby, crystal


def rule_detect():
    ruby = "arr.detect { |x| x > 10 }"
    crystal = "arr.find { |x| x > 10 }"
    yield ruby, crystal


def rule_time_now():
    ruby = "now = Time.now"
    crystal = "now = Time.local"
    yield ruby, crystal


def rule_open_uri():
    ruby = ('require "open-uri"\n'
            'body = URI.open("https://example.com").read')
    crystal = ('require "http/client"\n'
               'body = HTTP::Client.get("https://example.com").body')
    yield ruby, crystal


def rule_net_http():
    ruby = ('require "net/http"\n'
            'body = Net::HTTP.get(URI("https://example.com"))')
    crystal = ('require "http/client"\n'
               'body = HTTP::Client.get("https://example.com").body')
    yield ruby, crystal


def rule_require_relative():
    paths = ["./user", "../lib/helper", "./models/post"]
    for p in paths:
        ruby = f'require_relative "{p}"'
        crystal = f'require "{p}"'
        yield ruby, crystal


def rule_class_var_typed():
    ruby = ("class Logger\n"
            "  @@instance = nil\n"
            "  def self.instance\n"
            "    @@instance ||= new\n"
            "  end\nend")
    crystal = ("class Logger\n"
               "  @@instance : self?\n"
               "  def self.instance : self\n"
               "    @@instance ||= new\n"
               "  end\nend")
    yield ruby, crystal


def rule_keyword_args():
    ruby = ('def info(opts = {})\n'
            '  puts opts[:name]\n'
            'end\n'
            'info(name: "Ada")')
    crystal = ('def info(*, name : String)\n'
               '  puts name\n'
               'end\n'
               'info(name: "Ada")')
    yield ruby, crystal


def rule_int_div():
    ruby = "avg = total / count"
    crystal = "avg = total // count   # integer division"
    yield ruby, crystal


def rule_bigint():
    ruby = "big = 10 ** 20"
    crystal = ('require "big"\n'
               'big = BigInt.new(10) ** 20')
    yield ruby, crystal


def rule_env_safe():
    ruby = ('home = ENV["HOME"]\n'
            'puts home.upcase')
    crystal = ('home = ENV["HOME"]?\n'
               'if home\n  puts home.upcase\nend')
    yield ruby, crystal


def rule_optparse():
    ruby = ('require "optparse"\n'
            'OptionParser.new do |opts|\n'
            '  opts.on("-v") { @v = true }\n'
            'end.parse!')
    crystal = ('require "option_parser"\n'
               'verbose = false\n'
               'OptionParser.parse do |opts|\n'
               '  opts.on("-v", "verbose") { verbose = true }\n'
               'end')
    yield ruby, crystal


def rule_includes_module():
    modules = ["Comparable", "Enumerable", "Greetable", "Serializable"]
    for m in modules:
        ruby = f"class Foo\n  include {m}\nend"
        crystal = f"class Foo\n  include {m}\nend"
        # Same — still useful for confirming Crystal is fine
        yield ruby, crystal


def rule_freeze():
    consts = [
        ('CONFIG = {"x" => 1}.freeze', 'CONFIG = {"x" => 1}'),
        ('NAMES = %w(a b c).freeze',  'NAMES = ["a", "b", "c"]'),
    ]
    for ruby, crystal in consts:
        yield ruby, crystal


def rule_inherited_hook():
    ruby = ("class Base\n"
            "  def self.inherited(sub)\n"
            "    REGISTRY << sub\n"
            "  end\nend")
    crystal = ("class Base\n"
               "  REGISTRY = [] of Base.class\n\n"
               "  macro inherited\n"
               "    Base::REGISTRY << {{@type}}\n"
               "  end\nend")
    yield ruby, crystal


def rule_define_method():
    ruby = ("[:get, :post, :put].each do |verb|\n"
            "  define_method(verb) { |path| handle(verb, path) }\n"
            "end")
    crystal = ("{% for verb in %w(get post put) %}\n"
               "  def {{verb.id}}(path : String)\n"
               "    handle({{verb.symbolize}}, path)\n"
               "  end\n"
               "{% end %}")
    yield ruby, crystal


def rule_eval():
    ruby = 'eval("puts 1 + 2")'
    crystal = "macro emit\n  puts 1 + 2\nend\n\nemit"
    yield ruby, crystal


def rule_min_by():
    ruby = "people.min_by(&:age)"
    crystal = "people.min_by(&.age)"
    yield ruby, crystal


def rule_compact():
    ruby = "arr.compact!"
    crystal = "arr.reject!(&.nil?)"
    yield ruby, crystal


def rule_inplace_filter():
    ruby = "arr.delete_if { |x| x < 0 }"
    crystal = "arr.reject! { |x| x < 0 }"
    yield ruby, crystal


def rule_string_format():
    ruby = '"%.2f" % 3.14'
    crystal = 'sprintf("%.2f", 3.14)'
    yield ruby, crystal


def rule_substr():
    ruby = '"abc"[1, 2]   # "bc"'
    crystal = '"abc"[1, 2]   # "bc" (same in Crystal, but [1] is Char, not String)'
    yield ruby, crystal


def rule_dig_nilable():
    ruby = 'data.dig(:user, :address, :city)'
    crystal = 'data.dig?("user", "address", "city")'
    yield ruby, crystal


def rule_struct_value_type():
    ruby = ("Point = Struct.new(:x, :y) do\n"
            "  def +(other) = Point.new(x + other.x, y + other.y)\n"
            "end")
    crystal = ("struct Point\n"
               "  property x : Int32\n"
               "  property y : Int32\n\n"
               "  def initialize(@x : Int32, @y : Int32)\n"
               "  end\n\n"
               "  def +(other : Point) : Point\n"
               "    Point.new(x + other.x, y + other.y)\n"
               "  end\nend")
    yield ruby, crystal


def rule_open_struct():
    ruby = ('require "ostruct"\n'
            'o = OpenStruct.new(name: "Ada", age: 30)')
    crystal = ('o = {name: "Ada", age: 30}   # NamedTuple, compile-time keys\n'
               '# Or for runtime keys:\n'
               'h = {"name" => "Ada", "age" => "30"}  # Hash(String, String)')
    yield ruby, crystal


def rule_unless_postfix():
    ruby = "puts 'OK' unless errors.any?"
    crystal = "puts \"OK\" if errors.empty?"
    yield ruby, crystal


def rule_yield_self():
    ruby = "result = data.yield_self { |d| process(d) }"
    crystal = "result = data.try { |d| process(d) }"
    yield ruby, crystal


RULES = [
    rule_attr_accessor, rule_attr_reader, rule_attr_writer,
    rule_generic_class, rule_array_generic, rule_hash_generic,
    rule_thread_new, rule_queue,
    rule_rspec_describe, rule_expect_to_eq,
    rule_method_missing, rule_proc_new, rule_block_call,
    rule_struct_record, rule_safe_navigation,
    rule_exception_subclass, rule_raise_runtime,
    rule_send_dynamic, rule_responds_to, rule_kind_of,
    rule_to_proc, rule_lambda,
    rule_string_length, rule_first_nilable,
    rule_inject, rule_detect, rule_time_now,
    rule_open_uri, rule_net_http, rule_require_relative,
    rule_class_var_typed, rule_keyword_args, rule_int_div,
    rule_bigint, rule_env_safe, rule_optparse,
    rule_freeze, rule_inherited_hook, rule_define_method,
    rule_eval, rule_min_by, rule_compact,
    rule_inplace_filter, rule_string_format, rule_substr,
    rule_dig_nilable, rule_struct_value_type, rule_open_struct,
    rule_unless_postfix, rule_yield_self,
]


def make_pair(ruby_snippet, chosen_crystal, instr_template):
    instruction = instr_template.format(ruby=ruby_snippet)
    chosen_response = f"```crystal\n{chosen_crystal}\n```"
    rejected_response = f"```ruby\n{ruby_snippet}\n```"
    return {
        "system": SYSTEM,
        "instruction": instruction,
        "chosen_response": chosen_response,
        "rejected_response": rejected_response,
    }


def load_original_pairs():
    """Re-build the original 74 hand-curated pairs from build_dpo_pairs.py."""
    return [build_dpo_pairs.build_pair(p) for p in build_dpo_pairs.PAIRS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dpo_pairs_v2.jsonl")
    ap.add_argument("--validate", action="store_true",
                    help="Run crystal build on every chosen response (slow)")
    ap.add_argument("--max-per-rule", type=int, default=20)
    args = ap.parse_args()

    pairs = load_original_pairs()
    n_original = len(pairs)
    print(f"Loaded {n_original} hand-curated pairs from build_dpo_pairs.py")

    rule_counts = {}
    for rule in RULES:
        rule_pairs = []
        try:
            template_idx = 0
            for ruby, crystal in rule():
                if len(rule_pairs) >= args.max_per_rule:
                    break
                tmpl = INSTR_TEMPLATES[template_idx % len(INSTR_TEMPLATES)]
                template_idx += 1
                rule_pairs.append(make_pair(ruby, crystal, tmpl))
        except Exception as e:
            print(f"  rule {rule.__name__} errored: {e}")
            continue
        rule_counts[rule.__name__] = len(rule_pairs)
        pairs.extend(rule_pairs)

    n_total = len(pairs)
    n_synthetic = n_total - n_original
    print(f"\nGenerated {n_synthetic} synthetic pairs from {len(RULES)} rules.")
    print(f"Total: {n_total} pairs ({n_original} hand-curated + {n_synthetic} synthetic)\n")
    for n, c in sorted(rule_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {c:3d}  {n}")

    if args.validate:
        print(f"\nValidating chosen responses with `crystal build`...")
        ok = bad = noblock = 0
        for i, p in enumerate(pairs):
            m = re.search(r"```crystal\s*\n(.*?)\n```", p["chosen_response"], re.DOTALL)
            if not m:
                noblock += 1
                continue
            code = m.group(1)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".cr", delete=False) as f:
                f.write(code)
                path = f.name
            try:
                r = subprocess.run(["crystal", "build", "--no-codegen", path],
                                   capture_output=True, text=True, timeout=15)
                if r.returncode == 0:
                    ok += 1
                else:
                    bad += 1
                    if bad <= 5:
                        print(f"  pair {i} compile FAIL:\n    {code[:120]}\n    err: {r.stderr.strip()[-200:]}")
            except subprocess.TimeoutExpired:
                bad += 1
            finally:
                Path(path).unlink(missing_ok=True)
        print(f"\nCompile validation: {ok} OK, {bad} failed, {noblock} no code block")

    out_path = REPO / args.out
    out_path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    print(f"\nWrote {n_total} pairs to {out_path}")


if __name__ == "__main__":
    main()
