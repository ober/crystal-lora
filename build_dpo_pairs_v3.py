#!/usr/bin/env python3
"""v3 DPO dataset: hand-curated + v2 programmatic + 30+ new rule generators.

Extends build_dpo_pairs_v2.py with rules for stdlib divergence the v2 set
under-represents: JSON, File I/O, Process, Regex, Random, Comparable mixin,
ERB→ECR, OpenSSL, ENV with defaults, etc.

Usage:
  python3 build_dpo_pairs_v3.py --out dpo_pairs_v3.jsonl
  python3 build_dpo_pairs_v3.py --validate    # crystal-build every chosen
"""

import argparse
import json
import re
import subprocess
import tempfile
from itertools import product
from pathlib import Path

import build_dpo_pairs_v2 as v2

REPO = Path(__file__).resolve().parent
SYSTEM = v2.SYSTEM
INSTR_TEMPLATES = v2.INSTR_TEMPLATES


# ── New rule generators (each yields (ruby, crystal) tuples) ─────────


def rule_json_parse():
    cases = [
        ('h = JSON.parse(str)\nputs h["name"]',
         'h = JSON.parse(str).as_h\nputs h["name"]'),
        ('h = JSON.parse(File.read("config.json"))',
         'h = JSON.parse(File.read("config.json"))'),
        ('arr = JSON.parse(str)\narr.each { |x| puts x }',
         'arr = JSON.parse(str).as_a\narr.each { |x| puts x }'),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_json_serializable():
    classes = ["User", "Product", "Order"]
    fields_sets = [
        [("name", "String")],
        [("name", "String"), ("age", "Int32")],
        [("id", "Int64"), ("title", "String")],
    ]
    for cls, fields in product(classes, fields_sets):
        ruby_attrs = ", ".join(f":{n}" for n, _ in fields)
        ruby = (f"class {cls}\n"
                f"  attr_accessor {ruby_attrs}\n"
                f"  def to_json(*a)\n"
                f"    {{ {', '.join(f'{n!r} => {n}' for n, _ in fields)} }}.to_json(*a)\n"
                f"  end\nend")
        crystal = (f"class {cls}\n"
                   f"  include JSON::Serializable\n"
                   + "\n".join(f"  property {n} : {t}" for n, t in fields)
                   + f"\n\n  def initialize({', '.join(f'@{n} : {t}' for n, t in fields)})\n  end\nend")
        yield ruby, crystal


def rule_to_json_call():
    cases = [
        ("user.to_json", "user.to_json"),
        ("[1,2,3].to_json", "[1, 2, 3].to_json"),
        ('{a: 1}.to_json', '{a: 1}.to_json'),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_file_read():
    paths = ['"config.txt"', '"data.csv"', 'path']
    for p in paths:
        ruby = f"text = File.read({p})"
        crystal = f"text = File.read({p})"
        yield ruby, crystal


def rule_file_write():
    cases = [
        ('File.write("out.txt", data)',     'File.write("out.txt", data)'),
        ('File.open("log", "a") { |f| f.puts "hi" }',
         'File.open("log", "a") { |f| f.puts "hi" }'),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_file_each_line():
    ruby = ('File.foreach("data.txt") do |line|\n'
            '  puts line\n'
            'end')
    crystal = ('File.each_line("data.txt") do |line|\n'
               '  puts line\n'
               'end')
    yield ruby, crystal


def rule_pathname():
    ruby = ('require "pathname"\n'
            'p = Pathname.new("/tmp/foo")\n'
            'puts p.basename')
    crystal = ('p = Path["/tmp/foo"]\n'
               'puts p.basename')
    yield ruby, crystal


def rule_dir_glob():
    ruby = 'Dir.glob("*.cr").each { |f| puts f }'
    crystal = 'Dir.glob("*.cr") { |f| puts f }'
    yield ruby, crystal


def rule_dir_exist():
    ruby = 'Dir.exist?("/tmp")'
    crystal = 'Dir.exists?("/tmp")'
    yield ruby, crystal


def rule_file_exist():
    ruby = 'File.exist?(path)'
    crystal = 'File.exists?(path)'
    yield ruby, crystal


def rule_random():
    cases = [
        ("rand(100)",          "rand(100)"),
        ("rand(1..10)",        "rand(1..10)"),
        ("Random.new.rand(50)","Random.new.rand(50)"),
        ("[1,2,3].sample",     "[1, 2, 3].sample"),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_sleep():
    cases = [
        ("sleep 1",              "sleep 1.second"),
        ("sleep 0.5",            "sleep 500.milliseconds"),
        ("sleep 60",             "sleep 1.minute"),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_process_pid():
    ruby = "puts Process.pid"
    crystal = "puts Process.pid"
    yield ruby, crystal


def rule_system_exec():
    cases = [
        ('system("ls")',                     'system("ls")'),
        ('output = `ls -l`',                 'output = `ls -l`'),
        ('Process.spawn("git status")',      'Process.run("git", ["status"])'),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_env_default():
    ruby = 'host = ENV["HOST"] || "localhost"'
    crystal = 'host = ENV["HOST"]? || "localhost"'
    yield ruby, crystal


def rule_env_fetch():
    ruby = 'token = ENV.fetch("TOKEN") { raise "missing" }'
    crystal = 'token = ENV["TOKEN"]? || raise "missing"'
    yield ruby, crystal


def rule_regex_match():
    cases = [
        ('m = "abc".match(/(\\w+)/)\nputs m[1]',
         'if m = "abc".match(/(\\w+)/)\n  puts m[1]\nend'),
        ('"hello".scan(/l/)',
         '"hello".scan(/l/).map(&.[0])'),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_regex_gsub():
    cases = [
        ('"hello".gsub(/l/, "L")',  '"hello".gsub(/l/, "L")'),
        ('"hello".sub(/l/, "L")',   '"hello".sub(/l/, "L")'),
        ('"abc123".gsub(/\\d/) { |m| "X" }',
         '"abc123".gsub(/\\d/) { |m| "X" }'),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_string_chars():
    ruby = '"abc".chars'
    crystal = '"abc".chars'
    yield ruby, crystal


def rule_string_each_char():
    ruby = '"abc".each_char { |c| puts c }'
    crystal = '"abc".each_char { |c| puts c }'
    yield ruby, crystal


def rule_string_to_sym():
    ruby = '"hello".to_sym'
    crystal = '# Crystal symbols are compile-time only;\n# use the symbol literal instead, or a string for runtime values.\n:hello'
    yield ruby, crystal


def rule_symbol_to_proc():
    cases = [
        ("[1,2,3].map(&:to_s)",       "[1, 2, 3].map(&.to_s)"),
        ("words.map(&:upcase)",       "words.map(&.upcase)"),
        ("nums.select(&:even?)",      "nums.select(&.even?)"),
        ("users.map(&:active?)",      "users.map(&.active?)"),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_comparable_mixin():
    ruby = ("class Length\n"
            "  include Comparable\n"
            "  def <=>(other)\n"
            "    @meters <=> other.meters\n"
            "  end\nend")
    crystal = ("class Length\n"
               "  include Comparable(Length)\n\n"
               "  def initialize(@meters : Float64)\n"
               "  end\n\n"
               "  getter meters : Float64\n\n"
               "  def <=>(other : Length)\n"
               "    @meters <=> other.meters\n"
               "  end\nend")
    yield ruby, crystal


def rule_enumerable_mixin():
    ruby = ("class Bag\n"
            "  include Enumerable\n"
            "  def each\n"
            "    @items.each { |i| yield i }\n"
            "  end\nend")
    crystal = ("class Bag(T)\n"
               "  include Enumerable(T)\n\n"
               "  def initialize\n"
               "    @items = [] of T\n"
               "  end\n\n"
               "  def each(&)\n"
               "    @items.each { |i| yield i }\n"
               "  end\nend")
    yield ruby, crystal


def rule_iterable_mixin():
    ruby = ("class Counter\n"
            "  include Enumerable\n"
            "  def each\n"
            "    1.upto(@max) { |i| yield i }\n"
            "  end\nend")
    crystal = ("class Counter\n"
               "  include Enumerable(Int32)\n\n"
               "  def initialize(@max : Int32)\n"
               "  end\n\n"
               "  def each(&)\n"
               "    1.upto(@max) { |i| yield i }\n"
               "  end\nend")
    yield ruby, crystal


def rule_module_function():
    ruby = ("module MathUtil\n"
            "  module_function\n"
            "  def square(x); x * x; end\n"
            "end")
    crystal = ("module MathUtil\n"
               "  def self.square(x : Number)\n"
               "    x * x\n"
               "  end\n"
               "end")
    yield ruby, crystal


def rule_extend_self():
    ruby = ("module Helper\n"
            "  extend self\n"
            "  def greet; \"hi\"; end\n"
            "end")
    crystal = ("module Helper\n"
               "  extend self\n"
               "  def greet\n"
               "    \"hi\"\n"
               "  end\n"
               "end")
    yield ruby, crystal


def rule_global_var():
    ruby = '$config = { debug: true }'
    crystal = '# Crystal has no $globals — use a class var or module-level constant.\nmodule App\n  CONFIG = { debug: true }\nend'
    yield ruby, crystal


def rule_dollar_zero():
    ruby = "puts $0"
    crystal = "puts PROGRAM_NAME"
    yield ruby, crystal


def rule_stdin():
    ruby = 'line = $stdin.gets'
    crystal = 'line = STDIN.gets'
    yield ruby, crystal


def rule_stderr():
    ruby = '$stderr.puts "error"'
    crystal = 'STDERR.puts "error"'
    yield ruby, crystal


def rule_open_struct_alt():
    ruby = ('require "ostruct"\n'
            'point = OpenStruct.new(x: 1, y: 2)\n'
            'puts point.x')
    crystal = ('point = {x: 1, y: 2}\n'
               'puts point[:x]')
    yield ruby, crystal


def rule_logger():
    ruby = ('require "logger"\n'
            'log = Logger.new(STDOUT)\n'
            'log.info("started")')
    crystal = ('require "log"\n'
               'Log.info { "started" }')
    yield ruby, crystal


def rule_uri_parse():
    ruby = ('require "uri"\n'
            'u = URI.parse("https://example.com/path")\n'
            'puts u.host')
    crystal = ('uri = URI.parse("https://example.com/path")\n'
               'puts uri.host')
    yield ruby, crystal


def rule_uri_encode():
    ruby = 'enc = URI.encode_www_form_component("hello world")'
    crystal = 'enc = URI.encode_www_form("hello world")'
    yield ruby, crystal


def rule_base64():
    ruby = ('require "base64"\n'
            'enc = Base64.encode64(data)')
    crystal = ('require "base64"\n'
               'enc = Base64.encode(data)')
    yield ruby, crystal


def rule_digest_md5():
    ruby = ('require "digest"\n'
            'h = Digest::MD5.hexdigest("hello")')
    crystal = ('require "digest/md5"\n'
               'h = Digest::MD5.hexdigest("hello")')
    yield ruby, crystal


def rule_digest_sha256():
    ruby = ('require "digest"\n'
            'h = Digest::SHA256.hexdigest("hello")')
    crystal = ('require "digest/sha256"\n'
               'h = Digest::SHA256.hexdigest("hello")')
    yield ruby, crystal


def rule_openssl_hmac():
    ruby = ('require "openssl"\n'
            'sig = OpenSSL::HMAC.hexdigest("SHA256", key, msg)')
    crystal = ('require "openssl/hmac"\n'
               'sig = OpenSSL::HMAC.hexdigest(:sha256, key, msg)')
    yield ruby, crystal


def rule_yaml_parse():
    cases = [
        ('require "yaml"\nh = YAML.load_file("c.yml")',
         'require "yaml"\nh = YAML.parse(File.read("c.yml"))'),
        ('require "yaml"\nh = YAML.load(str)',
         'require "yaml"\nh = YAML.parse(str)'),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_csv_each():
    ruby = ('require "csv"\n'
            'CSV.foreach("data.csv") { |row| puts row.first }')
    crystal = ('require "csv"\n'
               'CSV.each_row(File.read("data.csv")) { |row| puts row.first }')
    yield ruby, crystal


def rule_each_with_object():
    cases = [
        ("[1,2,3].each_with_object([]) { |x, acc| acc << x*2 }",
         "[1, 2, 3].each_with_object([] of Int32) { |x, acc| acc << x*2 }"),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_inject_no_init():
    ruby = "[1,2,3].inject { |a, b| a + b }"
    crystal = "[1, 2, 3].reduce { |a, b| a + b }"
    yield ruby, crystal


def rule_each_slice():
    ruby = "[1,2,3,4,5].each_slice(2) { |s| puts s }"
    crystal = "[1, 2, 3, 4, 5].each_slice(2) { |s| puts s }"
    yield ruby, crystal


def rule_partition():
    ruby = "evens, odds = nums.partition(&:even?)"
    crystal = "evens, odds = nums.partition(&.even?)"
    yield ruby, crystal


def rule_group_by():
    cases = [
        ("words.group_by(&:length)",      "words.group_by(&.size)"),
        ("users.group_by(&:role)",        "users.group_by(&.role)"),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_tally():
    ruby = '%w(a b a c).tally'
    crystal = '%w(a b a c).tally'
    yield ruby, crystal


def rule_array_uniq():
    ruby = "arr.uniq { |x| x.id }"
    crystal = "arr.uniq { |x| x.id }"
    yield ruby, crystal


def rule_array_sort_by():
    cases = [
        ("users.sort_by(&:name)",        "users.sort_by(&.name)"),
        ("nums.sort_by { |n| -n }",      "nums.sort_by { |n| -n }"),
    ]
    for ruby, crystal in cases:
        yield ruby, crystal


def rule_array_max_by():
    ruby = "users.max_by(&:age)"
    crystal = "users.max_by(&.age)"
    yield ruby, crystal


def rule_array_filter_map():
    ruby = "arr.filter_map { |x| x*2 if x.even? }"
    crystal = "arr.compact_map { |x| x.even? ? x*2 : nil }"
    yield ruby, crystal


def rule_string_split_default():
    ruby = '"a b c".split'
    crystal = '"a b c".split'
    yield ruby, crystal


def rule_string_strip():
    ruby = '"  hi  ".strip'
    crystal = '"  hi  ".strip'
    yield ruby, crystal


def rule_string_squeeze():
    ruby = '"aaabbb".squeeze'
    crystal = '"aaabbb".squeeze'
    yield ruby, crystal


def rule_string_tr():
    ruby = '"abc".tr("a", "X")'
    crystal = '"abc".tr("a", "X")'
    yield ruby, crystal


def rule_array_flatten():
    ruby = "[[1,2],[3]].flatten"
    crystal = "[[1, 2], [3]].flatten"
    yield ruby, crystal


def rule_array_compact():
    ruby = "[1, nil, 2].compact"
    crystal = "[1, nil, 2].compact"
    yield ruby, crystal


def rule_hash_merge():
    ruby = "h1.merge(h2)"
    crystal = "h1.merge(h2)"
    yield ruby, crystal


def rule_hash_select():
    ruby = "h.select { |k, v| v > 0 }"
    crystal = "h.select { |k, v| v > 0 }"
    yield ruby, crystal


def rule_hash_transform_values():
    ruby = "h.transform_values { |v| v * 2 }"
    crystal = "h.transform_values { |v| v * 2 }"
    yield ruby, crystal


def rule_hash_to_a():
    ruby = "h.to_a"
    crystal = "h.to_a"
    yield ruby, crystal


def rule_hash_each():
    ruby = "h.each { |k, v| puts \"#{k}=#{v}\" }"
    crystal = "h.each { |k, v| puts \"#{k}=#{v}\" }"
    yield ruby, crystal


def rule_range_step():
    ruby = "(1..10).step(2) { |i| puts i }"
    crystal = "(1..10).step(2) { |i| puts i }"
    yield ruby, crystal


def rule_range_each():
    ruby = "(1..5).each { |i| puts i }"
    crystal = "(1..5).each { |i| puts i }"
    yield ruby, crystal


def rule_times():
    ruby = "5.times { puts 'hi' }"
    crystal = "5.times { puts \"hi\" }"
    yield ruby, crystal


def rule_upto():
    ruby = "1.upto(10) { |i| puts i }"
    crystal = "1.upto(10) { |i| puts i }"
    yield ruby, crystal


def rule_downto():
    ruby = "10.downto(1) { |i| puts i }"
    crystal = "10.downto(1) { |i| puts i }"
    yield ruby, crystal


def rule_loop_break():
    ruby = ("loop do\n"
            "  break if done?\n"
            "end")
    crystal = ("loop do\n"
               "  break if done?\n"
               "end")
    yield ruby, crystal


def rule_retry():
    ruby = ("begin\n"
            "  attempt\n"
            "rescue\n"
            "  retry\n"
            "end")
    crystal = ("attempts = 0\n"
               "begin\n"
               "  attempt\n"
               "rescue\n"
               "  attempts += 1\n"
               "  raise if attempts > 3\n"
               "  # Crystal has no `retry`; loop manually\n"
               "end")
    yield ruby, crystal


def rule_fail_keyword():
    ruby = 'fail "boom"'
    crystal = 'raise "boom"'
    yield ruby, crystal


def rule_singleton_method():
    ruby = "def obj.unique_method\n  42\nend"
    crystal = "# Crystal has no per-object singleton methods.\n# Either subclass or use a class-level method.\nclass MyType\n  def unique_method\n    42\n  end\nend"
    yield ruby, crystal


def rule_class_new_dynamic():
    ruby = "Klass = Class.new { def f; 1; end }"
    crystal = "# Crystal classes are compile-time; declare them statically.\nclass Klass\n  def f\n    1\n  end\nend"
    yield ruby, crystal


def rule_define_singleton_method():
    ruby = "obj.define_singleton_method(:bar) { 7 }"
    crystal = "# Crystal has no runtime method definition. Use a macro or subclass."
    yield ruby, crystal


def rule_attr_default():
    ruby = ("class User\n"
            "  attr_accessor :role\n"
            "  def initialize\n"
            "    @role = 'user'\n"
            "  end\nend")
    crystal = ("class User\n"
               "  property role : String = \"user\"\n"
               "end")
    yield ruby, crystal


def rule_initialize_with_defaults():
    ruby = ("class Box\n"
            "  def initialize(width: 10, height: 20)\n"
            "    @width, @height = width, height\n"
            "  end\nend")
    crystal = ("class Box\n"
               "  def initialize(@width : Int32 = 10, @height : Int32 = 20)\n"
               "  end\n"
               "end")
    yield ruby, crystal


def rule_yaml_serializable():
    ruby = ('require "yaml"\n'
            'class Config\n'
            '  attr_accessor :host\n'
            '  def to_yaml\n'
            '    { host: @host }.to_yaml\n'
            '  end\nend')
    crystal = ('require "yaml"\n'
               'class Config\n'
               '  include YAML::Serializable\n'
               '  property host : String\n\n'
               '  def initialize(@host : String)\n'
               '  end\n'
               'end')
    yield ruby, crystal


def rule_db_query():
    ruby = ('db = SQLite3::Database.new("a.db")\n'
            'rows = db.execute("select * from users")\n'
            'rows.each { |r| puts r["name"] }')
    crystal = ('require "db"\n'
               'require "sqlite3"\n'
               'DB.open("sqlite3://./a.db") do |db|\n'
               '  db.query("select * from users") do |rs|\n'
               '    rs.each do\n'
               '      puts rs.read(String)\n'
               '    end\n'
               '  end\n'
               'end')
    yield ruby, crystal


def rule_http_get():
    ruby = ('require "net/http"\n'
            'res = Net::HTTP.get_response(URI("https://example.com"))\n'
            'puts res.body')
    crystal = ('require "http/client"\n'
               'res = HTTP::Client.get("https://example.com")\n'
               'puts res.body')
    yield ruby, crystal


def rule_http_post():
    ruby = ('require "net/http"\n'
            'res = Net::HTTP.post_form(URI("https://api.example.com"), {"a" => "1"})')
    crystal = ('require "http/client"\n'
               'res = HTTP::Client.post_form("https://api.example.com", {"a" => "1"})')
    yield ruby, crystal


def rule_threads_to_fibers():
    ruby = ("threads = 5.times.map do |i|\n"
            "  Thread.new { do_work(i) }\n"
            "end\n"
            "threads.each(&:join)")
    crystal = ("done = Channel(Nil).new\n"
               "5.times do |i|\n"
               "  spawn do\n"
               "    do_work(i)\n"
               "    done.send(nil)\n"
               "  end\n"
               "end\n"
               "5.times { done.receive }")
    yield ruby, crystal


def rule_mutex():
    ruby = ("mutex = Mutex.new\n"
            "mutex.synchronize { @counter += 1 }")
    crystal = ("mutex = Mutex.new\n"
               "mutex.synchronize { @counter += 1 }")
    yield ruby, crystal


def rule_erb_template():
    ruby = ('require "erb"\n'
            'tpl = ERB.new("Hello, <%= name %>")\n'
            'puts tpl.result(binding)')
    crystal = ('require "ecr"\n'
               '# Crystal uses ECR (compile-time templates):\n'
               'name = "World"\n'
               'ECR.embed "tpl.ecr", io')
    yield ruby, crystal


NEW_RULES = [
    rule_json_parse, rule_json_serializable, rule_to_json_call,
    rule_file_read, rule_file_write, rule_file_each_line,
    rule_pathname, rule_dir_glob, rule_dir_exist, rule_file_exist,
    rule_random, rule_sleep, rule_process_pid, rule_system_exec,
    rule_env_default, rule_env_fetch,
    rule_regex_match, rule_regex_gsub,
    rule_string_chars, rule_string_each_char, rule_string_to_sym,
    rule_symbol_to_proc,
    rule_comparable_mixin, rule_enumerable_mixin, rule_iterable_mixin,
    rule_module_function, rule_extend_self,
    rule_global_var, rule_dollar_zero, rule_stdin, rule_stderr,
    rule_open_struct_alt, rule_logger,
    rule_uri_parse, rule_uri_encode,
    rule_base64, rule_digest_md5, rule_digest_sha256, rule_openssl_hmac,
    rule_yaml_parse, rule_csv_each,
    rule_each_with_object, rule_inject_no_init, rule_each_slice,
    rule_partition, rule_group_by, rule_tally,
    rule_array_uniq, rule_array_sort_by, rule_array_max_by, rule_array_filter_map,
    rule_string_split_default, rule_string_strip, rule_string_squeeze, rule_string_tr,
    rule_array_flatten, rule_array_compact,
    rule_hash_merge, rule_hash_select, rule_hash_transform_values,
    rule_hash_to_a, rule_hash_each,
    rule_range_step, rule_range_each, rule_times, rule_upto, rule_downto,
    rule_loop_break, rule_retry, rule_fail_keyword,
    rule_singleton_method, rule_class_new_dynamic, rule_define_singleton_method,
    rule_attr_default, rule_initialize_with_defaults,
    rule_yaml_serializable, rule_db_query,
    rule_http_get, rule_http_post,
    rule_threads_to_fibers, rule_mutex, rule_erb_template,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dpo_pairs_v3.jsonl")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--max-per-rule", type=int, default=20)
    args = ap.parse_args()

    # 1) Hand-curated 74 (from build_dpo_pairs.py)
    pairs = v2.load_original_pairs()
    n_hand = len(pairs)
    print(f"Loaded {n_hand} hand-curated pairs from build_dpo_pairs.py")

    # 2) v2 rules
    v2_count = 0
    for rule in v2.RULES:
        rule_pairs = []
        try:
            tidx = 0
            for ruby, crystal in rule():
                if len(rule_pairs) >= args.max_per_rule:
                    break
                tmpl = INSTR_TEMPLATES[tidx % len(INSTR_TEMPLATES)]
                tidx += 1
                rule_pairs.append(make_pair(ruby, crystal, tmpl))
        except Exception as e:
            print(f"  v2 rule {rule.__name__} errored: {e}")
            continue
        v2_count += len(rule_pairs)
        pairs.extend(rule_pairs)
    print(f"Generated {v2_count} v2-rule pairs from {len(v2.RULES)} rules")

    # 3) v3 new rules
    v3_count = 0
    rule_counts = {}
    for rule in NEW_RULES:
        rule_pairs = []
        try:
            tidx = 0
            for ruby, crystal in rule():
                if len(rule_pairs) >= args.max_per_rule:
                    break
                tmpl = INSTR_TEMPLATES[tidx % len(INSTR_TEMPLATES)]
                tidx += 1
                rule_pairs.append(make_pair(ruby, crystal, tmpl))
        except Exception as e:
            print(f"  v3 rule {rule.__name__} errored: {e}")
            continue
        rule_counts[rule.__name__] = len(rule_pairs)
        v3_count += len(rule_pairs)
        pairs.extend(rule_pairs)
    print(f"Generated {v3_count} v3-rule pairs from {len(NEW_RULES)} rules")

    n_total = len(pairs)
    print(f"\nTotal: {n_total} pairs ({n_hand} hand + {v2_count} v2 + {v3_count} v3)\n")
    print("Top v3 rules by pair count:")
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
            except subprocess.TimeoutExpired:
                bad += 1
            finally:
                Path(path).unlink(missing_ok=True)
        print(f"\nCompile validation: {ok} OK, {bad} fail, {noblock} no code block")

    out_path = REPO / args.out
    out_path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    print(f"\nWrote {n_total} pairs to {out_path}")


if __name__ == "__main__":
    main()
