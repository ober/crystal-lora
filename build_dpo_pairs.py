#!/usr/bin/env python3
"""
Build a DPO preference dataset of Ruby→Crystal "wrong → right" pairs.

Each entry is a complete Ruby snippet (rejected) and its idiomatic Crystal
equivalent (chosen). DPO turns these into preference triples so the model is
actively pushed *away* from Ruby-isms it would otherwise default to, not just
shown the right Crystal form once (which is what plain SFT does).

Output format: axolotl `chatml.argilla` schema (system, instruction,
chosen_response, rejected_response).

Usage:
  python3 build_dpo_pairs.py
  → dpo_pairs.jsonl
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "dpo_pairs.jsonl"

SYSTEM = (
    "You are an expert in Crystal, a statically-typed Ruby-syntax-inspired "
    "language compiled to native code via LLVM. Use only valid Crystal forms "
    "— never reach for Ruby, Java, or RSpec surface syntax. Crystal requires "
    "type annotations on instance variables and method arguments where they "
    "can't be inferred, uses `getter`/`setter`/`property` (not `attr_*`), "
    "uses `(T)` for generics (not `<T>`), uses `spawn`/`Channel(T)` (not "
    "`Thread.new`/`Queue`), uses the built-in `Spec` framework (not RSpec), "
    "and unions are written with `|` (`String?` = `String | Nil`)."
)


# Hand-curated wrong→right pairs. Each `wrong` is a Ruby snippet a model is
# likely to emit when asked for Crystal; each `correct` is the idiomatic
# Crystal equivalent. `src_lang` is shown to the model in the prompt.
PAIRS = [
    {
        "src_lang": "Ruby",
        "wrong": 'class User\n  attr_accessor :name\n  def initialize(name)\n    @name = name\n  end\nend',
        "correct": 'class User\n  property name : String\n  def initialize(@name : String)\n  end\nend',
        "notes": "Crystal uses `property`/`getter`/`setter` (typed) instead of Ruby's `attr_*`, "
                 "and the `def initialize(@name : String)` shorthand assigns and types in one step.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'class Account\n  attr_reader :id\n  attr_writer :secret\nend',
        "correct": 'class Account\n  getter id : Int64\n  setter secret : String\n  def initialize(@id : Int64, @secret : String)\n  end\nend',
        "notes": "`getter` = read-only, `setter` = write-only, `property` = both. All require types.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'class Box<T>\n  def initialize(value)\n    @value = value\n  end\nend',
        "correct": 'class Box(T)\n  getter value : T\n  def initialize(@value : T)\n  end\nend',
        "notes": "Crystal generics use parens: `Box(T)`, never `Box<T>`. Type parameters thread through "
                 "instance variables and method signatures.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'arr = []\narr << 1\narr << 2',
        "correct": 'arr = [] of Int32\narr << 1\narr << 2',
        "notes": "Empty array/hash literals need an explicit element type: `[] of Int32`, "
                 "`{} of String => Int32`. With one or more elements, Crystal infers.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'h = {}\nh["a"] = 1',
        "correct": 'h = {} of String => Int32\nh["a"] = 1',
        "notes": "An empty hash needs `{} of K => V`; a non-empty literal like `{\"a\" => 1}` infers.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'require "rspec"\n\ndescribe User do\n  it "has a name" do\n    expect(User.new("Ada").name).to eq("Ada")\n  end\nend',
        "correct": 'require "spec"\nrequire "./user"\n\ndescribe User do\n  it "has a name" do\n    User.new("Ada").name.should eq("Ada")\n  end\nend',
        "notes": "Crystal ships its own `Spec` framework (`require \"spec\"`, not `rspec`). "
                 "Assertions use `value.should eq(...)`, not `expect(value).to eq(...)`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'Thread.new do\n  do_work\nend.join',
        "correct": 'done = Channel(Nil).new\nspawn do\n  do_work\n  done.send(nil)\nend\ndone.receive',
        "notes": "Crystal uses fibers (`spawn`) and channels for concurrency. There is no public "
                 "`Thread.new`; for joining, send a sentinel through a `Channel(Nil)`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'q = Queue.new\nproducer = Thread.new { q << 42 }\nputs q.pop',
        "correct": 'ch = Channel(Int32).new(1)\nspawn { ch.send(42) }\nputs ch.receive',
        "notes": "`Queue`/`SizedQueue` → typed `Channel(T)`. The buffer size is a constructor arg.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'def add(a, b)\n  a + b\nend',
        "correct": 'def add(a : Int32, b : Int32) : Int32\n  a + b\nend',
        "notes": "Method args usually need types in Crystal (when not inferable from a generic context). "
                 "Return type is optional but recommended on public API.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'def initialize(name, age)\n  @name = name\n  @age = age\nend',
        "correct": 'def initialize(@name : String, @age : Int32)\nend',
        "notes": "The `@name : Type` parameter shorthand assigns the instance variable and ascribes "
                 "its type in one step.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'data = JSON.parse(json_str)\nputs data["name"]',
        "correct": 'require "json"\n\nclass User\n  include JSON::Serializable\n  property name : String\nend\n\nuser = User.from_json(json_str)\nputs user.name',
        "notes": "`JSON.parse` returns a `JSON::Any` (untyped). For typed access, define a class with "
                 "`include JSON::Serializable` and use `T.from_json(str)` / `obj.to_json`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'def greet(name)\n  raise ArgumentError, "blank" if name.empty?\n  "Hi, #{name}"\nend',
        "correct": 'def greet(name : String) : String\n  raise ArgumentError.new("blank") if name.empty?\n  "Hi, #{name}"\nend',
        "notes": "Crystal raises with `ExceptionClass.new(\"msg\")`, not the Ruby `raise Class, \"msg\"` "
                 "two-arg form.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'begin\n  risky\nrescue => e\n  puts e.message\nend',
        "correct": 'begin\n  risky\nrescue ex\n  puts ex.message\nend',
        "notes": "Crystal uses `rescue ex` (no `=>`), and you can specify a class: `rescue ex : IO::Error`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'puts foo&.bar&.baz',
        "correct": 'if (f = foo) && (b = f.bar)\n  puts b.baz\nend',
        "notes": "Crystal has no `&.` safe-nav operator. Narrow with assignment-in-condition, or "
                 "use `foo.try(&.bar).try(&.baz)`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'def lookup(key)\n  return nil unless @cache.has_key?(key)\n  @cache[key].upcase\nend',
        "correct": 'def lookup(key : String) : String?\n  return nil unless @cache.has_key?(key)\n  @cache[key].upcase\nend',
        "notes": "When a method can return `nil`, the return type must be a nilable union (`String?` = "
                 "`String | Nil`), or callers will see a type error.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'class Animal\n  def speak\n    raise NotImplementedError\n  end\nend',
        "correct": 'abstract class Animal\n  abstract def speak : String\nend',
        "notes": "Crystal has first-class `abstract class` and `abstract def`; subclasses are required "
                 "by the compiler to implement them — no runtime `NotImplementedError` dance needed.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'extern "C" do\n  attach_function :puts, [:string], :int\nend',
        "correct": 'lib LibC\n  fun puts(str : LibC::Char*) : LibC::Int\nend\n\nLibC.puts("hello")',
        "notes": "Crystal's FFI uses `lib LibName ... fun name(arg : Type) : Ret end`. There is no "
                 "`extern \"C\"` block or `attach_function` call.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'class Cache\n  def self.instance\n    @@instance ||= new\n  end\nend',
        "correct": 'class Cache\n  @@instance : self?\n  def self.instance : self\n    @@instance ||= new\n  end\nend',
        "notes": "Class variables need a type ascription. Method return types make singletons "
                 "type-check cleanly.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'def transform(xs, &block)\n  xs.map(&block)\nend',
        "correct": 'def transform(xs : Array(Int32), &block : Int32 -> Int32) : Array(Int32)\n  xs.map { |x| block.call(x) }\nend',
        "notes": "Block parameters get a typed proc signature `&block : ArgT -> RetT`. "
                 "Inside the body, call as `block.call(arg)`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'h = Hash.new(0)\nh["x"] += 1',
        "correct": 'h = Hash(String, Int32).new(0)\nh["x"] += 1',
        "notes": "`Hash.new(default)` needs explicit `Hash(K, V)` so the compiler knows what types "
                 "the default and the values are.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'arr = Array.new(5)',
        "correct": 'arr = Array(Int32).new(5, 0)',
        "notes": "`Array.new(n)` in Crystal requires both the element type and a default value: "
                 "`Array(T).new(n, default)`. There is no `nil`-filled untyped array.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'class User\n  def method_missing(name, *args)\n    @attrs[name.to_s]\n  end\nend',
        "correct": 'class User\n  macro method_missing(call)\n    @attrs[{{ call.name.stringify }}]\n  end\nend',
        "notes": "Crystal compiles — there is no runtime `method_missing`. Use the `method_missing` "
                 "*macro* (compile-time) for dispatch by name.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'name = nil\nputs name.length',
        "correct": 'name = nil.as(String?)\nif name\n  puts name.size\nend',
        "notes": "Calling a method on something that might be `nil` is a compile error. Narrow with "
                 "`if name` (after which Crystal knows it's non-nil) or `name.try &.size`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'enum Color\n  RED = 1\n  GREEN = 2\nend',
        "correct": 'enum Color\n  Red\n  Green\nend',
        "notes": "Crystal enums use UpperCamelCase members and auto-number from 0 unless you assign. "
                 "(They're a real type — `Color::Red.value # => 0`.)",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'send(:greet, "world")',
        "correct": 'case action\nwhen "greet" then greet("world")\nelse raise "unknown action: #{action}"\nend',
        "notes": "Crystal has no dynamic `send` — methods are dispatched at compile time. Use a "
                 "`case` over known names, or a macro for compile-time codegen.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'eval("puts 1 + 2")',
        "correct": 'macro emit_calc\n  puts 1 + 2\nend\n\nemit_calc',
        "notes": "Crystal compiles. There is no `eval`. Use a macro for compile-time code generation.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'require_relative "./user"\nrequire_relative "../lib/helper"',
        "correct": 'require "./user"\nrequire "../lib/helper"',
        "notes": "Crystal's relative `require` already takes a path-like string; there is no "
                 "`require_relative`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'mutex = Mutex.new\nThread.new do\n  mutex.synchronize { @counter += 1 }\nend',
        "correct": 'mutex = Mutex.new\nspawn do\n  mutex.synchronize { @counter += 1 }\nend',
        "notes": "`Mutex#synchronize` exists in Crystal with the same API. The change is `Thread.new` "
                 "→ `spawn` (fiber).",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'class Box\n  def initialize(value)\n    @value = value\n  end\nend\nb = Box.new(1)',
        "correct": 'class Box(T)\n  getter value : T\n  def initialize(@value : T)\n  end\nend\nb = Box(Int32).new(1)',
        "notes": "If a class wraps something whose type isn't known up front, parameterise it: "
                 "`class Box(T)` and use as `Box(Int32).new(1)`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'maybe_user = find_user(id)\nputs maybe_user.name',
        "correct": 'if user = find_user(id)\n  puts user.name\nelse\n  puts "not found"\nend',
        "notes": "If `find_user` may return `nil`, dereference inside an `if user = ...` so the "
                 "compiler narrows `user` to non-nil.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'def info(opts = {})\n  puts opts[:name]\nend\ninfo(name: "Ada")',
        "correct": 'def info(*, name : String)\n  puts name\nend\ninfo(name: "Ada")',
        "notes": "Crystal supports real keyword arguments (with types). The `*` separator forces "
                 "callers to use the `name:` keyword.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'class Logger\n  @@instance = nil\n  def self.global\n    @@instance ||= new\n  end\nend',
        "correct": 'class Logger\n  @@instance : self?\n  def self.global : self\n    @@instance ||= new\n  end\nend',
        "notes": "Class variables must be type-ascribed in Crystal. `self?` here is `Logger | Nil`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'arr.first  # returns nil if empty',
        "correct": 'arr.first?  # returns nil if empty (raises with first)',
        "notes": "`Array#first` *raises* on empty in Crystal. The `?`-suffixed `first?` / `last?` are "
                 "the nil-returning variants.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'class Cat < Animal\n  def speak; "meow"; end\nend',
        "correct": 'class Cat < Animal\n  def speak : String\n    "meow"\n  end\nend',
        "notes": "Annotate the return type for any method that overrides an abstract method or that's "
                 "part of a public interface — otherwise Crystal's union inference can surprise you.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'puts foo.respond_to?(:bar) ? foo.bar : "n/a"',
        "correct": 'puts foo.responds_to?(:bar) ? foo.bar : "n/a"',
        "notes": "Same idea, different name: `responds_to?` (with the `s`). It's compile-time when "
                 "the receiver type is known.",
    },
    {
        "src_lang": "Ruby/Java",
        "wrong": 'pool = ThreadPoolExecutor.new(4)\npool.submit { do_work }',
        "correct": 'wg = WaitGroup.new\n4.times do\n  wg.spawn { do_work }\nend\nwg.wait',
        "notes": "Crystal has no public thread pool. Use `WaitGroup` + `spawn` for fan-out, or "
                 "`ExecutionContext` for parallel execution across cores.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'a, b = [1, 2]',
        "correct": 'a, b = 1, 2',
        "notes": "Crystal multi-assign uses tuple values directly (`a, b = 1, 2`), which produces a "
                 "`Tuple(Int32, Int32)` and unpacks. Destructuring an `Array` doesn't work because "
                 "the size isn't statically known.",
    },

    # ── Spec framework: matchers and lifecycle ─────────────────────────
    {
        "src_lang": "Ruby/RSpec",
        "wrong": 'expect(user.name).to eq("Ada")\nexpect(user.age).to be > 18',
        "correct": 'user.name.should eq("Ada")\nuser.age.should be > 18',
        "notes": "Crystal `Spec` flips the syntax: `value.should matcher`, not `expect(value).to matcher`. "
                 "Matchers like `eq`, `be`, `be_a`, `be_nil`, `be_truthy`, `contain` are available.",
    },
    {
        "src_lang": "Ruby/RSpec",
        "wrong": 'expect { risky! }.to raise_error(IO::Error)',
        "correct": 'expect_raises(IO::Error) do\n  risky!\nend',
        "notes": "`expect_raises(ErrorClass)` is the Crystal Spec idiom — it's a top-level helper, not "
                 "a matcher chained off `expect`.",
    },
    {
        "src_lang": "Ruby/RSpec",
        "wrong": 'before(:each) do\n  @repo = Repo.new\nend\n\nafter(:each) do\n  @repo.close\nend',
        "correct": 'repo = uninitialized Repo\n\nbefore_each do\n  repo = Repo.new\nend\n\nafter_each do\n  repo.close\nend',
        "notes": "Crystal Spec uses `before_each` / `after_each` (no `:each` symbol). State is held in "
                 "local variables in the enclosing `describe` block, not in `@`-instance vars on the "
                 "example group.",
    },
    {
        "src_lang": "Ruby/RSpec",
        "wrong": 'let(:user) { User.new("Ada") }\n\nit "has a name" do\n  expect(user.name).to eq("Ada")\nend',
        "correct": 'describe User do\n  it "has a name" do\n    user = User.new("Ada")\n    user.name.should eq("Ada")\n  end\nend',
        "notes": "Crystal Spec has no `let` / `subject`. Build the value inline in each `it` block, or "
                 "share via `before_each` writing to a local variable.",
    },

    # ── Time / Date ─────────────────────────────────────────────────────
    {
        "src_lang": "Ruby",
        "wrong": 'now = Time.now\nputs now',
        "correct": 'now = Time.local\nputs now\n\n# UTC variant:\nputs Time.utc',
        "notes": "Crystal has no `Time.now`. Use `Time.local` (system timezone) or `Time.utc`. Both "
                 "return a `Time` value with sub-second precision.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'Date.today\nDate.parse("2024-01-15")',
        "correct": 'Time.local.date\nTime.parse_iso8601("2024-01-15T00:00:00Z")',
        "notes": "Crystal has no separate `Date` class; `Time` covers it. For ISO 8601 strings use "
                 "`Time.parse_iso8601`; for other formats use `Time.parse(str, format, location)`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": '5.minutes.ago\n3.days.from_now',
        "correct": 'Time.local - 5.minutes\nTime.local + 3.days',
        "notes": "Crystal has no ActiveSupport-style `5.minutes.ago` shortcut. The `5.minutes` part "
                 "exists (returns a `Time::Span`); subtract or add it from a `Time` explicitly.",
    },

    # ── HTTP / Process / File / ENV ─────────────────────────────────────
    {
        "src_lang": "Ruby",
        "wrong": 'require "net/http"\nbody = Net::HTTP.get(URI("https://example.com"))',
        "correct": 'require "http/client"\n\nresponse = HTTP::Client.get("https://example.com")\nbody = response.body',
        "notes": "Crystal's HTTP client is `HTTP::Client` (in `http/client`). It returns an "
                 "`HTTP::Client::Response`; the body is on `.body`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'output = `ls -la`\nsystem("git status")',
        "correct": 'output = `ls -la`\nProcess.run("git", args: ["status"], output: STDOUT, error: STDERR)',
        "notes": "Backticks work and capture stdout. For full control use `Process.run` with explicit "
                 "args (no shell interpolation, safe by default) and explicit IO routing.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'home = ENV["HOME"]\nputs home.upcase',
        "correct": 'home = ENV["HOME"]?\nif home\n  puts home.upcase\nelse\n  puts "$HOME unset"\nend',
        "notes": "`ENV[key]` raises if missing; use `ENV[key]?` for the nilable form, then narrow with "
                 "`if`. Calling `.upcase` on a `String?` is a compile error.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'File.open("data.txt", "r") do |f|\n  f.each_line { |l| puts l }\nend',
        "correct": 'File.open("data.txt") do |f|\n  f.each_line { |l| puts l }\nend\n\n# Or one-shot:\nFile.read("data.txt").each_line { |l| puts l }',
        "notes": "`File.open` defaults to read mode; the second arg is optional. For small files, "
                 "`File.read(path)` (whole file as `String`) or `File.each_line(path)` are simpler.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'Dir["**/*.cr"].each { |path| puts path }',
        "correct": 'Dir.glob("**/*.cr").each { |path| puts path }\n\n# Or, equivalently:\nDir["**/*.cr"].each { |path| puts path }',
        "notes": "`Dir[pattern]` and `Dir.glob(pattern)` both work in Crystal. They return "
                 "`Array(String)`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'rand(100)\nsrand(42)',
        "correct": 'rand(100)\nRandom::DEFAULT.new_seed(42_u64)\n\n# Or a private generator:\nrng = Random.new(42_u64)\nrng.rand(100)',
        "notes": "`rand(n)` works in Crystal. Seeding the global RNG uses "
                 "`Random::DEFAULT.new_seed`; for a private seedable RNG, construct `Random.new(seed)`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'require "optparse"\nOptionParser.new do |opts|\n  opts.on("-v", "--verbose") { @v = true }\nend.parse!',
        "correct": 'require "option_parser"\n\nverbose = false\nOptionParser.parse do |opts|\n  opts.on("-v", "--verbose", "Verbose output") { verbose = true }\nend',
        "notes": "Crystal's parser is `option_parser` (snake_case `require`), and the entry point is "
                 "`OptionParser.parse do |opts| ... end` — no `.parse!` call.",
    },

    # ── Iteration / Enumerable ──────────────────────────────────────────
    {
        "src_lang": "Ruby",
        "wrong": 'arr.detect { |x| x > 10 }',
        "correct": 'arr.find { |x| x > 10 }',
        "notes": "Crystal has `find` (aliased on `Enumerable`); there is no `detect`. Returns "
                 "`T?` — `nil` if no element matches.",
    },
    {
        "src_lang": "Ruby",
        "wrong": '[1, 2, 3].inject(0) { |sum, x| sum + x }',
        "correct": '[1, 2, 3].reduce(0) { |sum, x| sum + x }',
        "notes": "Crystal uses `reduce` (also accepts no-initial-value form). `inject` is not the "
                 "primary name in Crystal stdlib.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'arr.group_by { |x| x.kind }\n   .transform_values { |xs| xs.size }',
        "correct": 'arr.group_by { |x| x.kind }\n   .transform_values { |xs| xs.size }',
        "notes": "Both work in Crystal — but the block return type drives the resulting "
                 "`Hash(K, Array(T))` types, so a block that mixes return types causes inference to "
                 "widen to a union. Annotate the block if you need a specific key type.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'people.min_by(&:age)',
        "correct": 'people.min_by(&.age)',
        "notes": "Crystal's short-block syntax is `&.method`, not `&:method`. It's a proc that calls "
                 "the named method on its argument.",
    },
    {
        "src_lang": "Ruby",
        "wrong": '[3, 1, 2].sort { |a, b| b <=> a }',
        "correct": '[3, 1, 2].sort { |a, b| b <=> a }',
        "notes": "Same syntax, but in Crystal the block must return an `Int32` (or "
                 "`Comparable::Result`), not a generic `<=>` value. Sorting `Float` requires "
                 "`sort_by(&.itself)` because `NaN`-vs-`<=>` returns nilable.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'h = {a: 1, b: 2}\nh.each { |k, v| puts "#{k}=#{v}" }',
        "correct": 'h = {a: 1, b: 2}              # NamedTuple(a: Int32, b: Int32)\nh.each { |k, v| puts "#{k}=#{v}" }\n\n# For a real Hash:\nh2 = {"a" => 1, "b" => 2}     # Hash(String, Int32)\nh2.each { |k, v| puts "#{k}=#{v}" }',
        "notes": "`{a: 1}` is a `NamedTuple` in Crystal (compile-time keys), *not* a `Hash`. For "
                 "string-keyed runtime hashes, use `=>`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'data.dig(:user, :address, :city)',
        "correct": 'data.dig?("user", "address", "city")',
        "notes": "`#dig?` returns the nilable form (`nil` on missing key). `#dig` raises. Keys are the "
                 "actual hash keys — symbols only when the hash is `Hash(Symbol, _)`.",
    },

    # ── Numeric / type traps ────────────────────────────────────────────
    {
        "src_lang": "Ruby",
        "wrong": 'big = 10 ** 20',
        "correct": 'require "big"\n\nbig = BigInt.new(10) ** 20',
        "notes": "Crystal's primitive `Int64`/`UInt64` *overflow*. For arbitrary precision use "
                 "`BigInt` from `require \"big\"`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'avg = total / count',
        "correct": 'avg = total / count        # Float64 if either operand is Float\navg_int = total // count   # truncated integer division (when both are Int)',
        "notes": "`Int / Int` returns `Float64` in Crystal (a divergence from Ruby and most other "
                 "static-typed languages). For integer division use `//`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'def transform(x)\n  if x.is_a?(Integer)\n    x * 2\n  else\n    x.to_s\n  end\nend',
        "correct": 'def transform(x : Int32 | String) : Int32 | String\n  case x\n  when Int32  then x * 2\n  when String then x.to_s\n  else             raise "unreachable"\n  end\nend',
        "notes": "Crystal's `case ... when Type` narrows the type inside each branch (no `is_a?` "
                 "needed). Annotate the union explicitly so callers know what comes in.",
    },

    # ── Blocks, procs, and short-block syntax ───────────────────────────
    {
        "src_lang": "Ruby",
        "wrong": 'doubler = ->(x) { x * 2 }\nputs doubler.call(5)',
        "correct": 'doubler = ->(x : Int32) { x * 2 }\nputs doubler.call(5)',
        "notes": "Proc literals in Crystal need types on the parameters. The result is a "
                 "`Proc(Int32, Int32)`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": '[1, 2, 3].map(&:to_s)',
        "correct": '[1, 2, 3].map(&.to_s)',
        "notes": "The Crystal short-block is `&.method`. The Ruby `&:symbol` form does not work for "
                 "method calls.",
    },

    # ── Macros / compile-time hooks ─────────────────────────────────────
    {
        "src_lang": "Ruby",
        "wrong": 'class Logger\n  def self.included(base)\n    base.extend(ClassMethods)\n  end\nend',
        "correct": 'module Logger\n  macro included\n    extend ClassMethods\n  end\nend',
        "notes": "`included`, `inherited`, and `extended` are macro hooks in Crystal — they run at "
                 "compile time. Use them inside a `module` or `class`.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'class Base\n  def self.inherited(sub)\n    REGISTRY << sub\n  end\nend',
        "correct": 'class Base\n  REGISTRY = [] of Base.class\n\n  macro inherited\n    Base::REGISTRY << {{@type}}\n  end\nend',
        "notes": "Use the `inherited` macro and `{{@type}}` to refer to the inheriting class at "
                 "compile time. The registry must be typed.",
    },
    {
        "src_lang": "Ruby",
        "wrong": '[:get, :post, :put].each do |verb|\n  define_method(verb) { |path, &block| ... }\nend',
        "correct": '{% for verb in %w(get post put) %}\n  def {{verb.id}}(path : String, &block)\n    # ...\n  end\n{% end %}',
        "notes": "Compile-time codegen uses `{% for %}` ... `{% end %}` blocks with `{{ }}` "
                 "interpolation. Macro `id` strips the quotes around a string literal so you can "
                 "splice it as a method name.",
    },

    # ── Object identity / introspection ─────────────────────────────────
    {
        "src_lang": "Ruby",
        "wrong": 'a.equal?(b)\na.object_id',
        "correct": 'a.same?(b)\na.object_id   # exists, but only on Reference (not Value types)',
        "notes": "Reference identity is `same?` in Crystal. `object_id` exists on `Reference` but not "
                 "on `Value` (struct/Int/Float etc.) because value types have no stable identity.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'a.is_a?(String) && a.length > 0',
        "correct": 'if a.is_a?(String) && a.size > 0\n  # `a` is narrowed to String here\nend',
        "notes": "`is_a?` works the same way; in an `if`/`&&` chain it narrows the variable's type. "
                 "Note: it's `.size` (not `.length`) on `String` and most collections.",
    },

    # ── String / collection method-name traps ───────────────────────────
    {
        "src_lang": "Ruby",
        "wrong": 'name.length',
        "correct": 'name.size',
        "notes": "Crystal uses `size` on `String`, `Array`, `Hash`, etc. `length` is not a method on "
                 "`String` (`bytesize` is the byte count).",
    },
    {
        "src_lang": "Ruby",
        "wrong": '"abc"[1]   # => "b"',
        "correct": '"abc"[1]   # => \'b\' (Char, not String)\n"abc"[1, 1]   # => "b" (substring)\n"abc"[1..1]   # => "b"',
        "notes": "`String#[]` with a single `Int` returns a `Char`, not a one-character `String`. "
                 "Use the `(start, length)` or range form for substring slicing.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'arr.delete(item)\narr.compact!',
        "correct": 'arr.delete(item)        # returns the deleted element or nil\narr.reject!(&.nil?)     # in-place compact\n\n# Or non-mutating:\narr2 = arr.compact      # Array(T) without nils, requires nilable element type',
        "notes": "Crystal's `compact` requires the element type to be nilable. There's no in-place "
                 "`compact!` per se — use `reject!(&.nil?)` for the in-place variant.",
    },

    # ── FFI / Pointer / Slice ──────────────────────────────────────────
    {
        "src_lang": "Ruby/C",
        "wrong": 'lib LibFoo\n  fun process(buf : UInt8*, len : Int32) : Int32\nend\n\nbuf = "hello"\nLibFoo.process(buf, buf.size)',
        "correct": 'lib LibFoo\n  fun process(buf : UInt8*, len : Int32) : Int32\nend\n\nbuf = "hello"\nLibFoo.process(buf.to_unsafe, buf.bytesize)',
        "notes": "Pass `String` to a `UInt8*` parameter via `.to_unsafe`. Use `.bytesize` (bytes), "
                 "not `.size` (UTF-8 character count). Same applies to `Bytes` and `Slice`.",
    },
    {
        "src_lang": "Ruby/C",
        "wrong": 'arr = [1_u8, 2_u8, 3_u8]\nLibFoo.process(arr, arr.size)',
        "correct": 'slice = Slice[1_u8, 2_u8, 3_u8]\nLibFoo.process(slice.to_unsafe, slice.size)',
        "notes": "For a contiguous fixed-size buffer, use `Slice(T)` (or `Bytes` = `Slice(UInt8)`); "
                 "`.to_unsafe` gives a `Pointer(T)` you can pass to FFI.",
    },
    {
        "src_lang": "Ruby",
        "wrong": 'value.not_nil!',
        "correct": 'if value\n  value.something\nelse\n  raise "expected non-nil"\nend\n\n# Only when you really, truly know it is non-nil:\nvalue.not_nil!',
        "notes": "`not_nil!` raises on `nil`. Prefer narrowing with `if value` so the compiler "
                 "knows the type. Reserve `not_nil!` for cases where the invariant is enforced "
                 "elsewhere and you want a fail-fast check.",
    },
]


def render_block(code: str) -> str:
    return f"```crystal\n{code.rstrip()}\n```"


def render_block_other(code: str, lang: str = "ruby") -> str:
    # Use ruby fence for the Ruby snippet so syntax-highlighting in any
    # downstream renderer is honest about what it is.
    return f"```{lang}\n{code.rstrip()}\n```"


def build_pair(p: dict) -> dict:
    src = p["src_lang"]
    wrong = p["wrong"].strip()
    correct = p["correct"].strip()
    notes = p.get("notes", "").strip()
    src_fence = "ruby" if "Ruby" in src else "text"

    instruction = (
        f"Show me how to write the Crystal-idiomatic version of this {src} form:\n\n"
        f"{render_block_other(wrong, src_fence)}"
    )

    chosen_parts = [render_block(correct)]
    if notes:
        chosen_parts.append(notes)
    chosen = "\n\n".join(chosen_parts)

    rejected = render_block_other(wrong, src_fence)

    return {
        "system": SYSTEM,
        "instruction": instruction,
        "chosen_response": chosen,
        "rejected_response": rejected,
    }


def main():
    n = 0
    with open(OUT, "w") as out:
        for p in PAIRS:
            out.write(json.dumps(build_pair(p)) + "\n")
            n += 1
    print(f"Wrote {OUT}  ({n} pairs)")


if __name__ == "__main__":
    main()
