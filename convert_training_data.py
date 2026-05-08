#!/usr/bin/env python3
"""
Convert Crystal language sources into LoRA training data.

Sources (all live under ~/mine/):
  1. crystal/src/**/*.cr            - Stdlib source. Extracts doc-comments + signatures.
  2. crystal/samples/*.cr           - Runnable example programs.
  3. crystal/spec/std/**/*.cr       - Spec files showing real usage.
  4. crystal/doc/man/*.adoc         - CLI man pages (crystal, crystal-build, ...).
  5. crystal/doc/changelogs/*.md    - Per-version changelogs.
  6. crystal/{README,CONTRIBUTING,UPGRADING,SECURITY,NOTICE}.md
  7. crystal-mcp/src/**/*.cr        - Real-world Crystal codebase (an MCP server).
  8. crystal-mcp/spec/**/*.cr       - Spec patterns for a real project.
  9. Hand-written CONVENTION_EXAMPLES — Crystal-vs-Ruby idioms (3x weighted).

Output formats:
  - training_data.jsonl           ChatML/ShareGPT (LLaMA-Factory, Axolotl)
  - training_data_together.jsonl  Together AI {messages: [...]}
  - training_data_alpaca.jsonl    Alpaca {instruction, input, output}
  - training_data_alpaca.json     Pretty-printed Alpaca

Each entry is a single-turn conversation teaching the model about Crystal.
"""

import json
import os
import re
import glob
import hashlib

# ── Paths ────────────────────────────────────────────────────────────
CRYSTAL_DIR = os.path.expanduser("~/mine/crystal")
MCP_DIR     = os.path.expanduser("~/mine/crystal-mcp")
OUTPUT_DIR  = os.path.expanduser("~/mine/crystal-lora")

SYSTEM_PROMPT = (
    "You are an expert in Crystal, a statically-typed, Ruby-syntax-inspired "
    "language that compiles to native code via LLVM. You provide accurate, "
    "idiomatic Crystal code with correct types, `require` statements, and "
    "method signatures. Crystal looks like Ruby but is type-checked at "
    "compile time: no `method_missing`, no `eval`, methods overload on "
    "argument types, unions are first-class (`String | Nil`, abbreviated "
    "`String?`), generics use parens (`Array(Int32)`, `Hash(String, Int32)`), "
    "and macros run at compile time. Use `getter`/`setter`/`property` instead "
    "of Ruby's `attr_*`. Concurrency uses fibers and channels (`spawn`, "
    "`Channel(T)`). The standard test framework is the built-in `Spec` "
    "(via `require \"spec\"`), not RSpec. When writing code, always include "
    "the needed `require` statements."
)


def normalize_code(code: str) -> str:
    """Normalize escaped newlines and clean up code strings."""
    code = code.replace("\\n", "\n")
    code = code.replace("\\\"", '"')
    code = code.replace("\\\\", "\\")
    code = re.sub(r'\n{3,}', '\n\n', code)
    return code.rstrip()


# ═══════════════════════════════════════════════════════════════════════
# Crystal Convention Teaching Examples (hand-written, 3x weighted)
# ═══════════════════════════════════════════════════════════════════════

CONVENTION_EXAMPLES = [
    {
        "q": "How do I start a Crystal program?",
        "a": "Crystal files end in `.cr`. Use `require` to pull in stdlib modules "
             "or shards. There's no boilerplate `main` — top-level code runs at "
             "program start:\n\n"
             "```crystal\n"
             "require \"json\"\n\n"
             "def greet(name : String) : String\n"
             "  \"Hello, #{name}!\"\n"
             "end\n\n"
             "puts greet(\"World\")\n"
             "```\n\n"
             "Compile with `crystal build foo.cr` (or `crystal build --release foo.cr` "
             "for an optimised binary) and run with `crystal run foo.cr` for an "
             "interpreter-style workflow.",
    },
    {
        "q": "How do I declare types and type unions in Crystal?",
        "a": "Crystal infers most types but you can ascribe them. A union is "
             "expressed with `|`, and a nilable type has the shorthand `T?`:\n\n"
             "```crystal\n"
             "name : String = \"Ada\"\n"
             "age : Int32 = 36\n"
             "maybe_name : String? = nil      # same as String | Nil\n"
             "id : Int32 | String = \"abc\"     # explicit union\n"
             "```\n\n"
             "Inside a conditional, Crystal *narrows* the type:\n\n"
             "```crystal\n"
             "def length_of(s : String?) : Int32\n"
             "  if s\n"
             "    s.size  # here `s` is narrowed to String\n"
             "  else\n"
             "    0\n"
             "  end\n"
             "end\n"
             "```",
    },
    {
        "q": "What's the Crystal equivalent of Ruby's attr_accessor?",
        "a": "Crystal uses `getter`, `setter`, and `property` — these are macros "
             "that generate the methods at compile time. They also accept type "
             "annotations:\n\n"
             "```crystal\n"
             "class User\n"
             "  property name : String\n"
             "  getter id : Int64       # read-only\n"
             "  setter password : String # write-only\n\n"
             "  def initialize(@name : String, @id : Int64, @password : String)\n"
             "  end\n"
             "end\n"
             "```\n\n"
             "Note the `@var : Type` shorthand in `initialize` — it both takes the "
             "argument and assigns it to the instance variable in one step.",
    },
    {
        "q": "How do generics work in Crystal?",
        "a": "Generics use parentheses, not angle brackets. Type variables are "
             "single uppercase letters by convention:\n\n"
             "```crystal\n"
             "class Stack(T)\n"
             "  @items = [] of T\n\n"
             "  def push(x : T) : Nil\n"
             "    @items.push x\n"
             "  end\n\n"
             "  def pop : T?\n"
             "    @items.pop?\n"
             "  end\n"
             "end\n\n"
             "stack = Stack(Int32).new\n"
             "stack.push(1)\n"
             "stack.push(2)\n"
             "puts stack.pop  # => 2\n"
             "```\n\n"
             "Empty literals need an explicit type: `[] of Int32`, "
             "`{} of String => Int32`, `Set(String).new`.",
    },
    {
        "q": "How do I spawn fibers and use channels in Crystal?",
        "a": "Crystal has lightweight fibers (cooperative green threads) via "
             "`spawn`, and typed channels via `Channel(T)`:\n\n"
             "```crystal\n"
             "channel = Channel(Int32).new\n\n"
             "spawn do\n"
             "  10.times do |i|\n"
             "    channel.send(i * i)\n"
             "  end\n"
             "  channel.close\n"
             "end\n\n"
             "while value = channel.receive?\n"
             "  puts value\n"
             "end\n"
             "```\n\n"
             "`receive?` returns `Nil` on a closed channel; `receive` raises "
             "`Channel::ClosedError`. For multiplexing, use `Channel.select`.",
    },
    {
        "q": "How do I handle exceptions in Crystal?",
        "a": "Crystal uses `begin/rescue/ensure/end` (Ruby-style). Custom exception "
             "types subclass `Exception`:\n\n"
             "```crystal\n"
             "class ParseError < Exception\n"
             "end\n\n"
             "def parse_int(s : String) : Int32\n"
             "  begin\n"
             "    s.to_i\n"
             "  rescue ArgumentError\n"
             "    raise ParseError.new(\"not an integer: #{s}\")\n"
             "  end\n"
             "end\n"
             "```\n\n"
             "Inside a method body, you can omit the `begin`:\n\n"
             "```crystal\n"
             "def parse_int(s : String) : Int32\n"
             "  s.to_i\n"
             "rescue ArgumentError\n"
             "  raise ParseError.new(\"not an integer: #{s}\")\n"
             "end\n"
             "```",
    },
    {
        "q": "How do I write tests in Crystal?",
        "a": "Crystal has a built-in spec framework — no RSpec, no Minitest. "
             "`require \"spec\"`, then describe/it blocks:\n\n"
             "```crystal\n"
             "require \"spec\"\n"
             "require \"./my_module\"\n\n"
             "describe MyModule do\n"
             "  describe \"#greet\" do\n"
             "    it \"returns a greeting\" do\n"
             "      MyModule.greet(\"World\").should eq \"Hello, World!\"\n"
             "    end\n\n"
             "    it \"raises on empty input\" do\n"
             "      expect_raises(ArgumentError) do\n"
             "        MyModule.greet(\"\")\n"
             "      end\n"
             "    end\n"
             "  end\n"
             "end\n"
             "```\n\n"
             "Run with `crystal spec` from the project root. Spec files conventionally "
             "live in `spec/*_spec.cr`.",
    },
    {
        "q": "How do macros work in Crystal?",
        "a": "Crystal macros run at compile time, take AST nodes as arguments, "
             "and emit code via interpolation in `{{ ... }}` (paste a value) or "
             "`{% ... %}` (control flow). They are NOT runtime metaprogramming — "
             "everything is checked statically:\n\n"
             "```crystal\n"
             "macro define_getters(*names)\n"
             "  {% for name in names %}\n"
             "    def {{name.id}}\n"
             "      @{{name.id}}\n"
             "    end\n"
             "  {% end %}\n"
             "end\n\n"
             "class Point\n"
             "  define_getters x, y\n\n"
             "  def initialize(@x : Int32, @y : Int32); end\n"
             "end\n"
             "```\n\n"
             "Use `macro` for code generation; method dispatch is regular runtime "
             "polymorphism plus compile-time overloading on argument types.",
    },
    {
        "q": "What's the difference between Tuple, NamedTuple, and Hash in Crystal?",
        "a": "Three different things — picking the right one matters for performance:\n\n"
             "- **`Tuple`** — fixed-size, heterogeneous, stack-allocated. Literal: "
             "`{1, \"two\", 3.0}`. Type: `Tuple(Int32, String, Float64)`. Index "
             "access only with literal integers.\n"
             "- **`NamedTuple`** — fixed-size, named fields, stack-allocated. "
             "Literal: `{name: \"Ada\", age: 36}`. Type: "
             "`NamedTuple(name: String, age: Int32)`. Access with `.name` or `[:name]`.\n"
             "- **`Hash`** — dynamic-size, homogeneous keys/values, heap-allocated. "
             "Literal: `{\"name\" => \"Ada\"}`. Type: `Hash(String, String)`. The "
             "general-purpose dictionary.\n\n"
             "```crystal\n"
             "tup     = {1, 2, 3}                  # Tuple(Int32, Int32, Int32)\n"
             "named   = {name: \"Ada\", age: 36}     # NamedTuple\n"
             "hsh     = {\"name\" => \"Ada\"}         # Hash(String, String)\n"
             "```\n\n"
             "`{a: 1}` is a NamedTuple — **not** a Hash. Use `=> ` syntax for Hash.",
    },
    {
        "q": "How do I call a C library from Crystal?",
        "a": "Use a `lib` block to declare the C functions, then call them like "
             "regular Crystal methods. Crystal handles the marshalling:\n\n"
             "```crystal\n"
             "@[Link(\"m\")]   # link against libm\n"
             "lib LibM\n"
             "  fun cos(x : Float64) : Float64\n"
             "  fun sin(x : Float64) : Float64\n"
             "end\n\n"
             "puts LibM.cos(0.0)  # => 1.0\n"
             "puts LibM.sin(0.0)  # => 0.0\n"
             "```\n\n"
             "Use `Pointer(T)` for raw pointers, `Slice(T)` for "
             "`(T*, size)` pairs, `LibC` for libc bindings (already in stdlib). "
             "The `@[Link]` annotation specifies which library to link.",
    },
    {
        "q": "What's idiomatic Crystal for case/pattern-matching?",
        "a": "Use `case ... when`. Crystal supports type matching, value matching, "
             "and ranges. A trailing `else` covers the rest; with no `else`, "
             "Crystal warns if not all cases are covered:\n\n"
             "```crystal\n"
             "def describe(x)\n"
             "  case x\n"
             "  when Int32        then \"integer\"\n"
             "  when String       then \"string of size #{x.size}\"\n"
             "  when 0..9         then \"single digit\"\n"
             "  when nil          then \"nothing\"\n"
             "  else                    \"something else\"\n"
             "  end\n"
             "end\n"
             "```\n\n"
             "Inside a `when Type` branch, `x` is narrowed to that type. Crystal "
             "does *not* have full pattern matching (no destructuring of tuples in "
             "`when` clauses) — for that, assign first, then match.",
    },
    {
        "q": "How do I read a file and parse JSON in Crystal?",
        "a": "Use `File.read` for the contents and `JSON.parse` for an "
             "untyped-ish `JSON::Any`, or a class with `JSON::Serializable` for "
             "typed parsing:\n\n"
             "```crystal\n"
             "require \"json\"\n\n"
             "class Config\n"
             "  include JSON::Serializable\n"
             "  property host : String\n"
             "  property port : Int32\n"
             "end\n\n"
             "json = File.read(\"config.json\")\n"
             "cfg  = Config.from_json(json)\n"
             "puts cfg.host\n\n"
             "# Or untyped:\n"
             "any = JSON.parse(json)\n"
             "puts any[\"host\"].as_s\n"
             "```\n\n"
             "`JSON::Serializable` is the typed, fast path; `JSON.parse` is for "
             "ad-hoc inspection.",
    },
    {
        "q": "How do I create a new Crystal project with shards?",
        "a": "`crystal init` scaffolds a new project and `shards` is the package "
             "manager:\n\n"
             "```bash\n"
             "crystal init app my_app           # creates a runnable app\n"
             "crystal init lib my_lib           # creates a library shard\n"
             "cd my_app\n"
             "shards install                    # install dependencies\n"
             "shards build --release            # build via shards\n"
             "crystal spec                      # run the test suite\n"
             "```\n\n"
             "The `shard.yml` declares dependencies. `lib/` holds installed shards. "
             "Use `shards check` to verify lock consistency.",
    },
    {
        "q": "What are common mistakes when porting Ruby to Crystal?",
        "a": "Crystal looks like Ruby but is statically typed. Common mistakes:\n\n"
             "- `attr_accessor :name` (Ruby) → `property name : String` (Crystal — type required)\n"
             "- `method_missing` — doesn't exist; use macros instead\n"
             "- `eval` — doesn't exist; Crystal compiles\n"
             "- `nil.to_s` works in both but `nil.foo` is a compile error in Crystal\n"
             "- `{a: 1}` is a `NamedTuple` in Crystal, not a `Hash` — use `{\"a\" => 1}` for a hash\n"
             "- `[]` (empty array) needs `[] of T` for a type\n"
             "- Symbol GC — Crystal symbols are atoms enumerated at compile time\n"
             "- `String.scan` returns `Array(Regex::MatchData)`, not `Array(String)`\n"
             "- Mixins via `include`/`extend` work, but the included module's "
             "abstract methods must be implemented for the type-check to pass\n"
             "- Procs are typed: `Proc(Int32, Int32)`, not just \"a callable\"\n\n"
             "When in doubt, `crystal build` early — the compiler catches most "
             "porting bugs before runtime.",
    },
    {
        "q": "How do I write a basic HTTP server in Crystal?",
        "a": "Use `HTTP::Server` from the standard library — no external shard "
             "needed:\n\n"
             "```crystal\n"
             "require \"http/server\"\n\n"
             "server = HTTP::Server.new do |context|\n"
             "  context.response.content_type = \"text/plain\"\n"
             "  context.response.print \"Hello, world!\"\n"
             "end\n\n"
             "address = server.bind_tcp 8080\n"
             "puts \"Listening on http://#{address}\"\n"
             "server.listen\n"
             "```\n\n"
             "The block runs on a fiber per request. For routing, compose multiple "
             "handlers via `HTTP::Handler` or use a shard like Kemal/Lucky for "
             "richer ergonomics.",
    },
    {
        "q": "How do I enforce immutability in Crystal?",
        "a": "Crystal has no `freeze` like Ruby, but several tools encourage "
             "immutability:\n\n"
             "- **`struct`** — value type, copied on assignment, immutable by "
             "default unless you mark fields mutable.\n"
             "- **`getter`** without a `setter` makes a read-only attribute.\n"
             "- **No `setter`/`property=`** means the field can only be assigned "
             "in `initialize`.\n\n"
             "```crystal\n"
             "struct Point\n"
             "  getter x : Float64\n"
             "  getter y : Float64\n\n"
             "  def initialize(@x, @y)\n"
             "  end\n\n"
             "  def +(other : Point) : Point\n"
             "    Point.new(@x + other.x, @y + other.y)\n"
             "  end\n"
             "end\n"
             "```\n\n"
             "`struct` is stack-allocated and copied; prefer it for small value "
             "types. Use `class` for shared, mutable, or polymorphic objects.",
    },
    {
        "q": "What's the Crystal way to do enums?",
        "a": "Crystal enums are first-class types backed by integers, with optional "
             "`@[Flags]` for bitfield enums. Each member is a typed constant:\n\n"
             "```crystal\n"
             "enum Color\n"
             "  Red\n"
             "  Green\n"
             "  Blue\n\n"
             "  def hex : String\n"
             "    case self\n"
             "    when Red    then \"#ff0000\"\n"
             "    when Green  then \"#00ff00\"\n"
             "    when Blue   then \"#0000ff\"\n"
             "    else             \"\"\n"
             "    end\n"
             "  end\n"
             "end\n\n"
             "puts Color::Red.hex   # => \"#ff0000\"\n\n"
             "@[Flags]\n"
             "enum Permission\n"
             "  Read\n"
             "  Write\n"
             "  Execute\n"
             "end\n\n"
             "p = Permission::Read | Permission::Write\n"
             "p.read?   # => true\n"
             "p.execute? # => false\n"
             "```\n\n"
             "Enums can have methods, can carry custom integer values "
             "(`Red = 1`), and the compiler exhaustiveness-checks `case`.",
    },
    {
        "q": "How do I iterate lazily in Crystal?",
        "a": "Use `Iterator` — most enumerable operations have a lazy counterpart "
             "via `.each.lazy` or by calling iterator-returning methods. Lazy "
             "iterators don't allocate intermediate collections:\n\n"
             "```crystal\n"
             "result = (1..Int32::MAX)\n"
             "  .each\n"
             "  .map { |i| i * i }\n"
             "  .select(&.even?)\n"
             "  .first(5)\n"
             "  .to_a\n"
             "# => [4, 16, 36, 64, 100]\n"
             "```\n\n"
             "Calling `.map`/`.select` on a Range or Iterator returns another "
             "Iterator — terminal calls like `.first(n)`, `.to_a`, `.sum`, or "
             "`.each { ... }` actually materialise.",
    },
    {
        "q": "How do I use compile-time conditionals and feature flags in Crystal?",
        "a": "Use `{% if ... %} ... {% end %}` macro blocks for compile-time "
             "branches, and the `crystal build -Dflag` switch for user flags:\n\n"
             "```crystal\n"
             "{% if flag?(:linux) %}\n"
             "  PLATFORM = \"linux\"\n"
             "{% elsif flag?(:darwin) %}\n"
             "  PLATFORM = \"darwin\"\n"
             "{% else %}\n"
             "  PLATFORM = \"unknown\"\n"
             "{% end %}\n\n"
             "{% if flag?(:my_feature) %}\n"
             "  def fancy_path; \"on\"; end\n"
             "{% else %}\n"
             "  def fancy_path; \"off\"; end\n"
             "{% end %}\n"
             "```\n\n"
             "Built-in flags include `:linux`, `:darwin`, `:windows`, `:bits64`, "
             "`:release`, `:debug`. Custom flags: `crystal build -Dmy_feature foo.cr`.",
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Crystal vs other languages: divergence catalogue (hand-written)
# ═══════════════════════════════════════════════════════════════════════

DIVERGENCE_ENTRIES = [
    # (wrong_form, source_language, correct_form, optional example)
    ("attr_accessor :name", "Ruby", "property name : String",
     "class User\n  property name : String\n  def initialize(@name); end\nend"),
    ("attr_reader :id", "Ruby", "getter id : Int64",
     "class User\n  getter id : Int64\n  def initialize(@id); end\nend"),
    ("attr_writer :secret", "Ruby", "setter secret : String",
     "class User\n  setter secret : String\nend"),
    ("require_relative \"./foo\"", "Ruby", "require \"./foo\"",
     "require \"./foo\"\nrequire \"json\""),
    ("describe MyClass do ... end (RSpec)", "Ruby/RSpec",
     "describe MyClass do ... end (Crystal Spec, via require \"spec\")",
     "require \"spec\"\nrequire \"./my_class\"\n\ndescribe MyClass do\n  it \"works\" do\n    MyClass.new.should_not be_nil\n  end\nend"),
    ("Array.new(5)", "Ruby (untyped)", "Array(Int32).new(5, 0)",
     "Array(Int32).new(5, 0) # => [0, 0, 0, 0, 0]\nArray(String).new       # => [] of String"),
    ("[]", "Ruby (untyped empty array)", "[] of T",
     "xs = [] of Int32\nxs << 1\nxs << 2"),
    ("{}", "Ruby (untyped empty hash)", "{} of K => V",
     "h = {} of String => Int32\nh[\"a\"] = 1"),
    ("{a: 1, b: 2} as a Hash", "Ruby", "NamedTuple in Crystal — use {\"a\" => 1, \"b\" => 2} for a Hash",
     "named = {a: 1, b: 2}            # NamedTuple(a: Int32, b: Int32)\nhash  = {\"a\" => 1, \"b\" => 2}    # Hash(String, Int32)"),
    ("method_missing", "Ruby", "macros (compile-time code generation)",
     "macro define_proxy(*names)\n  {% for name in names %}\n    def {{name.id}}; @inner.{{name.id}}; end\n  {% end %}\nend"),
    ("eval(\"...\")", "Ruby", "no equivalent — Crystal compiles; use macros for code generation",
     "macro generate_constant(name, value)\n  {{name.id.upcase}} = {{value}}\nend\n\ngenerate_constant pi, 3.14159"),
    ("def foo(a, b)", "Ruby (untyped)", "def foo(a : Int32, b : Int32) : Int32",
     "def add(a : Int32, b : Int32) : Int32\n  a + b\nend"),
    ("puts foo&.bar&.baz", "Ruby safe-nav", "Try with explicit narrowing or `try`",
     "if (f = foo) && (b = f.bar)\n  puts b.baz\nend\n\n# Or with `try`:\nputs foo.try(&.bar).try(&.baz)"),
    ("Hash#default = something", "Ruby", "Hash.new(default) at construction",
     "h = Hash(String, Int32).new(0)\nh[\"x\"] += 1   # => 1"),
    ("Array#first / Array#last (no arg)", "Ruby (returns nil if empty)",
     "Array#first / #last (raises) — use #first? / #last? for nil-safe",
     "[1, 2, 3].first   # => 1\n([] of Int32).first?   # => nil\n([] of Int32).first    # raises Enumerable::EmptyError"),
    ("def initialize(name)\n  @name = name\nend", "Ruby (verbose)",
     "def initialize(@name : String); end (shorthand)",
     "class User\n  def initialize(@name : String); end\nend"),
    ("nil.foo", "Ruby (NoMethodError at runtime)",
     "compile error in Crystal — explicit nil-check required",
     "if x\n  x.foo  # x is narrowed to non-nil\nend\n\n# Or:\nx.try &.foo"),
    ("Kernel#exit", "Ruby", "exit (top-level method, same name)",
     "exit 0\nexit 1"),
    ("$LOAD_PATH / $:", "Ruby", "no equivalent — use shards / require paths",
     "require \"./local_file\"\nrequire \"some_shard\""),
    ("respond_to?(:method)", "Ruby", "responds_to?(:method) — same idea, compile-time when possible",
     "if obj.responds_to?(:greet)\n  obj.greet\nend"),
    ("Object#freeze", "Ruby", "no equivalent — use struct (value type) or no setters for immutability",
     "struct Point\n  getter x : Int32\n  getter y : Int32\n  def initialize(@x, @y); end\nend"),
    ("class << self", "Ruby (open singleton class)",
     "def self.foo (one method at a time) or wrap in `module SelfMethods; ...`",
     "class Logger\n  def self.global : Logger\n    @@global ||= Logger.new\n  end\nend"),
    ("send(:foo)", "Ruby", "no dynamic dispatch — use macros or explicit dispatch",
     "case action\nwhen \"start\" then start\nwhen \"stop\"  then stop\nelse raise \"unknown action\"\nend"),
    ("Class.new { ... }", "Ruby (anonymous class)",
     "no anonymous classes — define a named class",
     "class MyClass\n  # ...\nend"),
    ("a, b = [1, 2]", "Ruby (multi-assign)", "a, b = 1, 2 (Tuple unpacking)",
     "a, b = 1, 2\nname, age = {\"Ada\", 36}"),
    ("each_with_index { |x, i| ... }", "Ruby", "same — works in Crystal",
     "[10, 20, 30].each_with_index do |x, i|\n  puts \"#{i}: #{x}\"\nend"),
    ("yield", "Ruby", "yield — same, but the method must take a block",
     "def each_double\n  yield 1\n  yield 2\n  yield 3\nend\n\neach_double { |x| puts x * 2 }"),
    ("def foo(&block)", "Ruby", "def foo(&block : Int32 -> String) (typed proc)",
     "def transform(xs : Array(Int32), &block : Int32 -> Int32) : Array(Int32)\n  xs.map { |x| block.call(x) }\nend"),
    ("rescue => e", "Ruby (catch-all)", "rescue ex (or rescue ex : SomeError)",
     "begin\n  risky_call\nrescue ex : IO::Error\n  puts \"io: #{ex.message}\"\nrescue ex\n  puts \"other: #{ex.message}\"\nend"),
    ("Thread.new { ... }", "Ruby/Java", "spawn { ... } (fiber)",
     "spawn do\n  do_work\nend\n\n# For blocking work in a real OS thread, see: WaitGroup, Mutex, ExecutionContext"),
    ("Mutex / synchronize", "Ruby", "Mutex#synchronize — same API",
     "mutex = Mutex.new\nmutex.synchronize do\n  shared_state += 1\nend"),
    ("Queue / SizedQueue", "Ruby", "Channel(T) — typed channels",
     "ch = Channel(Int32).new(buffer_size: 10)\nspawn { ch.send(42) }\nputs ch.receive"),
    ("array.zip(other)", "Ruby", "same — works in Crystal",
     "[1, 2, 3].zip([\"a\", \"b\", \"c\"])\n# => [{1, \"a\"}, {2, \"b\"}, {3, \"c\"}]"),
    ("Kernel#p", "Ruby", "p (top-level — inspects)",
     "p [1, 2, 3]   # prints [1, 2, 3]\np \"hello\"   # prints \"hello\""),
    ("puts nil", "Ruby (prints empty)", "puts nil — prints empty line in Crystal too",
     "puts nil       # prints \"\\n\"\nputs            # also prints \"\\n\""),
    ("String#[start, length]", "Ruby", "String#[start, length] — same; range slicing also works",
     "\"hello\"[1, 3]    # => \"ell\"\n\"hello\"[1..3]    # => \"ell\""),
    ("Regexp#match", "Ruby", "Regex#match (returns Regex::MatchData?)",
     "if md = \"hello world\".match(/(\\w+)\\s+(\\w+)/)\n  puts md[1]   # => hello\n  puts md[2]   # => world\nend"),
    ("Integer#times", "Ruby", "Int#times — same",
     "5.times { |i| puts i }   # 0..4"),
    ("Object#tap", "Ruby", "Object#tap — same",
     "[1, 2, 3].tap { |a| puts a.size }.first   # => 1"),
    ("Object#then / #yield_self", "Ruby 2.6+", "then / itself — `Object#then` works",
     "5.then { |x| x * x }   # => 25"),
]


# ═══════════════════════════════════════════════════════════════════════
# Crystal stdlib doc-comment extraction
# ═══════════════════════════════════════════════════════════════════════

# Matches a definition line: class, module, struct, enum, def, macro, fun, lib, alias, annotation
DEF_RE = re.compile(
    r'^\s*(?:abstract\s+|private\s+|protected\s+)*'
    r'(class|module|struct|enum|def|macro|fun|lib|annotation|alias)'
    r'\s+([\w:]+(?:\s*\([^)]*\))?(?:[!?]?))',
)


def extract_doc_blocks(filepath: str) -> list[dict]:
    """
    Walk a Crystal file and pair leading `# ...` comment blocks with the
    immediately-following def/class/module/struct/enum/macro/fun/lib/alias.

    Returns a list of dicts: {kind, name, signature, doc, file, lineno}.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (OSError, PermissionError):
        return []

    entries: list[dict] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # collect a contiguous comment block (each line starting with `#`)
        if stripped.startswith("#") and not stripped.startswith("#!"):
            start_i = i
            comment_lines: list[str] = []
            while i < n:
                s = lines[i].strip()
                if s.startswith("#"):
                    # Strip leading "# " or "#"
                    body = re.sub(r'^\s*#\s?', '', lines[i].rstrip("\n"))
                    comment_lines.append(body)
                    i += 1
                else:
                    break

            # Skip blank lines
            j = i
            while j < n and not lines[j].strip():
                j += 1

            if j >= n:
                break

            target_line = lines[j].rstrip("\n")
            target_stripped = target_line.lstrip()
            m = DEF_RE.match(target_stripped)
            if not m:
                # Not a definition — discard the comment, advance
                i = j + 1
                continue

            kind = m.group(1)
            name = m.group(2).split('(')[0].rstrip("!?")

            # Capture full signature (handle multi-line for long arg lists)
            sig_lines = [target_stripped]
            paren_depth = target_stripped.count('(') - target_stripped.count(')')
            k = j + 1
            while paren_depth > 0 and k < n:
                sig_lines.append(lines[k].strip())
                paren_depth += lines[k].count('(') - lines[k].count(')')
                k += 1
            sig = ' '.join(sig_lines).rstrip(",")

            # Trim signature: stop at first newline inside the body (e.g., "do" / first stmt)
            # For def/macro/fun, stop at end of arg list and return-type annotation.
            sig = re.sub(r'\s+', ' ', sig)
            sig = sig.split(' #')[0].rstrip()

            doc_text = "\n".join(comment_lines).strip()

            # Only keep entries with a meaningful doc block (3+ words)
            if len(doc_text.split()) < 3:
                i = j + 1
                continue

            # Skip private/protected? (we still want them for completeness, but
            # filter very short ones / internal-looking names)
            if name.startswith('_') or name == 'self':
                i = j + 1
                continue

            entries.append({
                "kind": kind,
                "name": name,
                "signature": sig,
                "doc": doc_text,
                "file": filepath,
                "lineno": j + 1,
            })

            i = j + 1
        else:
            i += 1

    return entries


def crystal_module_for_file(filepath: str) -> str:
    """Best-effort module/class enclosing path for a stdlib file."""
    rel = os.path.relpath(filepath, os.path.join(CRYSTAL_DIR, "src"))
    rel = rel.replace(os.sep, "/")
    # Strip extension
    if rel.endswith(".cr"):
        rel = rel[:-3]
    return rel


def convert_stdlib_docs() -> tuple[list, list]:
    """Mine doc-comments from all crystal/src/**/*.cr files."""
    chatml_entries: list = []
    alpaca_entries: list = []

    src_dir = os.path.join(CRYSTAL_DIR, "src")
    if not os.path.isdir(src_dir):
        return chatml_entries, alpaca_entries

    for filepath in sorted(glob.glob(os.path.join(src_dir, "**", "*.cr"), recursive=True)):
        # Skip vendored / irrelevant subtrees
        rel = os.path.relpath(filepath, src_dir)
        if rel.startswith("compiler/") or rel.startswith("llvm/"):
            # The compiler internals are huge and Crystal-implementation-specific.
            # Skip to keep training focused on user-facing stdlib.
            continue

        entries = extract_doc_blocks(filepath)
        module_path = crystal_module_for_file(filepath)

        for e in entries:
            kind = e["kind"]
            name = e["name"]
            sig = e["signature"]
            doc = e["doc"]

            # Skip purely internal-looking docs (less than 5 words)
            if len(doc.split()) < 5:
                continue

            sid_basis = f"{module_path}::{kind}::{name}::{e['lineno']}"
            sid = f"stdlib:{hashlib.md5(sid_basis.encode()).hexdigest()[:12]}"

            # Variant A: Document the API entry as-is
            q = f"What does Crystal's `{name}` ({kind}) in `{module_path}` do?"
            a_parts = [
                f"`{name}` is a `{kind}` defined in Crystal's stdlib at "
                f"`src/{module_path}.cr` (line {e['lineno']}).",
                f"**Signature:**\n```crystal\n{sig}\n```",
                f"**Documentation:**\n\n{doc}",
            ]
            a = "\n\n".join(a_parts)

            chatml_entries.append(make_chatml([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ], f"{sid}:doc"))
            alpaca_entries.append(make_alpaca(q, a, source_id=f"{sid}:doc"))

            # Variant B: "How do I use X?" if the doc contains an example block
            if "```" in doc:
                q2 = f"Show me how to use `{name}` from Crystal's `{module_path}`."
                a2 = doc + f"\n\n**Defined in** `src/{module_path}.cr` as: `{sig}`"
                chatml_entries.append(make_chatml([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q2},
                    {"role": "assistant", "content": a2},
                ], f"{sid}:howto"))
                alpaca_entries.append(make_alpaca(q2, a2, source_id=f"{sid}:howto"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Crystal samples — runnable example programs
# ═══════════════════════════════════════════════════════════════════════

SAMPLE_DESCRIPTIONS = {
    "2048.cr":              "the 2048 game",
    "binary-trees.cr":      "the binary trees benchmark",
    "brainfuck.cr":         "a Brainfuck interpreter",
    "channel_primes.cr":    "a sieve of Eratosthenes using channels",
    "channel_select.cr":    "Channel.select for multiplexing receives",
    "conway.cr":            "Conway's Game of Life",
    "egrep.cr":             "a tiny grep-like tool reading from stdin",
    "fannkuch-redux.cr":    "the fannkuch-redux benchmark",
    "fibonacci.cr":         "computing Fibonacci numbers",
    "havlak.cr":            "the Havlak loop-recognition benchmark",
    "http_server.cr":       "a basic HTTP server",
    "mandelbrot.cr":        "rendering the Mandelbrot set",
    "matmul.cr":            "matrix multiplication",
    "meteor.cr":            "the meteor puzzle",
    "mt_gc_test.cr":        "a multi-threaded GC stress test",
    "nbodies.cr":           "the n-body benchmark",
    "neural_net.cr":        "a tiny neural network",
    "noise.cr":             "Perlin noise generation",
    "pig.cr":               "the dice game Pig",
    "pretty_json.cr":       "pretty-printing JSON",
    "quine.cr":             "a self-printing program (quine)",
    "red_black_tree.cr":    "a red-black tree",
    "sieve.cr":             "the sieve of Eratosthenes",
    "spectral-norm.cr":     "the spectral-norm benchmark",
    "sudoku.cr":            "a Sudoku solver",
    "tcp_client.cr":        "a TCP client",
    "tcp_server.cr":        "a TCP server",
    "text_raytracer.cr":    "a text-mode ray tracer",
    "tree.cr":              "a tree data structure",
    "wordcount.cr":         "counting words in input",
    "mandelbrot2.cr":       "an alternative Mandelbrot implementation",
    "degree_days.cr":       "computing heating/cooling degree-days",
}


def convert_samples() -> tuple[list, list]:
    chatml_entries: list = []
    alpaca_entries: list = []

    samples_dir = os.path.join(CRYSTAL_DIR, "samples")
    if not os.path.isdir(samples_dir):
        return chatml_entries, alpaca_entries

    for cr_file in sorted(glob.glob(os.path.join(samples_dir, "*.cr"))):
        try:
            with open(cr_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, PermissionError):
            continue

        if len(content) < 30 or len(content) > 30000:
            continue

        basename = os.path.basename(cr_file)
        desc = SAMPLE_DESCRIPTIONS.get(basename, f"the {basename} sample")
        relpath = os.path.relpath(cr_file, CRYSTAL_DIR)
        source_id = f"sample:{basename}"

        q = f"Show me {desc} in Crystal."
        a = (
            f"Here's the implementation from `{relpath}`:\n\n"
            f"```crystal\n{content.rstrip()}\n```"
        )
        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], f"{source_id}:full"))
        alpaca_entries.append(make_alpaca(q, a, source_id=f"{source_id}:full"))

        # Bonus framing: "How do I do X in Crystal?"
        topic = basename.replace(".cr", "").replace("_", " ").replace("-", " ")
        q2 = f"How would I implement {topic} in Crystal?"
        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q2},
            {"role": "assistant", "content": a},
        ], f"{source_id}:howto"))
        alpaca_entries.append(make_alpaca(q2, a, source_id=f"{source_id}:howto"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Crystal spec/std files — usage examples
# ═══════════════════════════════════════════════════════════════════════

def convert_specs() -> tuple[list, list]:
    """Use spec files as 'real-world usage' Q/A pairs."""
    chatml_entries: list = []
    alpaca_entries: list = []

    spec_dir = os.path.join(CRYSTAL_DIR, "spec", "std")
    if not os.path.isdir(spec_dir):
        return chatml_entries, alpaca_entries

    for spec_file in sorted(glob.glob(os.path.join(spec_dir, "**", "*.cr"), recursive=True)):
        try:
            with open(spec_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, PermissionError):
            continue

        # Skip absurdly large or empty
        if len(content) < 200 or len(content) > 25000:
            continue

        rel = os.path.relpath(spec_file, CRYSTAL_DIR)
        # Topic from filename: spec/std/array_spec.cr → "array"
        basename = os.path.basename(spec_file).replace("_spec.cr", "")
        topic = basename.replace("_", " ").replace("/", " ")
        source_id = f"spec:{rel}"

        q = f"Show me usage tests for Crystal's `{topic}` module."
        a = (
            f"Here are spec examples from `{rel}`:\n\n"
            f"```crystal\n{content.rstrip()}\n```"
        )
        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], f"{source_id}:full"))
        alpaca_entries.append(make_alpaca(q, a, source_id=f"{source_id}:full"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Crystal man pages — CLI usage
# ═══════════════════════════════════════════════════════════════════════

def convert_man_pages() -> tuple[list, list]:
    chatml_entries: list = []
    alpaca_entries: list = []

    man_dir = os.path.join(CRYSTAL_DIR, "doc", "man")
    if not os.path.isdir(man_dir):
        return chatml_entries, alpaca_entries

    for adoc in sorted(glob.glob(os.path.join(man_dir, "*.adoc"))):
        try:
            with open(adoc, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, PermissionError):
            continue

        if len(content) < 100:
            continue

        basename = os.path.basename(adoc).replace(".adoc", "")
        # crystal-build → "crystal build"
        cmd_name = basename.replace("-", " ", 1) if basename != "crystal" else "crystal"
        rel = os.path.relpath(adoc, CRYSTAL_DIR)
        source_id = f"man:{basename}"

        q = f"How does the `{cmd_name}` command work in Crystal?"
        a = (
            f"Here's the man page for `{cmd_name}` (from `{rel}`):\n\n"
            f"```\n{content.rstrip()}\n```"
        )
        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], f"{source_id}:full"))
        alpaca_entries.append(make_alpaca(q, a, source_id=f"{source_id}:full"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Crystal changelogs — version-specific release notes
# ═══════════════════════════════════════════════════════════════════════

def split_markdown_sections(content: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in content.split("\n"):
        if line.startswith("#"):
            if current_heading or current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.lstrip("#").strip()
            current_body = []
        else:
            current_body.append(line)

    if current_heading or current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))

    return sections


def convert_changelogs() -> tuple[list, list]:
    chatml_entries: list = []
    alpaca_entries: list = []

    changelogs_dir = os.path.join(CRYSTAL_DIR, "doc", "changelogs")
    if not os.path.isdir(changelogs_dir):
        return chatml_entries, alpaca_entries

    for md_file in sorted(glob.glob(os.path.join(changelogs_dir, "v*.md"))):
        try:
            with open(md_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, PermissionError):
            continue

        if len(content) < 200:
            continue

        version = os.path.basename(md_file).replace(".md", "")
        rel = os.path.relpath(md_file, CRYSTAL_DIR)
        source_id = f"changelog:{version}"

        # Whole-file entry
        q = f"What changed in Crystal {version}?"
        # Truncate aggressively — changelogs can be huge
        if len(content) > 12000:
            content = content[:12000] + "\n\n... (truncated; see full changelog)"
        a = content.rstrip()
        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], f"{source_id}:full"))
        alpaca_entries.append(make_alpaca(q, a, source_id=f"{source_id}:full"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Top-level Crystal docs (README, CONTRIBUTING, UPGRADING, ...)
# ═══════════════════════════════════════════════════════════════════════

TOPLEVEL_DOCS = ["README.md", "CONTRIBUTING.md", "UPGRADING.md", "SECURITY.md", "NOTICE.md"]


def convert_toplevel_docs() -> tuple[list, list]:
    chatml_entries: list = []
    alpaca_entries: list = []

    for doc in TOPLEVEL_DOCS:
        path = os.path.join(CRYSTAL_DIR, doc)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, PermissionError):
            continue

        if len(content) < 100:
            continue

        source_id = f"doc:{doc}"

        # Whole document if reasonably sized
        if len(content) < 8000:
            first_line = content.strip().split("\n")[0]
            doc_title = first_line.lstrip("#").strip() if first_line.startswith("#") else doc
            q = f"Tell me about Crystal's {doc_title}."
            chatml_entries.append(make_chatml([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": content.strip()},
            ], f"{source_id}:full"))
            alpaca_entries.append(make_alpaca(q, content.strip(),
                                              source_id=f"{source_id}:full"))

        # Section-level entries
        for heading, body in split_markdown_sections(content):
            if not heading or not body or len(body) < 100:
                continue
            if heading.lower() in ("table of contents", "contents", "readme"):
                continue
            q = f"Explain '{heading}' in Crystal's {doc}."
            chatml_entries.append(make_chatml([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": body},
            ], f"{source_id}:{heading[:40]}"))
            alpaca_entries.append(make_alpaca(q, body,
                                              source_id=f"{source_id}:{heading[:40]}"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# crystal-mcp source files — real-world Crystal codebase
# ═══════════════════════════════════════════════════════════════════════

def convert_mcp_sources() -> tuple[list, list]:
    chatml_entries: list = []
    alpaca_entries: list = []

    if not os.path.isdir(MCP_DIR):
        return chatml_entries, alpaca_entries

    for cr_file in sorted(glob.glob(os.path.join(MCP_DIR, "src", "**", "*.cr"), recursive=True)):
        try:
            with open(cr_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, PermissionError):
            continue

        if len(content) < 50 or len(content) > 30000:
            continue

        rel = os.path.relpath(cr_file, MCP_DIR)
        source_id = f"mcp-src:{rel}"

        q = f"Show me a real-world Crystal example: `{rel}` from a working MCP server."
        a = (
            f"Here's `{rel}` from the crystal-mcp project — a Model Context "
            f"Protocol server for Crystal:\n\n"
            f"```crystal\n{content.rstrip()}\n```"
        )
        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], f"{source_id}:full"))
        alpaca_entries.append(make_alpaca(q, a, source_id=f"{source_id}:full"))

        # Also extract doc blocks from MCP source the same way as stdlib
        for e in extract_doc_blocks(cr_file):
            doc = e["doc"]
            if len(doc.split()) < 5:
                continue
            kind = e["kind"]
            name = e["name"]
            sig = e["signature"]
            sid_basis = f"{rel}::{kind}::{name}::{e['lineno']}"
            sid = f"mcp-doc:{hashlib.md5(sid_basis.encode()).hexdigest()[:12]}"
            q2 = f"What does `{name}` ({kind}) in `{rel}` (crystal-mcp) do?"
            a2 = (
                f"`{name}` is a `{kind}` in `{rel}`:\n\n"
                f"```crystal\n{sig}\n```\n\n"
                f"{doc}"
            )
            chatml_entries.append(make_chatml([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q2},
                {"role": "assistant", "content": a2},
            ], f"{sid}:doc"))
            alpaca_entries.append(make_alpaca(q2, a2, source_id=f"{sid}:doc"))

    return chatml_entries, alpaca_entries


def convert_mcp_specs() -> tuple[list, list]:
    chatml_entries: list = []
    alpaca_entries: list = []

    if not os.path.isdir(MCP_DIR):
        return chatml_entries, alpaca_entries

    for cr_file in sorted(glob.glob(os.path.join(MCP_DIR, "spec", "**", "*.cr"), recursive=True)):
        try:
            with open(cr_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, PermissionError):
            continue

        if len(content) < 100 or len(content) > 25000:
            continue

        rel = os.path.relpath(cr_file, MCP_DIR)
        basename = os.path.basename(cr_file).replace("_spec.cr", "").replace(".cr", "")
        topic = basename.replace("_", " ")
        source_id = f"mcp-spec:{rel}"

        q = f"Show me a real Crystal spec file for testing {topic}."
        a = (
            f"Here's `{rel}` from the crystal-mcp project:\n\n"
            f"```crystal\n{content.rstrip()}\n```"
        )
        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], f"{source_id}:full"))
        alpaca_entries.append(make_alpaca(q, a, source_id=f"{source_id}:full"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Convention examples (3x weighted)
# ═══════════════════════════════════════════════════════════════════════

def generate_convention_entries() -> tuple[list, list]:
    chatml_entries: list = []
    alpaca_entries: list = []

    for i, ex in enumerate(CONVENTION_EXAMPLES):
        for rep in range(3):  # 3x weighting
            sid = f"convention:crystal-idiom-{i}:rep{rep}"
            chatml_entries.append(make_chatml([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["q"]},
                {"role": "assistant", "content": ex["a"]},
            ], sid))
            alpaca_entries.append(make_alpaca(ex["q"], ex["a"], source_id=sid))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Divergence (Crystal vs Ruby/other) — 2x weighted
# ═══════════════════════════════════════════════════════════════════════

def generate_divergence_entries() -> tuple[list, list]:
    chatml_entries: list = []
    alpaca_entries: list = []

    for i, (wrong, src_lang, correct, example) in enumerate(DIVERGENCE_ENTRIES):
        sid_base = f"divergence:crystal:{i}"

        # Variant 1: "Can I use X in Crystal?"
        q1 = f"In Crystal, can I use `{wrong}` (from {src_lang})?"
        a1_parts = [f"No (or only partially). In Crystal, the idiomatic form is: `{correct}`."]
        if example:
            a1_parts.append(f"**Example:**\n```crystal\n{example}\n```")
        a1 = "\n\n".join(a1_parts)

        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
        ], f"{sid_base}:can-i"))
        alpaca_entries.append(make_alpaca(q1, a1, source_id=f"{sid_base}:can-i"))

        # Variant 2: "What's the Crystal equivalent of X?"
        q2 = f"What's the Crystal equivalent of `{wrong}` from {src_lang}?"
        a2_parts = [f"Use `{correct}`."]
        if example:
            a2_parts.append(f"```crystal\n{example}\n```")
        a2 = "\n\n".join(a2_parts)

        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q2},
            {"role": "assistant", "content": a2},
        ], f"{sid_base}:equiv"))
        alpaca_entries.append(make_alpaca(q2, a2, source_id=f"{sid_base}:equiv"))

        # 2x weight: emit again with slightly different phrasing
        q3 = f"I wrote `{wrong}` (familiar from {src_lang}) — what's wrong in Crystal?"
        a3 = (
            f"That form doesn't exist or behaves differently in Crystal. "
            f"Use `{correct}` instead."
        )
        if example:
            a3 += f"\n\n**Crystal version:**\n```crystal\n{example}\n```"
        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q3},
            {"role": "assistant", "content": a3},
        ], f"{sid_base}:fix"))
        alpaca_entries.append(make_alpaca(q3, a3, source_id=f"{sid_base}:fix"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Helpers / output
# ═══════════════════════════════════════════════════════════════════════

def make_chatml(conversations: list[dict], source_id: str) -> dict:
    return {"conversations": conversations, "source": source_id}


def make_alpaca(instruction: str, output: str, inp: str = "", source_id: str = "") -> dict:
    return {
        "instruction": instruction,
        "input": inp,
        "output": output,
        "source": source_id,
    }


def deduplicate(entries: list[dict], key_field: str = "source") -> list[dict]:
    seen: set = set()
    deduped: list[dict] = []
    for entry in entries:
        sid = entry.get(key_field, "")
        if sid and sid not in seen:
            seen.add(sid)
            deduped.append(entry)
        elif not sid:
            content_hash = hashlib.md5(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                deduped.append(entry)
    return deduped


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chatml: list = []
    all_alpaca: list = []

    sources = [
        ("convention examples (3x)",      generate_convention_entries),
        ("divergence (Crystal vs Ruby)",  generate_divergence_entries),
        ("crystal/src stdlib doc-comments", convert_stdlib_docs),
        ("crystal/samples",               convert_samples),
        ("crystal/spec/std",              convert_specs),
        ("crystal/doc/man",               convert_man_pages),
        ("crystal/doc/changelogs",        convert_changelogs),
        ("crystal top-level docs",        convert_toplevel_docs),
        ("crystal-mcp/src",               convert_mcp_sources),
        ("crystal-mcp/spec",              convert_mcp_specs),
    ]

    for label, fn in sources:
        print(f"Converting {label} ...")
        try:
            c, a = fn()
        except FileNotFoundError as e:
            print(f"  ! skipped: {e}")
            continue
        all_chatml.extend(c)
        all_alpaca.extend(a)
        print(f"  → {len(c)} ChatML, {len(a)} Alpaca entries")

    print(f"\nTotal before dedup: {len(all_chatml)} ChatML, {len(all_alpaca)} Alpaca")
    all_chatml = deduplicate(all_chatml, "source")
    all_alpaca = deduplicate(all_alpaca, "source")
    print(f"Total after dedup:  {len(all_chatml)} ChatML, {len(all_alpaca)} Alpaca")

    chatml_path = os.path.join(OUTPUT_DIR, "training_data.jsonl")
    with open(chatml_path, "w") as f:
        for entry in all_chatml:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nWrote {chatml_path}")

    together_path = os.path.join(OUTPUT_DIR, "training_data_together.jsonl")
    with open(together_path, "w") as f:
        for entry in all_chatml:
            together_entry = {"messages": entry["conversations"]}
            f.write(json.dumps(together_entry, ensure_ascii=False) + "\n")
    print(f"Wrote {together_path}")

    alpaca_path = os.path.join(OUTPUT_DIR, "training_data_alpaca.jsonl")
    with open(alpaca_path, "w") as f:
        for entry in all_alpaca:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {alpaca_path}")

    alpaca_json_path = os.path.join(OUTPUT_DIR, "training_data_alpaca.json")
    with open(alpaca_json_path, "w") as f:
        json.dump(all_alpaca, f, ensure_ascii=False, indent=2)
    print(f"Wrote {alpaca_json_path}")

    print("\n── Stats by source category ──")
    source_counts: dict = {}
    for entry in all_chatml:
        src = entry.get("source", "unknown")
        category = src.split(":")[0]
        source_counts[category] = source_counts.get(category, 0) + 1
    for cat, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    chatml_size = os.path.getsize(chatml_path)
    together_size = os.path.getsize(together_path)
    alpaca_size = os.path.getsize(alpaca_path)
    print(f"\nFile sizes: ChatML={chatml_size/1024/1024:.1f}MB, "
          f"Together={together_size/1024/1024:.1f}MB, "
          f"Alpaca={alpaca_size/1024/1024:.1f}MB")


if __name__ == "__main__":
    main()
