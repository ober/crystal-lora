#!/usr/bin/env bash
# Fetch every upstream source the v3 build_*.py scripts read.
#
# Idempotent — re-running skips repos that are already cloned. Pass --force
# to nuke and re-clone. Pass --skip-stdlib if you've already got a local
# Crystal stdlib checkout under ~/mine/crystal.
#
# After this finishes you can run:
#   python3 build_cpt_corpus_v2.py --limit 500 --keep-clones
#   python3 build_cpt_docs.py
#   python3 merge_cpt_v3.py
#   python3 build_sft_v3.py
#   python3 build_sft_llm.py --files 3000      (needs OPENROUTER_API_KEY)
#   python3 build_dpo_pairs_v3.py
#
# Disk usage when finished:
#   ~/mine/crystal             ~120 MB  (Crystal stdlib + compiler source)
#   /tmp/crystal-book          ~15 MB
#   /tmp/crystal-lang_*        ~10 MB
#   /tmp/veelenga_*            ~2 MB
#   crystal_corpus_raw/        ~3-5 GB AFTER you run build_cpt_corpus_v2.py --keep-clones
#                              (500 shallow-cloned repos)

set -euo pipefail

FORCE=0
SKIP_STDLIB=0
for arg in "$@"; do
  case "$arg" in
    --force)        FORCE=1 ;;
    --skip-stdlib)  SKIP_STDLIB=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: \`$1\` not found in PATH. Install: $2" >&2
    exit 1
  }
}

require git    "https://git-scm.com/downloads"
require gh     "https://cli.github.com/  (only needed by build_cpt_corpus_v2.py later)"
require python3 "system package manager"
command -v crystal >/dev/null 2>&1 || \
  echo "WARN: \`crystal\` compiler not in PATH. SFT/DPO compile-gates will skip-validate. Install from https://crystal-lang.org/install/"

clone() {
  # clone <url> <dest>
  local url="$1" dest="$2"
  if [[ -d "$dest/.git" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      echo "  [force] removing $dest"
      rm -rf "$dest"
    else
      echo "  [skip] $dest already exists"
      return 0
    fi
  fi
  echo "  cloning $url → $dest"
  git clone --depth=1 --quiet "$url" "$dest"
}

echo "=== 1. Crystal stdlib (used by build_sft_v3.py + build_cpt_docs.py) ==="
if [[ "$SKIP_STDLIB" == "1" ]]; then
  echo "  --skip-stdlib set; expecting ~/mine/crystal to already exist"
  test -d "$HOME/mine/crystal/src" || {
    echo "  ERROR: $HOME/mine/crystal/src not found, drop --skip-stdlib" >&2
    exit 1
  }
else
  mkdir -p "$HOME/mine"
  clone "https://github.com/crystal-lang/crystal.git" "$HOME/mine/crystal"
fi

echo
echo "=== 2. Doc corpora (used by build_cpt_docs.py) ==="
clone "https://github.com/crystal-lang/crystal-book.git"      "/tmp/crystal-book"
clone "https://github.com/crystal-lang/crystal-website.git"   "/tmp/crystal-lang_crystal-website"
clone "https://github.com/crystal-lang/rfcs.git"              "/tmp/crystal-lang_rfcs"
clone "https://github.com/veelenga/awesome-crystal.git"       "/tmp/veelenga_awesome-crystal"

echo
echo "=== 3. (optional) crystal-mcp — read by build_cpt_corpus_v2.py local pass ==="
if [[ -d "$HOME/mine/crystal-mcp" ]]; then
  echo "  [skip] $HOME/mine/crystal-mcp already exists"
else
  echo "  not cloning (private repo / not needed) — build_cpt_corpus_v2.py will print a 'skipping' notice and continue"
fi

echo
echo "=== 4. (informational) gh auth — required by build_cpt_corpus_v2.py ==="
if gh auth status >/dev/null 2>&1; then
  echo "  gh authenticated OK"
else
  echo "  WARN: \`gh auth login\` not done. \`build_cpt_corpus_v2.py\` will fail at the \`gh search repos\` call."
fi

echo
echo "=== Done. Next steps ==="
cat <<'EOF'

  # 1. Build CPT corpus (~30-60 min; clones 500 repos to crystal_corpus_raw/)
  python3 build_cpt_corpus_v2.py --out cpt_corpus_v3.jsonl --limit 500 --keep-clones

  # 2. Build doc corpus (~30 sec)
  python3 build_cpt_docs.py

  # 3. Merge + dedup CPT
  python3 merge_cpt_v3.py    # → cpt_corpus_v3_merged.jsonl

  # 4. Mine SFT pairs (parallel compile-gate; needs `crystal` compiler; ~5 min)
  python3 build_sft_v3.py    # → sft_v3_mined.jsonl

  # 5. (optional) LLM-augment SFT via Claude Haiku 4.5 — needs OPENROUTER_API_KEY (~$15-20 of API)
  python3 build_sft_llm.py --files 3000 --out sft_v3_llm.jsonl

  # 6. Build DPO pairs (programmatic; ~10 sec)
  python3 build_dpo_pairs_v3.py    # → dpo_pairs_v3.jsonl

  # 7. Train on RunPod — see REPRODUCE.md for the pod orchestration details
  python3 runpod_train.py up && python3 runpod_train.py push && python3 runpod_train.py train

EOF
