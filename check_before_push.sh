#!/usr/bin/env bash
# Run from the repo root before every push. Needs git. Exit code 1 if something looks risky.
set -u
bad=0
hdr() { printf '\n== %s ==\n' "$1"; }

hdr "1. secret files tracked by git (should print nothing)"
if git ls-files | grep -E '(^|/)\.env($|\.)' | grep -v '\.env\.example$'; then bad=1; fi

hdr "2. files containing secret-looking strings (should print nothing; only file names are shown)"
if git grep -I -l -E 'sk-or-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|Bearer [A-Za-z0-9._-]{20,}|(api[_-]?key|token|secret|password)[[:space:]]*[=:][[:space:]]*["'"'"']?[A-Za-z0-9_-]{16,}' -- . ':!.env.example'; then bad=1; fi

hdr "3. secrets anywhere in git history (should print nothing)"
PAT='sk-or-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}'
if [ -n "$(git log --all --oneline -G"$PAT" -- . ':!.env.example' 2>/dev/null | head -1)" ]; then
  git log --all --oneline -G"$PAT" -- . ':!.env.example' | head; bad=1
fi
if git log --all --name-only --pretty=format: 2>/dev/null | grep -E '(^|/)\.env$' | head -1 | grep -q .; then
  echo "!! a .env file exists in the history"; bad=1
fi

hdr "4. personal absolute paths or e-mail addresses (review manually)"
git grep -n -I -E '/Users/[A-Za-z0-9_.-]+|C:\\Users|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' -- . ':!.env.example' || echo "none"

hdr "5. files larger than 5 MB (review)"
git ls-files -z | xargs -0 -I{} du -k {} 2>/dev/null | awk '$1>5120' || true

hdr "6. ignore rules work (each line should show a matching rule)"
git check-ignore -v .env results/runs_backup_test.jsonl 2>/dev/null || echo "!! .env is NOT ignored"

echo
if [ "$bad" -eq 0 ]; then echo "OK: nothing critical found."; else echo "STOP: fix the items above before pushing."; exit 1; fi
