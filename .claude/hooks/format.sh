#!/usr/bin/env bash
#
# PostToolUse hook: format and lint the file Claude just wrote.
#
# Runs the hooks of .pre-commit-config.yaml on that one file, so that
# Claude's edits come out in the same shape as a commit would, with the
# same pinned tool versions. clang-format rewrites the file in place and
# the hook is silent about it; a complaint the tools cannot fix
# themselves is fed back to Claude, which then fixes it.
#
# Without pre-commit, clang-format is tried directly with whatever
# version is installed; without that, the hook says so once and steps
# aside. Either way the edit itself has already happened and is never
# undone.

set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

file=$(python3 -c 'import json, sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))' 2>/dev/null)
[ -n "$file" ] && [ -f "$file" ] || exit 0

# Only files inside the work tree.
case "$file" in
    "$PWD"/*) rel=${file#"$PWD"/} ;;
    /*) exit 0 ;;
    *) rel=$file ;;
esac

if command -v pre-commit >/dev/null 2>&1 && [ -f .pre-commit-config.yaml ]; then
    # First pass: the formatters rewrite the file (exit 1 when they do).
    # Second pass: only what they could not fix is left, and that is
    # what Claude should hear about.
    if ! out=$(pre-commit run --files "$rel" 2>&1); then
        if ! out=$(pre-commit run --files "$rel" 2>&1); then
            printf '%s\n' "$out" | grep -vE '^\S.*(Passed|Skipped)$' >&2
            echo "format hook: pre-commit still fails on $rel; fix the findings above" >&2
            exit 2
        fi
    fi
    exit 0
fi

# Fallback without pre-commit: bare clang-format, if installed.
case "$rel" in
    *.cpp|*.cxx|*.cc|*.h|*.hpp)
        if command -v clang-format >/dev/null 2>&1; then
            clang-format -i "$rel"
        else
            echo "format hook: clang-format not installed; 'pip install pre-commit' and run 'pre-commit install-hooks'" >&2
        fi
        ;;
esac
exit 0
