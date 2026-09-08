#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Installs what the base image lacks for building, testing and linting cqr.
# The image already ships CMake, Ninja, g++, clang, clang-format and
# clang-tidy.
#
#   libmkl-dev     Intel MKL (see the Prerequisites section of README.md):
#                  the distro package, since Intel's own apt repo
#                  (apt.repos.intel.com) is blocked by the web network policy.
#   libomp-dev     clang's OpenMP runtime and its omp.h, which clang-tidy
#                  needs (.pre-commit-config.yaml explains).
#   pre-commit     the formatting and linting driver, with its hook
#                  environments built up front so the first edit does not
#                  pay for it (.claude/hooks/format.sh runs it after every
#                  edit).
set -euo pipefail

# Async: let the session start while the packages install in the background.
echo '{"async": true, "asyncTimeout": 600000}'

# Only manage dependencies in the remote (web) environment; leave local
# machines (which may already have a oneAPI MKL install) untouched.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive

# `sudo` if we are not root, otherwise run apt directly.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Idempotent: probe each package so a re-run on a warm container installs
# nothing. clang-format and clang-tidy are probed too, in case an image
# without them ever shows up.
missing=()
[ -f /usr/include/mkl/mkl_compact.h ] || missing+=(libmkl-dev)
dpkg -s libomp-dev >/dev/null 2>&1 || missing+=(libomp-dev)
command -v clang-format >/dev/null 2>&1 || missing+=(clang-format)
command -v clang-tidy >/dev/null 2>&1 || missing+=(clang-tidy)

if [ ${#missing[@]} -eq 0 ]; then
  echo "session-start: packages already present, nothing to install"
else
  echo "session-start: installing ${missing[*]}"
  # Third-party sources in the base image are unreachable through the
  # session proxy and make `update` exit non-zero. The Ubuntu archive
  # itself refreshes fine, and that is where these packages come from, so
  # a partial update is not fatal here; the install is what has to succeed.
  $SUDO apt-get update -qq || echo "session-start: apt-get update reported errors, continuing"
  $SUDO apt-get install -y --no-install-recommends "${missing[@]}"
fi

# pre-commit and its hook environments (clang-format at the pinned version,
# the whitespace checks). Not fatal: format.sh falls back to the system
# clang-format when pre-commit is missing.
if command -v pre-commit >/dev/null 2>&1 || python3 -m pip install --quiet pre-commit; then
  if pre-commit install-hooks >/dev/null 2>&1; then
    echo "session-start: pre-commit hook environments ready"
  else
    echo "session-start: pre-commit install-hooks failed; format.sh will fall back to clang-format"
  fi
else
  echo "session-start: pip install pre-commit failed; format.sh will fall back to clang-format"
fi

echo "session-start: build with 'cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq && cmake --build build && ctest --test-dir build'"
echo "session-start: 'pre-commit run --all-files' checks the style; see AGENTS.md"
echo "session-start: for clang-tidy, 'CXX=clang++ cmake -S . -B build-tidy -DBLA_VENDOR=Intel10_64lp_seq' then 'pre-commit run --hook-stage manual clang-tidy --all-files'"
