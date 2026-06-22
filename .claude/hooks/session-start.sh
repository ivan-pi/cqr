#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Installs the one missing prerequisite for building/testing cqr: Intel MKL.
# The base image already ships CMake, g++/gcc and GNU Make.
#
# cqr requires Intel MKL (see the Prerequisites section of README.md); the
# distro `libmkl-dev` package supplies it. We use the distro package rather
# than oneAPI because Intel's own apt repo (apt.repos.intel.com) is blocked
# by the web network policy.
set -euo pipefail

# Async: let the session start while MKL installs in the background.
echo '{"async": true, "asyncTimeout": 600000}'

# Only manage dependencies in the remote (web) environment; leave local
# machines (which may already have a oneAPI MKL install) untouched.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Idempotent: skip the apt work if the compact header is already present.
if [ -f /usr/include/mkl/mkl_compact.h ]; then
  echo "Intel MKL already present; nothing to do."
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive

# `sudo` if we are not root, otherwise run apt directly.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends libmkl-dev

echo "Intel MKL installed (libmkl-dev)."
