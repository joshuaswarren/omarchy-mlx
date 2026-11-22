#!/bin/sh
# Compile and run the honeykrisp identity gate tests. Dependency: a C++17
# compiler (g++ or clang++). Exits nonzero on the first failing expectation.
set -eu
cd "$(dirname "$0")"
CXX=${CXX:-g++}
OUT=$(mktemp /tmp/omarchy-identity-test.XXXXXX)
trap 'rm -f "$OUT"' EXIT
$CXX -std=c++17 -O0 -Wall -Wextra -o "$OUT" honeykrisp_identity_test.cpp
"$OUT"
